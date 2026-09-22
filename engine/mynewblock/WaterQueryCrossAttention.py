import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
# ==========================
# Water Token Encoder
# ==========================

WATER_FEATURES = ('temperature', 'do', 'ph', 'turbidity')
WATER_MEANS = (27.878, 4.547, 8.069, 0.125)
WATER_STDS = (0.673, 1.939, 0.229, 0.914)


def validate_water_features(features=None):
    features = list(WATER_FEATURES if features is None else features)
    if not features or len(set(features)) != len(features):
        raise ValueError('water_features must be nonempty and unique')
    if any(name not in WATER_FEATURES for name in features):
        raise ValueError(f'water_features must be chosen from {WATER_FEATURES}')
    return features

class WaterTokenEncoder(nn.Module):
    def __init__(self, num_params=None, embed_dim=64, num_heads=4,
                 means=WATER_MEANS, stds=WATER_STDS, dropout=0.0,
                 water_features=None, water_self_attn=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.water_features = validate_water_features(water_features)
        self.num_params = len(self.water_features)
        if num_params is not None and num_params != self.num_params:
            raise ValueError('num_params must equal len(water_features)')
        indices = [WATER_FEATURES.index(name) for name in self.water_features]
        means = torch.as_tensor(means, dtype=torch.float32)
        stds = torch.as_tensor(stds, dtype=torch.float32)
        if means.shape != (4,) or stds.shape != (4,):
            raise ValueError('water_means/water_stds require four values in raw sensor order')
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError('water statistics must be finite, with strictly positive stds')
        self.register_buffer('feature_indices', torch.tensor(indices), persistent=False)
        self.use_water_self_attn = bool(water_self_attn)
        self.save_attention = False
        self.last_water_self_attn = None
        self.value_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),      # 去掉 inplace
            nn.Linear(16, embed_dim)
        )
        self.param_embedding = nn.Parameter(torch.empty(self.num_params, embed_dim))
        nn.init.trunc_normal_(self.param_embedding, std=0.02)
        self.norm = nn.LayerNorm(embed_dim) if water_self_attn else None
        self.water_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) if water_self_attn else None
        # 简化的 delta_proj，去掉冗余 LayerNorm
        self.delta_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ) if water_self_attn else None
        self.register_parameter('interaction_scale', nn.Parameter(torch.zeros(1)) if water_self_attn else None)
        self.register_buffer("mean", means[indices].clone())
        self.register_buffer("std", stds[indices].clone())

    def select_features(self, water):
        """Dataset input is always raw [B,4]; compact [B,N<4] follows water_features."""
        if water.ndim != 2:
            raise ValueError('water must have shape [B,4] or [B,len(water_features)]')
        if water.shape[1] == len(WATER_FEATURES):
            return water.index_select(1, self.feature_indices)
        if water.shape[1] != self.num_params:
            raise ValueError(f'Expected 4 raw or {self.num_params} selected water values')
        return water

    def forward(self, water):
        self.last_water_self_attn = None
        water = self.select_features(water).float()
        water = (water - self.mean) / self.std
        water = torch.nan_to_num(water, nan=0.0)  # 缺失值：标准化后取0，等价于按均值填充
        water = torch.clamp(water, -5.0, 5.0)
        water = water.unsqueeze(-1)                    # [B,N,1]
        value_tokens = self.value_encoder(water)       # [B,N,64]
        param_tokens = self.param_embedding.unsqueeze(0)
        water_tokens = value_tokens + param_tokens     # [B,N,64]

        if self.use_water_self_attn:
            x = self.norm(water_tokens)
            attn_out, weights = self.water_self_attn(x, x, x, need_weights=self.save_attention)
            if self.save_attention:
                self.last_water_self_attn = weights.detach().cpu()
            delta = self.delta_proj(attn_out)
            water_tokens = water_tokens + self.interaction_scale * delta
        return water_tokens
    



# class WaterQueryModulator(nn.Module):
#     def __init__(
#         self,
#         water_dim=64,
#         query_dim=256,
#         hidden_dim=128,
#         scale=0.1
#     ):
#         super().__init__()

#         self.scale = scale

#         # -------------------------
#         # water token importance pooling
#         # -------------------------
#         self.token_score = nn.Sequential(
#             nn.LayerNorm(water_dim),
#             nn.Linear(water_dim, water_dim),
#             nn.SiLU(),
#             nn.Linear(water_dim, 1)
#         )

#         # -------------------------
#         # water prior projection
#         # -------------------------
#         self.prior_proj = nn.Sequential(
#             nn.LayerNorm(water_dim),
#             nn.Linear(water_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU()
#         )

#         # -------------------------
#         # generate gamma and beta
#         # -------------------------
#         self.to_gamma_beta = nn.Linear(
#             hidden_dim,
#             query_dim * 2
#         )

#         # 初始化为 0，保证一开始不影响原始 query
#         nn.init.zeros_(self.to_gamma_beta.weight)
#         nn.init.zeros_(self.to_gamma_beta.bias)

#     def forward(self, query, water_tokens):
#         """
#         query:
#             [B, num_queries, query_dim]

#         water_tokens:
#             [B, 4, water_dim]

#         return:
#             query_mod:
#                 [B, num_queries, query_dim]

#             attn_weights:
#                 [B, 4]
#         """

#         # -------------------------
#         # 1. token importance aggregation
#         # -------------------------

#         # [B,4,1]
#         score = self.token_score(water_tokens)

#         # [B,4,1]
#         attn_weights = F.softmax(score, dim=1)

#         # [B,water_dim]
#         water_prior = (
#             attn_weights * water_tokens
#         ).sum(dim=1)

#         # -------------------------
#         # 2. generate gamma and beta
#         # -------------------------

#         # [B,hidden_dim]
#         water_prior = self.prior_proj(water_prior)

#         # [B,2*query_dim]
#         gamma_beta = self.to_gamma_beta(water_prior)

#         # [B,query_dim], [B,query_dim]
#         gamma, beta = gamma_beta.chunk(2, dim=-1)

#         # 限制调制幅度，防止训练初期 query 被破坏
#         gamma = torch.tanh(gamma) * self.scale
#         beta = beta * self.scale

#         # [B,1,query_dim]
#         gamma = gamma.unsqueeze(1)
#         beta = beta.unsqueeze(1)

#         # -------------------------
#         # 3. dynamic query modulation
#         # -------------------------

#         query_mod = query * (1.0 + gamma) + beta

#         return query_mod, attn_weights.squeeze(-1)
    
    
    
class WaterAwareQueryCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim=112,
        water_dim=64,
        water_heads=4,
        num_heads=8,
        dropout=0.0,
        ffn_ratio=4.0,
        init_scale=0.0,
        water_features=None,
        water_means=WATER_MEANS,
        water_stds=WATER_STDS,
        water_self_attn=True,
        learnable_gate=True,
    ):
        super().__init__()

        assert query_dim % num_heads == 0

        self.query_norm = nn.LayerNorm(query_dim)
        self.water_norm = nn.LayerNorm(water_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=water_dim,
            vdim=water_dim
        )

        hidden_dim = int(query_dim * ffn_ratio)

        self.ffn = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.learnable_gate = bool(learnable_gate)
        self.save_attention = False
        self.last_query_water_attn = None
        self.attn_scale = nn.Parameter(
            torch.zeros(1) + init_scale
        ) if learnable_gate else 1.0

        self.ffn_scale = nn.Parameter(
            torch.zeros(1) + init_scale
        ) if learnable_gate else 1.0

        self.water_encoder = WaterTokenEncoder(
            water_features=water_features,
            water_self_attn=water_self_attn,
            means=water_means,
            stds=water_stds,
            embed_dim=water_dim,
            num_heads=water_heads,
            dropout=dropout
        )

    def forward(self, query, water, return_attn=False):
        """
        query:
            [B, num_queries, query_dim]

        water:
            [B,4] raw sensor columns or [B,N<4] selected columns

        return:
            [B, num_queries, query_dim]
        """
        self.last_query_water_attn = None
        self.water_encoder.save_attention = self.save_attention
        water_tokens = self.water_encoder(water)
        q = self.query_norm(query)
        w = self.water_norm(water_tokens)

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=w,
            value=w,
            need_weights=return_attn or self.save_attention,
            average_attn_weights=True
        )

        if self.save_attention:
            self.last_query_water_attn = attn_weights.detach().cpu()

        query = query + self.attn_scale * attn_out

        ffn_out = self.ffn(query)

        query = query + self.ffn_scale * ffn_out

        if return_attn:
            return query, attn_weights

        return query
