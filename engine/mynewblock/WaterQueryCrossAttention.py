import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
# ==========================
# Water Token Encoder
# ==========================

class WaterTokenEncoder(nn.Module):
    def __init__(self, num_params=4, embed_dim=64, num_heads=4,
                 means = [27.878, 4.547, 8.069, 0.125], stds  = [0.673, 1.939, 0.229, 0.914], dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.value_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),      # 去掉 inplace
            nn.Linear(16, embed_dim)
        )
        self.param_embedding = nn.Parameter(torch.empty(num_params, embed_dim))
        nn.init.trunc_normal_(self.param_embedding, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.water_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 简化的 delta_proj，去掉冗余 LayerNorm
        self.delta_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.interaction_scale = nn.Parameter(torch.zeros(1))
        self.register_buffer("mean", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(stds, dtype=torch.float32))

    def forward(self, water):
        water = water.float()
        water = (water - self.mean) / self.std
        water = torch.nan_to_num(water, nan=0.0)  # 缺失值：标准化后取0，等价于按均值填充
        water = torch.clamp(water, -5.0, 5.0)
        water = water.unsqueeze(-1)                    # [B,4,1]
        value_tokens = self.value_encoder(water)       # [B,4,64]
        param_tokens = self.param_embedding.unsqueeze(0)
        water_tokens = value_tokens + param_tokens     # [B,4,64]

        x = self.norm(water_tokens)
        attn_out, _ = self.water_self_attn(x, x, x, need_weights=False)
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
        init_scale=0.0
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

        self.attn_scale = nn.Parameter(
            torch.zeros(1) + init_scale
        )

        self.ffn_scale = nn.Parameter(
            torch.zeros(1) + init_scale
        )

        self.water_encoder = WaterTokenEncoder(
            num_params=4,
            embed_dim=water_dim,
            num_heads=water_heads,
            dropout=dropout
        )

    def forward(self, query, water, return_attn=False):
        """
        query:
            [B, num_queries, query_dim]

        water_tokens:
            [B, 4, water_dim]

        return:
            [B, num_queries, query_dim]
        """
        water_tokens = self.water_encoder(water)
        q = self.query_norm(query)
        w = self.water_norm(water_tokens)

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=w,
            value=w,
            need_weights=return_attn,
            average_attn_weights=True
        )

        query = query + self.attn_scale * attn_out

        ffn_out = self.ffn(query)

        query = query + self.ffn_scale * ffn_out

        if return_attn:
            return query, attn_weights

        return query
