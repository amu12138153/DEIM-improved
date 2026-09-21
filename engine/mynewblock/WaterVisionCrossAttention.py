import torch
import torch.nn as nn
from ..core import register

# ==========================
# Water Token Encoder
# ==========================
@register()
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


# ==========================
# Water Guided Cross Attention
# ==========================
@register()
class WaterCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels=512,
        embed_dim=64,
        num_heads=4,
        ffn_ratio=4,
        dropout=0.0,
        init_scale=0.0
    ):
        super().__init__()

        self.visual_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=1
        )

        self.water_encoder = WaterTokenEncoder(
            num_params=4,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=dropout
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        hidden_dim = int(embed_dim * ffn_ratio)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.visual_norm = nn.LayerNorm(embed_dim)
        self.water_norm = nn.LayerNorm(embed_dim)
        
        self.visual_recover = nn.Conv2d(
            embed_dim,
            in_channels,
            kernel_size=1
        )
#       self.attn_scale = nn.Parameter(
#           torch.zeros(1) + init_scale
#       )
#
#       self.ffn_scale = nn.Parameter(
#           torch.zeros(1) + init_scale
#       )
        # 新增：残差融合强度，初始化为 0
#       self.gamma = nn.Parameter(torch.zeros(1))    
        self.attn_scale = nn.Parameter(torch.ones(1) * init_scale)
        self.ffn_scale = nn.Parameter(torch.ones(1) * init_scale)
        self.gamma = nn.Parameter(torch.ones(1) * init_scale)        
        
        
    def forward(self, visual_feat, water,return_attn=False):
        """
        visual_feat:
            [B,512,40,40]

        water:
            [B,4]

        return:
            [B,512,40,40]
        """

        # 保存原始视觉特征
        identity = visual_feat
        water_tokens = self.water_encoder(water)

        # 512 -> 64
        feat = self.visual_proj(visual_feat)

        B, C, H, W = feat.shape

        # [B,64,H,W] -> [B,H*W,64]
        visual_tokens = (
            feat
            .flatten(2)
            .transpose(1, 2)
        )

        # [B,4,64]
        q = self.visual_norm(visual_tokens)
        w = self.water_norm(water_tokens)
        # cross attention
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=w,
            value=w,
            need_weights=return_attn,
            average_attn_weights=True
        )

        visual_tokens = visual_tokens + self.attn_scale * attn_out
        # FFN
        ffn_out = self.ffn(visual_tokens)

        visual_tokens = visual_tokens + self.ffn_scale * ffn_out

        # [B,H*W,64] -> [B,64,H,W]
        fused_feat = (
            visual_tokens
            .transpose(1, 2)
            .reshape(B, C, H, W)
        )

        # 64 -> 512
        delta = self.visual_recover(fused_feat)

        # 外部残差融合
        out = identity + self.gamma * delta

        if return_attn:
            return out, attn_weights

        return out


# ==========================
# Test
# ==========================
if __name__ == "__main__":

    model = WaterCrossAttention(
        in_channels=512,
        embed_dim=64,
        num_heads=4
    )

    visual_feat = torch.randn(
        2,
        512,
        40,
        40
    )

    water = torch.tensor([
        [5.8, 7.3, 10.2, 25.1],
        [4.9, 6.8, 12.5, 24.3]
    ])

    out = model(
        visual_feat,
        water
    )

    print("output shape:", out.shape)