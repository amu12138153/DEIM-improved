import torch.nn.functional as F
import numbers
from einops import rearrange
import torch
import torch.nn as nn
import torch.fft as fft
__all__ = ['AAFM']

class ComplexFFT(nn.Module):
    #傅里叶变换b,c,h,w--两个bchw
    def __init__(self):
        super(ComplexFFT, self).__init__()

    def forward(self, x):
        # 对输入进行复数FFT变换
        x_fft = fft.fft2(x, dim=(-2, -1))  # 2D FFT，沿最后两个维度进行
        real = x_fft.real  # 提取实部
        imag = x_fft.imag  # 提取虚部
        return real, imag
class ComplexIFFT(nn.Module):
    #傅里叶逆变换两个bchw--bchw
    def __init__(self):
        super(ComplexIFFT, self).__init__()

    def forward(self, real, imag):
        # 复合实部和虚部，作为复杂信号
        x_complex = torch.complex(real, imag)
        # 进行复数IFFT逆变换
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))  # 2D IFFT
        return x_ifft.real  # 输出结果的实部
class Conv1x1(nn.Module):
    #通道混合
    def __init__(self, in_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0,groups=in_channels*2)

    def forward(self, x):
        return self.conv(x)

class Stage2_fft(nn.Module):
    #傅里叶变换-卷积-逆变换
    def __init__(self, in_channels):
        super(Stage2_fft, self).__init__()
        self.c_fft = ComplexFFT()
        self.conv1x1 = Conv1x1(in_channels)
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)
        conv_out = self.conv1x1(combined)

        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]
        imag_out = conv_out[:, out_channels:, :, :]

        output = self.c_ifft(real_out, imag_out)

        return output

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    #输入是B,HW,C x_out=x//sprt(xigma+u)*weight
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    #层归一化 通道数仿射
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Multi-Scale Flow Gating Network
class FeedForward(nn.Module):
    #升维-------深度可分离卷积------gelu激活--降维
            #---深度可分离卷积--^
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features , hidden_features , kernel_size=3, stride=1, padding=1,
                                groups=hidden_features , bias=bias)
        self.dwconv_2=nn.Conv2d(hidden_features , hidden_features , kernel_size=5, padding='same',
                                groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1=self.dwconv(x1)
        x2=self.dwconv_2(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class spr_sa(nn.Module):
    #卷积-池化-通道权重-激活-卷积 动态增强重要通道特征
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)


    def forward(self, x):
        x = self.conv_0(x)
        x1= F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.softmax(x1, dim=1)
        x=x1*x
        x = self.act(x)
        x = self.conv_1(x)
        return x


class AAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(AAFM, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.spr_sa=spr_sa(dim//2,2)
        self.linear_0 = nn.Conv2d(dim, dim , 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim//2, dim//2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim//2 * 3, dim//2 * 3, kernel_size=3, stride=1, padding=1, groups=dim//2 * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim//2, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim//2, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim//2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        self.fft=Stage2_fft(in_channels=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim//2, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y, x = self.linear_0(x).chunk(2, dim=1)

        y_d = self.spr_sa(y)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        gate_output = self.gate(x)
        if torch.isnan(gate_output).any():
            gate_output = torch.nan_to_num(gate_output, nan=0.0)  # 将 NaN 转换为 0

        dynamic_k = int(C * gate_output.view(b, -1).mean())
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)
        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Frequency Adaptive Interaction Module (FAIM)
        # stage1
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(out_att)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(y_d)

        # S-I
        attened_x = out_att * torch.sigmoid(spatial_map)
        # C-I
        conv_x = y_d * torch.sigmoid(channel_map)

        x = torch.cat([attened_x, conv_x], dim=1)
        out = self.project_out(x)
        # stage 2
        # out=self.fft(out)
        return out
class AAFMBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(AAFMBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AAFM(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

