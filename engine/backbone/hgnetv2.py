
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d
from ..core import register
import logging
from .common import get_activation
#加入你要用的模块 但最好视情况而定 因为会导致加载太多包
from engine.mynewblock import *

kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2']


class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            act='relu',
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            # self.act = nn.ReLU()
            self.act = get_activation(act)
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
            act='relu',
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            act=act,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            act=act,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act='relu'):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            act=act,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            act=act,
            )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)#2倍下采样
        x = F.pad(x, (0, 1, 0, 1))#填充+1
        x2 = self.stem2a(x)#-1
        x2 = F.pad(x2, (0, 1, 0, 1))#+1
        x2 = self.stem2b(x2)#-1
        x1 = self.pool(x)#不变
        x = torch.cat([x1, x2], dim=1)#2倍
        x = self.stem3(x)#2倍
        x = self.stem4(x)#不变
        return x#4倍下采样通道数变16


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)#h=1,W=1
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
#inchs-outchs特征不变 上面还有个stage
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
            act='relu',
            addconv=None,
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
            #  "stage1": [16, 16, 64, 1, False, False, 3, 3],
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()#inchs-midchs不改变尺寸
        for i in range(layer_num):
            if light_block:
                print("light_block")
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        act=act,
                        #"stage2": [64, 32, 256, 1, True, False, 3, 3],
                    )
                )
            elif addconv in {'StarConv', 'ODConv', 'DSConv', 'IDConv', 'DCNv2', 'CondConv2D', 'WTConv', 'PSConv',
                             'FFConv', 'RFAConv',
                             'RFCAConv', 'RFCBAMConv', 'SCConv', 'LDConv', 'DEConv', 'TBConv', 'PConv', 'RepConv',
                             'DynamicConv',
                             'GBConv', 'MSGDConv', 'FADConv', 'FDConv', 'ARConv', 'wConv2D', 'MBRConv3', 'MBRConv5',
                             'SFSConv',
                             'PATConv', 'StripConv', 'GCConv', 'CGHalfConv', 'HLKConv'
                             } | {'WFEConv', 'GSConv', 'SPConv', 'MRFAConv'}:  # |后面的是二次创新的模块，方便小伙伴观察
                print("使用了addconv=", addconv)
                module_conv = globals()[addconv]  # 从全局变量中获取
                if addconv in {'FADConv'}:
                    self.layers.append(
                        module_conv(in_channels=in_chs if i == 0 else mid_chs, out_channels=mid_chs, kernel_size=3,
                                    stride=1)
                    )
                elif addconv in {'StripConv'}:
                    in_dim = in_chs if i == 0 else mid_chs
                    if mid_chs <= 32:
                        k = 5
                    elif mid_chs <= 128:
                        k = 7
                    else:
                        k = 11
                    self.layers.append(
                        module_conv(
                            dim=in_dim,
                            out_dim=mid_chs,
                            k1=1,
                            k2=k
                        )
                    )
                else:
                    self.layers.append(
                        module_conv(
                            in_chs if i == 0 else mid_chs,  # 第一层用输入通道，其余用中间通道
                            mid_chs
                        )
                    )
            else:
                print("conv")
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                        act=act,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            print("se")
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        elif agg in {'CBAM', 'CAA', 'CDFA', 'ECA', 'GCSA', 'SSA', 'LocalAttention', 'CLAttention', 'SCSA', 'CASAtt',
                     'LWGA',
                     'LRSA', 'KSFA', 'EMA', 'SeaAttention', 'PolaLinearAttention', 'LEGA', 'TSA', 'FCAttention', 'ASSA',
                     'DualdomainSA',
                     'RSA', 'GSA', 'CGAttention', 'DLKA', 'CloAttention', 'BiLevelRoutingAttention', 'RCA',
                     'ShuffleAttn',
                     'SCGA', 'SAA', 'VMMAttention', 'AlternatingAttention', 'ConvAtt', 'HFP', 'HMHA', 'MALAttention',
                     'GCBAM', 'MSCA', 'DynamicSpatialAttention', 'CSAM', 'CASAB', 'MultiScaleAttention', 'AAFM', 'IIA',
                     'PATConvAttention', 'DSPM', 'RCSSC'}:  # 自己在主干中添加注意力，方式一

            module_agg = globals()[agg]  # 从全局变量中获取
            print("使用了agg_blocks=", agg)
            self.aggregation = nn.Sequential(
                ConvBNAct(total_chs, out_chs, kernel_size=1, use_lab=use_lab),
                module_agg(out_chs),
            )
        elif agg == 'aafm':
            num_heads = max(1, (out_chs // 2) // 32)
            self.aggregation = nn.Sequential(
                ConvBNAct(
                    total_chs,
                    out_chs,
                    kernel_size=1,
                    stride=1,
                    use_lab=use_lab,
                    act=act,
                ),
                AAFM(
                    dim=out_chs,
                    num_heads=num_heads,
                    bias=False
                )
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            print("ese")
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()
#  "stage1": [16, 16, 64, 1, False, False, 3, 3],
    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)#inchs-midchas
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,#2倍下采样不改变通道数
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='aafm',
            drop_path=0.,
            act='relu',
            addconv=None,
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num

    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
                act=act,
            )
        else:
            self.downsample = nn.Identity()
        #  "stage1": [16, 16, 64, 1, False, False, 3, 3],
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    act=act,
                    addconv=addconv
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        'Atto': {      # only 3 stages
            'stem_channels': [3, 16, 16],#3，640，640-16，640/4，640/4
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 256, 1, True, True, 3, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'Femto': {      # only 3 stages
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'Pico': {      # only 3 stages
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/',
                 act='relu',
                 addconv=None,
                 agg=None,
                 use_water_quality=True,
                 water_fuse_idx=2,
                 ):
        super().__init__()
        print("======== 使用的是我修改后的 HGNetv2 ========")
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]
        print(f"        ### Backbone.act: {act} ###     ")
        print(f"        ### Backbone.act: {act} ###     ")
        self.use_water_quality=use_water_quality
        self.water_fuse_idx=water_fuse_idx
        if use_water_quality:
            self.water_fusion = WaterCrossAttention(in_channels=512,embed_dim=64,num_heads=4,ffn_ratio=4,dropout=0.0,init_scale=0.01)

        # stem
        self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab,
                act=act)

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    act=act,
                    addconv=addconv,
                    agg=agg)
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
           #16，640 / 4，640 / 4 #  "stage1": [16, 16, 64, 1, False, False, 3, 3],
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

            # ... (前面的代码保持不变，定位到文件末尾的 __init__ 方法中)

        pretrained = pretrained
        print('加载预训练权重:', pretrained)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                model_path = os.path.join(local_model_dir, f'PPHGNetV2_{name}_stage1.pth')
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        if torch.distributed.get_rank() == 0:
                            print(GREEN + "Downloading pretrained HGNetV2..." + RESET)
                            state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu',
                                                                       model_dir=local_model_dir)
                            torch.distributed.barrier()
                        else:
                            torch.distributed.barrier()
                            state = torch.load(model_path, map_location='cpu')
                    else:
                        # 非分布式情况下直接下载
                        print(GREEN + "Downloading pretrained HGNetV2..." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu',
                                                                   model_dir=local_model_dir)

                # 🔥加载部分预训练参数（仅 stem 和 stage1）
                self.load_pretrained_filtered(state, keep_keys=['stem.*', 'stages.0.*'])

            except Exception as e:
                print(f"{str(e)}")
                logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                logging.error(GREEN + f"Please manually download from {download_url} to {local_model_dir}" + RESET)

    def load_pretrained_filtered(self,
                                 state_dict,
                                 keep_keys=['stem.*', 'stages.0.*', 'stages.1.*', 'stages.2.*'],
                                 verbose=False):
        import re

        # Step 1: 只保留匹配的 key
        keys_to_keep = []
        for k in list(state_dict.keys()):
            if any(re.match(pat, k) for pat in keep_keys):
                keys_to_keep.append(k)

        filtered_state = {}
        model_dict = self.state_dict()  # 这里改成self.state_dict()，拿当前模型参数字典

        for k in keys_to_keep:
            if k in model_dict and state_dict[k].shape == model_dict[k].shape:
                filtered_state[k] = state_dict[k]
            else:
                if verbose:
                    print(f"❌ Skip loading: {k}")
                    if k in model_dict:
                        print(f"   Shape mismatch: expected {model_dict[k].shape}, got {state_dict[k].shape}")
                    else:
                        print(f"   Key not found in model")

        # Step 2: 加载参数，必须self调用
        for key in filtered_state.keys():
            print(key)

        self.load_state_dict(filtered_state, strict=False)

    @staticmethod
    def load_partial_state_dict(model, state_dict):
        model_dict = model.state_dict()
        # 只保留shape完全一致的参数
        filtered_dict = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}

        # 更新模型参数
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        missing = set(model_dict.keys()) - set(filtered_dict.keys())
        unexpected = set(state_dict.keys()) - set(filtered_dict.keys())
        print("Missing keys:", missing)
        print("   #########################################################")
        print("Unexpected keys:", unexpected)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x,water):
        x = self.stem(x)#3,640,640-16,640/4,640/4
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.use_water_quality and water is not None and idx == self.water_fuse_idx:
                x = self.water_fusion(x,water)
               

            if idx in self.return_idx:
                outs.append(x)
        return outs

