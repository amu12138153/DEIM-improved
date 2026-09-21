"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""
import torch 
import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    
    def forward(self, x, targets=None,water=None):
        if water is None and targets is not None:
            if len(targets) > 0 and "water_quality" in targets[0]:
                water = torch.stack(
                    [t["water_quality"] for t in targets],
                    dim=0
                )
                water = water.to(next(self.parameters()).device)
        # x = self.backbone(x, water=water)
        # x = self.encoder(x)
        # x = self.decoder(x, targets)版本一
        x = self.backbone(x,water=water)
        x = self.encoder(x, water=water)
        x = self.decoder(x, targets,water=water)

        return x
    

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
