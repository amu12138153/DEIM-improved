"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

import torch.nn as nn
from calflops import calculate_flops


def _module_uses_water(module: nn.Module) -> bool:
    """
    检查某个模块及其子模块中，
    是否存在已开启的水质融合开关。
    """
    return any(
        getattr(submodule, "use_water_quality", False)
        for submodule in module.modules()
    )


class ModelForInfo(nn.Module):
    """
    给 calflops 使用的包装模型。

    calflops 只会传入 images，
    包装器负责额外构造 [B, 4] 的水质张量，
    从而让水质融合分支真正参与前向计算。
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model

        self.water_flags = {
            "backbone": _module_uses_water(
                self.model.backbone
            ),
            "encoder": _module_uses_water(
                self.model.encoder
            ),
            "decoder": _module_uses_water(
                self.model.decoder
            ),
        }

        print(
            "\n========== PROFILER FUSION CHECK =========="
        )

        print(
            "Backbone fusion:",
            self.water_flags["backbone"]
        )

        print(
            "Encoder fusion :",
            self.water_flags["encoder"]
        )

        print(
            "Decoder fusion :",
            self.water_flags["decoder"]
        )

        active_stages = [
            stage_name
            for stage_name, enabled
            in self.water_flags.items()
            if enabled
        ]

        if active_stages:
            print(
                "Active stage   :",
                ", ".join(active_stages)
            )
        else:
            print(
                "Active stage   : visual only"
            )

        print(
            "===========================================\n"
        )

        self.input_printed = False

    def forward(self, images):
        batch_size = images.shape[0]

        # 实际水质顺序：
        # [temperature, do, ph, turbidity]
        #
        # 具体数值不会影响理论 MACs/FLOPs，
        # 只要形状为 [B, 4] 并执行水质分支即可。
        water = images.new_tensor(
            [25.0, 5.0, 7.0, 10.0]
        )

        water = water.reshape(
            1,
            4
        ).repeat(
            batch_size,
            1
        )

        if not self.input_printed:
            print(
                "========== PROFILER INPUT CHECK =========="
            )

            print(
                "image shape:",
                tuple(images.shape)
            )

            print(
                "water shape:",
                tuple(water.shape)
            )

            print(
                "water first sample:",
                water[0]
            )

            print(
                "==========================================\n"
            )

            self.input_printed = True

        return self.model(
            images,
            water=water
        )


def stats(
    cfg,
    input_shape: Tuple = (
        1,
        3,
        640,
        640
    ),
) -> Tuple[int, dict]:

    base_size = (
        cfg
        .train_dataloader
        .collate_fn
        .base_size
    )

    input_shape = (
        1,
        3,
        base_size,
        base_size
    )

    # 原始部署模型
    base_model = (
        copy
        .deepcopy(cfg.model)
        .deploy()
        .eval()
    )

    # 包装后，calflops虽然只传入images，
    # 但forward内部会同时给模型传入water。
    model_for_info = ModelForInfo(
        base_model
    ).eval()

    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False
    )

    params = sum(
        parameter.numel()
        for parameter
        in model_for_info.parameters()
    )

    del model_for_info

    return params, {
        "Model FLOPs:%s   MACs:%s   Params:%s"
        % (
            flops,
            macs,
            params
        )
    }