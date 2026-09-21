"""
统计 DEIMv2 不同水质融合位置下的：
1. Params
2. MACs
3. FLOPs

统一条件：
- batch size = 1
- image size = 640 × 640
- water shape = [1, 4]
- 水质顺序：[temperature, do, ph, turbidity]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from calflops import calculate_flops


# 让脚本能够导入项目中的 engine
CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "../.."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from engine.core import YAMLConfig


class ModelForFlops(nn.Module):
    """
    包装原模型，使 FLOPs 工具在启用水质融合时，
    自动传入形状为 [B, 4] 的水质张量。

    对理论 Params、MACs、FLOPs 来说，
    具体水质数值不会改变运算次数。

    重要的是：
        water 不是 None，
        从而真正执行水质融合分支。
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()

        self.model = model

        # 读取三个位置的实际开关
        self.backbone_water = bool(
            getattr(
                self.model.backbone,
                "use_water_quality",
                False,
            )
        )

        self.encoder_water = bool(
            getattr(
                self.model.encoder,
                "use_water_quality",
                False,
            )
        )

        self.decoder_water = bool(
            getattr(
                self.model.decoder,
                "use_water_quality",
                False,
            )
        )

        self.water_flags = {
            "backbone": self.backbone_water,
            "encoder": self.encoder_water,
            "decoder": self.decoder_water,
        }

        self.active_stages = [
            stage_name
            for stage_name, enabled
            in self.water_flags.items()
            if enabled
        ]

        # 当前实验要求一次只打开一个融合位置
        if len(self.active_stages) > 1:
            raise RuntimeError(
                "当前同时启用了多个水质融合位置："
                f"{self.active_stages}。\n"
                "为了进行四组消融实验，每次应只开启一个位置。"
            )

        self.use_water = (
            len(self.active_stages) == 1
        )

        print(
            "\n========== FLOPS CONFIG CHECK =========="
        )

        print(
            "Backbone water:",
            self.backbone_water,
        )

        print(
            "Encoder water :",
            self.encoder_water,
        )

        print(
            "Decoder water :",
            self.decoder_water,
        )

        if self.active_stages:
            print(
                "Active fusion stage:",
                self.active_stages[0],
            )
        else:
            print(
                "Active fusion stage: visual only"
            )

        print(
            "========================================\n"
        )

        self._input_printed = False

    def forward(
        self,
        images: torch.Tensor,
    ):
        """
        images:
            [B, 3, 640, 640]
        """

        water = None

        if self.use_water:
            batch_size = images.shape[0]

            # 水质顺序：
            # [temperature, do, ph, turbidity]
            #
            # 具体数值不影响理论 MACs 和 FLOPs，
            # 这里只用于确保融合分支真正执行。
            water = images.new_tensor(
                [
                    24.5,
                    5.0,
                    7.0,
                    11.0,
                ]
            )

            water = (
                water
                .reshape(1, 4)
                .repeat(
                    batch_size,
                    1,
                )
            )

        if not self._input_printed:
            print(
                "\n========== FLOPS INPUT CHECK =========="
            )

            print(
                "image shape:",
                tuple(images.shape),
            )

            print(
                "water shape:",
                (
                    None
                    if water is None
                    else tuple(water.shape)
                ),
            )

            if water is not None:
                print(
                    "water first sample:",
                    water[0],
                )

            print(
                "=======================================\n"
            )

            self._input_printed = True

        outputs = self.model(
            images,
            water=water,
        )

        return outputs


def main(args):
    # ========================================================
    # 1. 读取配置并创建模型
    # ========================================================
    cfg = YAMLConfig(
        args.config,
        resume=None,
    )

    base_model = cfg.model

    # 使用部署/推理模式统计标准前向复杂度
    base_model = base_model.deploy()
    base_model.eval()

    # ========================================================
    # 2. 包装模型，让 profiler 真正传入 water
    # ========================================================
    model = ModelForFlops(
        base_model
    )

    model.eval()

    # ========================================================
    # 3. 统计 Params、MACs、FLOPs
    # ========================================================
    flops, macs, profiler_params = (
        calculate_flops(
            model=model,

            # 标准模型复杂度通常统一使用 batch=1
            input_shape=(
                1,
                3,
                args.image_size,
                args.image_size,
            ),

            output_as_string=True,
            output_precision=6,

            print_results=True,
        )
    )

    # 直接统计参数，作为精确整数结果
    exact_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\n========== FINAL COMPLEXITY RESULT =========="
    )

    print(
        f"Config           : {args.config}"
    )

    print(
        f"Input size       : "
        f"1 × 3 × {args.image_size} × {args.image_size}"
    )

    print(
        f"Params exact     : {exact_params:,}"
    )

    print(
        f"Params trainable : {trainable_params:,}"
    )

    print(
        f"MACs             : {macs}"
    )

    print(
        f"FLOPs            : {flops}"
    )

    print(
        "============================================\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=str,
        help="模型 YAML 配置文件路径",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="统计时的输入图像尺寸",
    )

    args = parser.parse_args()

    main(args)