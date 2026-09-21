"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA, Warmup


# ============================================================
# 水质字段和可比较损失配置
# ============================================================

WATER_KEYS = (
    "water_quality",
    "water",
)

# 训练和验证都能使用相同定义计算的损失类型
COMPARABLE_LOSS_TYPES = {
    "boxes",
    "focal",
    "vfl",
    "mal",
}


# ============================================================
# 水质数据提取
# ============================================================

def extract_water_quality(
    targets: Optional[Sequence[dict]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    从 targets 中提取水质数据。

    支持字段：
        target["water_quality"]
        target["water"]

    水质模式返回：
        [batch_size, num_water_parameters]

    纯视觉模式返回：
        None
    """

    if not targets:
        return None

    selected_key = next(
        (
            key
            for key in WATER_KEYS
            if any(key in target for target in targets)
        ),
        None,
    )

    if selected_key is None:
        return None

    missing_indices = [
        index
        for index, target in enumerate(targets)
        if selected_key not in target
    ]

    if missing_indices:
        raise KeyError(
            f"同一个 batch 中只有部分样本包含 "
            f"'{selected_key}'，缺失样本索引："
            f"{missing_indices}"
        )

    water_values = []

    for target in targets:
        value = target[selected_key]

        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value,
                dtype=torch.float32,
            )

        value = value.to(
            device=device,
            dtype=torch.float32,
        )

        # 每个样本统一整理成一维水质向量
        value = value.reshape(-1)

        water_values.append(value)

    feature_lengths = {
        value.numel()
        for value in water_values
    }

    if len(feature_lengths) != 1:
        raise ValueError(
            "同一个 batch 中水质向量长度不一致："
            f"{sorted(feature_lengths)}"
        )

    water_batch = torch.stack(
        water_values,
        dim=0,
    )

    return water_batch


# ============================================================
# 模型前向：兼容纯视觉和 water
# ============================================================

def forward_model(
    model: torch.nn.Module,
    samples,
    targets,
    device: torch.device,
    training: bool,
):
    """
    统一处理纯视觉模型和水质融合模型。

    训练阶段：
        model(samples, targets=targets, water=water)

    验证阶段：
        model(samples, water=water)

    当数据集中没有 water_quality 时：
        water=None

    只要 DEIM.forward() 中 water 的默认值为 None，
    同一套代码即可同时支持纯视觉和水质融合。
    """

    water = extract_water_quality(
        targets=targets,
        device=device,
    )

    if training:
        return model(
            samples,
            targets=targets,
            water=water,
        )

    return model(
        samples,
        water=water,
    )


# ============================================================
# 损失工具
# ============================================================

def _to_float_dict(loss_dict):
    """
    将损失字典中的标量 Tensor 转成 Python float。
    """

    return {
        key: (
            float(value.detach().item())
            if torch.is_tensor(value)
            else float(value)
        )
        for key, value in loss_dict.items()
    }


@torch.no_grad()
def compute_comparable_main_losses(
    criterion: torch.nn.Module,
    outputs,
    targets,
    epoch: int,
):
    """
    计算训练集与验证集具有相同定义的最终层主损失。

    仅包含：
        分类损失：
            loss_focal / loss_vfl / loss_mal

        边界框损失：
            loss_bbox
            loss_giou

    不包含：
        aux_outputs
        enc_aux_outputs
        dn_outputs
        pre_outputs
        loss_fgl
        loss_ddf

    原因：
        eval 模式下模型通常只返回最终层预测，不返回 aux_outputs。

        如果拿完整 train loss 和验证最终层 loss 比较，
        两边的损失组成不同，不能可靠判断过拟合。

        因此额外计算：
            train_main_loss
            val_main_loss

        二者定义完全一致。
    """

    required_keys = (
        "pred_logits",
        "pred_boxes",
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in outputs
    ]

    if missing_keys:
        raise KeyError(
            "最终层主损失需要 outputs 中包含 "
            f"{required_keys}，当前缺少："
            f"{missing_keys}"
        )

    main_outputs = {
        "pred_logits": outputs["pred_logits"],
        "pred_boxes": outputs["pred_boxes"],
    }

    # 使用最终层预测重新进行 Hungarian matching
    indices = criterion.matcher(
        main_outputs,
        targets,
        epoch=epoch,
    )["indices"]

    if hasattr(criterion, "_clear_cache"):
        criterion._clear_cache()

    # 目标框总数，用于损失归一化
    num_boxes = sum(
        len(target["labels"])
        for target in targets
    )

    num_boxes = torch.as_tensor(
        [num_boxes],
        dtype=torch.float,
        device=main_outputs["pred_logits"].device,
    )

    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(
            num_boxes
        )

    world_size = dist_utils.get_world_size()

    num_boxes = torch.clamp(
        num_boxes / world_size,
        min=1,
    ).item()

    losses = {}

    for loss_name in criterion.losses:

        # 跳过 local/FGL 等仅适用于完整训练输出的损失
        if loss_name not in COMPARABLE_LOSS_TYPES:
            continue

        meta = criterion.get_loss_meta_info(
            loss_name,
            main_outputs,
            targets,
            indices,
        )

        current_loss_dict = criterion.get_loss(
            loss_name,
            main_outputs,
            targets,
            indices,
            num_boxes,
            **meta,
        )

        # 使用与正式 criterion 相同的损失权重
        for key, value in current_loss_dict.items():
            if key in criterion.weight_dict:
                losses[key] = (
                    value
                    * criterion.weight_dict[key]
                )

    if not losses:
        raise RuntimeError(
            "没有生成可比较的最终层主损失。"
            f" 当前 criterion.losses="
            f"{criterion.losses}"
        )

    losses = {
        key: torch.nan_to_num(
            value,
            nan=0.0,
        )
        for key, value in losses.items()
    }

    return losses


# ============================================================
# COCO F1
# ============================================================

def compute_coco_f1(
    coco_eval,
    iou_threshold: float = 0.5,
):
    """
    从 COCO Precision-Recall 数据计算 Max F1。

    返回：
        f1_50_max
        precision_50_at_f1
        recall_50_at_f1

    这里的 F1 指：
        Max F1@IoU=0.5

    不是固定 confidence threshold 下的 F1。
    """

    empty_result = {
        "f1_50_max": 0.0,
        "precision_50_at_f1": 0.0,
        "recall_50_at_f1": 0.0,
    }

    if (
        coco_eval is None
        or not hasattr(coco_eval, "eval")
        or not coco_eval.eval
        or "precision" not in coco_eval.eval
    ):
        return empty_result

    precision = coco_eval.eval["precision"]

    # COCO precision:
    # [IoU, Recall, Category, Area, MaxDet]
    if precision.ndim != 5:
        return empty_result

    iou_thresholds = np.asarray(
        coco_eval.params.iouThrs
    )

    recall_thresholds = np.asarray(
        coco_eval.params.recThrs
    )

    iou_index = int(
        np.argmin(
            np.abs(
                iou_thresholds
                - iou_threshold
            )
        )
    )

    area_labels = list(
        coco_eval.params.areaRngLbl
    )

    if "all" in area_labels:
        area_index = area_labels.index(
            "all"
        )
    else:
        area_index = 0

    max_dets = list(
        coco_eval.params.maxDets
    )

    if 100 in max_dets:
        max_dets_index = max_dets.index(
            100
        )
    else:
        max_dets_index = (
            len(max_dets) - 1
        )

    # [Recall, Category]
    precision_at_iou = precision[
        iou_index,
        :,
        :,
        area_index,
        max_dets_index,
    ].astype(np.float64)

    # COCO 中 -1 表示无有效数据
    precision_at_iou[
        precision_at_iou < 0
    ] = np.nan

    valid_count = np.sum(
        np.isfinite(
            precision_at_iou
        ),
        axis=1,
    )

    precision_sum = np.nansum(
        precision_at_iou,
        axis=1,
    )

    mean_precision = np.zeros_like(
        recall_thresholds,
        dtype=np.float64,
    )

    valid_rows = valid_count > 0

    mean_precision[valid_rows] = (
        precision_sum[valid_rows]
        / valid_count[valid_rows]
    )

    valid_precision = mean_precision[
        valid_rows
    ]

    valid_recall = recall_thresholds[
        valid_rows
    ]

    if valid_precision.size == 0:
        return empty_result

    f1_curve = (
        2.0
        * valid_precision
        * valid_recall
        / (
            valid_precision
            + valid_recall
            + 1e-16
        )
    )

    best_index = int(
        np.argmax(f1_curve)
    )

    return {
        "f1_50_max": float(
            f1_curve[best_index]
        ),
        "precision_50_at_f1": float(
            valid_precision[best_index]
        ),
        "recall_50_at_f1": float(
            valid_recall[best_index]
        ),
    }


# ============================================================
# 训练一个 epoch
# ============================================================

def train_one_epoch(
    self_lr_scheduler,
    lr_scheduler,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    """
    训练一个 epoch。

    返回的统计中：

        loss:
            完整 DEIM 训练损失。
            包含 aux、encoder、DN、pre 等损失。
            用于反向传播。

        main_loss:
            最终层主损失。
            与验证 main_loss 定义完全一致。
            用于判断过拟合。
    """

    model.train()
    criterion.train()

    if len(data_loader) == 0:
        raise RuntimeError(
            "train_dataloader 没有任何 batch。"
        )

    metric_logger = MetricLogger(
        delimiter="  "
    )

    metric_logger.add_meter(
        "lr",
        SmoothedValue(
            window_size=1,
            fmt="{value:.6f}",
        ),
    )

    # 当前 logger 会在第一个 batch 前打印。
    # 如果不初始化，会出现 total/count = 0/0。
    metric_logger.update(
        lr=float(
            optimizer.param_groups[0]["lr"]
        )
    )

    header = f"Epoch: [{epoch}]"

    print_freq = kwargs.get(
        "print_freq",
        10,
    )

    writer: SummaryWriter = kwargs.get(
        "writer",
        None,
    )

    ema: ModelEMA = kwargs.get(
        "ema",
        None,
    )

    scaler: GradScaler = kwargs.get(
        "scaler",
        None,
    )

    lr_warmup_scheduler: Warmup = kwargs.get(
        "lr_warmup_scheduler",
        None,
    )

    cur_iters = (
        epoch * len(data_loader)
    )

    if isinstance(
        device,
        torch.device,
    ):
        autocast_device_type = (
            device.type
        )
    else:
        autocast_device_type = (
            str(device).split(":")[0]
        )

    for step, (samples, targets) in enumerate(
        metric_logger.log_every(
            data_loader,
            print_freq,
            header,
        )
    ):
        samples = samples.to(device)

        targets = [
            {
                key: (
                    value.to(device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value
                in target.items()
            }
            for target in targets
        ]

        # 只打印第一个真实 batch
        if (
            step == 0
            and dist_utils.is_main_process()
            and not hasattr(
                train_one_epoch,
                "_input_printed",
            )
        ):
            water_debug = (
                extract_water_quality(
                    targets,
                    device,
                )
            )

            print(
                "\n========== INPUT CHECK =========="
            )

            print(
                "epoch:",
                epoch,
            )

            print(
                "batch size:",
                len(targets),
            )

            print(
                "target keys:",
                list(
                    targets[0].keys()
                ),
            )

            print(
                "water shape:",
                (
                    None
                    if water_debug is None
                    else tuple(
                        water_debug.shape
                    )
                ),
            )

            print(
                "=================================\n"
            )

            train_one_epoch._input_printed = True

        global_step = (
            epoch
            * len(data_loader)
            + step
        )

        metas = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "epoch_step": len(data_loader),
        }

        optimizer.zero_grad()

        # ====================================================
        # AMP 训练
        # ====================================================
        if scaler is not None:

            with torch.autocast(
                device_type=(
                    autocast_device_type
                ),
                cache_enabled=True,
            ):
                outputs = forward_model(
                    model=model,
                    samples=samples,
                    targets=targets,
                    device=device,
                    training=True,
                )

            if (
                "pred_boxes" in outputs
                and (
                    torch.isnan(
                        outputs["pred_boxes"]
                    ).any()
                    or torch.isinf(
                        outputs["pred_boxes"]
                    ).any()
                )
            ):
                clean_state = {
                    key.replace(
                        "module.",
                        "",
                    ): value
                    for key, value
                    in model.state_dict().items()
                }

                dist_utils.save_on_master(
                    {
                        "model": clean_state
                    },
                    "./NaN.pth",
                )

                raise RuntimeError(
                    "pred_boxes 中出现 NaN 或 Inf。"
                )

            with torch.autocast(
                device_type=(
                    autocast_device_type
                ),
                enabled=False,
            ):
                # 完整训练损失
                loss_dict = criterion(
                    outputs,
                    targets,
                    **metas,
                )

                # 可比较的最终层主损失
                main_loss_dict = (
                    compute_comparable_main_losses(
                        criterion=criterion,
                        outputs=outputs,
                        targets=targets,
                        epoch=epoch,
                    )
                )

            loss = sum(
                loss_dict.values()
            )

            scaler.scale(
                loss
            ).backward()

            if max_norm > 0:
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm,
                )

            scaler.step(
                optimizer
            )

            scaler.update()

        # ====================================================
        # FP32 训练
        # ====================================================
        else:
            outputs = forward_model(
                model=model,
                samples=samples,
                targets=targets,
                device=device,
                training=True,
            )

            # 完整训练损失
            loss_dict = criterion(
                outputs,
                targets,
                **metas,
            )

            # 可比较的最终层主损失
            main_loss_dict = (
                compute_comparable_main_losses(
                    criterion=criterion,
                    outputs=outputs,
                    targets=targets,
                    epoch=epoch,
                )
            )

            loss = sum(
                loss_dict.values()
            )

            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm,
                )

            optimizer.step()

        # ====================================================
        # EMA
        # ====================================================
        if ema is not None:
            ema.update(model)

        # ====================================================
        # 学习率更新
        # ====================================================
        if self_lr_scheduler:
            optimizer = lr_scheduler.step(
                cur_iters + step,
                optimizer,
            )

        elif (
            lr_warmup_scheduler
            is not None
        ):
            lr_warmup_scheduler.step()

        # ====================================================
        # 完整训练损失统计
        # ====================================================
        loss_dict_reduced = (
            dist_utils.reduce_dict(
                loss_dict
            )
        )

        reduced_loss_values = (
            _to_float_dict(
                loss_dict_reduced
            )
        )

        loss_value = sum(
            reduced_loss_values.values()
        )

        # ====================================================
        # 可比较主损失统计
        # ====================================================
        main_loss_dict_reduced = (
            dist_utils.reduce_dict(
                main_loss_dict
            )
        )

        main_loss_values = (
            _to_float_dict(
                main_loss_dict_reduced
            )
        )

        main_loss_value = sum(
            main_loss_values.values()
        )

        if not math.isfinite(
            loss_value
        ):
            print(
                "训练完整损失无效：",
                loss_value,
            )

            print(
                reduced_loss_values
            )

            sys.exit(1)

        if not math.isfinite(
            main_loss_value
        ):
            raise RuntimeError(
                "训练主损失为 NaN 或 Inf："
                f"{main_loss_value}"
            )

        # 避免和完整损失字段重名
        main_component_values = {
            f"main_{key}": value
            for key, value
            in main_loss_values.items()
        }

        metric_logger.update(
            loss=loss_value,
            main_loss=main_loss_value,
            **reduced_loss_values,
            **main_component_values,
        )

        metric_logger.update(
            lr=(
                optimizer
                .param_groups[0]["lr"]
            )
        )

        # ====================================================
        # Batch 级 TensorBoard
        # ====================================================
        if (
            writer is not None
            and dist_utils.is_main_process()
            and global_step % 10 == 0
        ):
            writer.add_scalar(
                "Loss/total",
                loss_value,
                global_step,
            )

            writer.add_scalar(
                "Loss/main",
                main_loss_value,
                global_step,
            )

            for (
                parameter_group_index,
                parameter_group,
            ) in enumerate(
                optimizer.param_groups
            ):
                writer.add_scalar(
                    (
                        "Lr/pg_"
                        f"{parameter_group_index}"
                    ),
                    parameter_group["lr"],
                    global_step,
                )

            for key, value in (
                reduced_loss_values.items()
            ):
                writer.add_scalar(
                    f"Loss/{key}",
                    value,
                    global_step,
                )

            for key, value in (
                main_component_values.items()
            ):
                writer.add_scalar(
                    f"Loss/{key}",
                    value,
                    global_step,
                )

    # ========================================================
    # Epoch 级统计
    # ========================================================
    metric_logger.synchronize_between_processes()

    print(
        "Averaged training stats:",
        metric_logger,
    )

    train_stats = {
        key: float(
            meter.global_avg
        )
        for key, meter
        in metric_logger.meters.items()
    }

    if (
        writer is not None
        and dist_utils.is_main_process()
    ):
        writer.add_scalar(
            "Epoch/Loss_train",
            train_stats["loss"],
            epoch,
        )

        writer.add_scalar(
            "Epoch/MainLoss_train",
            train_stats["main_loss"],
            epoch,
        )

        writer.flush()

    return train_stats


# ============================================================
# 验证
# ============================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch=0,
    writer=None,
):
    """
    验证阶段计算：

        main_loss
        mAP50:95
        mAP50
        mAP75
        Max F1@0.5
        Precision
        Recall

    验证 main_loss 与训练 main_loss 定义相同。
    """

    model.eval()
    criterion.eval()

    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    metric_logger = MetricLogger(
        delimiter="  "
    )

    header = "Test:"

    if coco_evaluator is not None:
        iou_types = (
            coco_evaluator.iou_types
        )
    else:
        iou_types = ()

    for step, (samples, targets) in enumerate(
        metric_logger.log_every(
            data_loader,
            10,
            header,
        )
    ):
        samples = samples.to(device)

        targets = [
            {
                key: (
                    value.to(device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value
                in target.items()
            }
            for target in targets
        ]

        # eval 模式下不传 targets，
        # 防止触发训练专用 denoising 分支。
        # water 独立传入。
        outputs = forward_model(
            model=model,
            samples=samples,
            targets=targets,
            device=device,
            training=False,
        )

        # ====================================================
        # 验证主损失
        # ====================================================
        main_loss_dict = (
            compute_comparable_main_losses(
                criterion=criterion,
                outputs=outputs,
                targets=targets,
                epoch=epoch,
            )
        )

        main_loss_dict_reduced = (
            dist_utils.reduce_dict(
                main_loss_dict
            )
        )

        main_loss_values = (
            _to_float_dict(
                main_loss_dict_reduced
            )
        )

        main_loss_value = sum(
            main_loss_values.values()
        )

        if not math.isfinite(
            main_loss_value
        ):
            print(
                "验证主损失无效：",
                main_loss_value,
            )

            print(
                main_loss_values
            )

            raise RuntimeError(
                "验证主损失为 NaN 或 Inf。"
            )

        main_component_values = {
            f"main_{key}": value
            for key, value
            in main_loss_values.items()
        }

        metric_logger.update(
            main_loss=main_loss_value,
            **main_component_values,
        )

        # ====================================================
        # 检测结果后处理
        # ====================================================
        orig_target_sizes = torch.stack(
            [
                target["orig_size"]
                for target in targets
            ],
            dim=0,
        )

        results = postprocessor(
            outputs,
            orig_target_sizes,
        )

        predictions = {
            target["image_id"].item(): output
            for target, output
            in zip(
                targets,
                results,
            )
        }

        if coco_evaluator is not None:
            coco_evaluator.update(
                predictions
            )

    # ========================================================
    # 验证损失汇总
    # ========================================================
    metric_logger.synchronize_between_processes()

    print(
        "Averaged validation stats:",
        metric_logger,
    )

    # ========================================================
    # COCO 汇总
    # ========================================================
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {
        key: float(
            meter.global_avg
        )
        for key, meter
        in metric_logger.meters.items()
    }

    # ========================================================
    # 提取 mAP 和 F1
    # ========================================================
    if coco_evaluator is not None:

        if "bbox" in iou_types:
            bbox_eval = (
                coco_evaluator
                .coco_eval["bbox"]
            )

            bbox_stats = (
                bbox_eval.stats.tolist()
            )

            # 原始 COCO 12 项结果
            stats[
                "coco_eval_bbox"
            ] = bbox_stats

            stats[
                "map_50_95"
            ] = float(
                bbox_stats[0]
            )

            stats[
                "map_50"
            ] = float(
                bbox_stats[1]
            )

            stats[
                "map_75"
            ] = float(
                bbox_stats[2]
            )

            stats[
                "map_small"
            ] = float(
                bbox_stats[3]
            )

            stats[
                "map_medium"
            ] = float(
                bbox_stats[4]
            )

            stats[
                "map_large"
            ] = float(
                bbox_stats[5]
            )

            stats.update(
                compute_coco_f1(
                    bbox_eval,
                    iou_threshold=0.5,
                )
            )

        if "segm" in iou_types:
            stats[
                "coco_eval_masks"
            ] = (
                coco_evaluator
                .coco_eval["segm"]
                .stats
                .tolist()
            )

    # ========================================================
    # Epoch 级 TensorBoard
    # ========================================================
    if (
        writer is not None
        and dist_utils.is_main_process()
    ):
        writer.add_scalar(
            "Epoch/MainLoss_val",
            stats["main_loss"],
            epoch,
        )

        tensorboard_metrics = {
            "map_50_95":
                "Metrics/mAP_50_95",

            "map_50":
                "Metrics/mAP_50",

            "map_75":
                "Metrics/mAP_75",

            "map_small":
                "Metrics/mAP_small",

            "map_medium":
                "Metrics/mAP_medium",

            "map_large":
                "Metrics/mAP_large",

            "f1_50_max":
                "Metrics/F1_50_max",

            "precision_50_at_f1":
                "Metrics/Precision_50_at_F1",

            "recall_50_at_f1":
                "Metrics/Recall_50_at_F1",
        }

        for (
            stats_key,
            tensorboard_name,
        ) in tensorboard_metrics.items():

            if stats_key in stats:
                writer.add_scalar(
                    tensorboard_name,
                    float(
                        stats[stats_key]
                    ),
                    epoch,
                )

        writer.flush()

    return stats, coco_evaluator