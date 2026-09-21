# -*- coding: utf-8 -*-
"""
DEIMv2 + Query-Water 单张图片推理

功能：
1. 指定一张 JPG 图片路径。
2. 从指定 CSV 中按照 JPG 文件名查找：
       file_name, temperature, do, ph, turbidity
3. 使用与 DEIMv2 当前 val_dataloader 相同的数据预处理。
4. 将 CSV 中的水质数据写入：
       target['water_quality']
   然后执行：
       model(samples, targets)
5. 使用 DEIMv2 原有 postprocessor 还原到原图坐标。
6. 在原图上绘制预测框、类别和置信度。
7. 同时保存 prediction JSON，方便后续分析。

注意：
- 这个版本使用 cfg.val_dataloader.dataset 来取得和验证阶段一致的图像预处理。
- 因此，待推理图片应当存在于 YAML 的 val_dataloader 对应 COCO 数据集中。
- CSV 第一列必须是 JPG 文件名（可带路径，脚本只比较 basename）。
- 水质顺序固定为：temperature, do, ph, turbidity。
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from engine.core import YAMLConfig


# ============================================================
# 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DEIMv2 单张图片 + CSV 水质数据推理"
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="DEIMv2 YAML 配置文件"
    )

    parser.add_argument(
        "--resume",
        "-r",
        required=True,
        help="checkpoint，例如 best_stg1.pth"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="需要推理的 JPG 图片完整路径"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="水质 CSV 文件路径"
    )

    parser.add_argument(
        "--output",
        default="single_infer_result",
        help="输出目录"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5457,
        help="绘制预测框的最低置信度，默认 0.5457"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda / cpu，默认 cuda"
    )

    return parser.parse_args()


# ============================================================
# 文件名标准化
# ============================================================

def normalize_filename(path_or_name: str) -> str:
    """只取 basename，并统一大小写与路径分隔符。"""
    value = str(path_or_name).replace("\\", "/").strip()
    return os.path.basename(value).lower()


# ============================================================
# CSV 读取
# ============================================================

def read_sensor_csv(csv_path: str) -> Dict[str, Dict[str, float]]:
    """
    读取传感器 CSV。

    期望列：
        file_name
        temperature
        do
        ph
        turbidity

    允许 CSV 编码：
        utf-8-sig / utf-8 / gb18030 / gbk
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")

    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if reader.fieldnames is None:
                raise RuntimeError("CSV 没有表头。")

            headers = {h.strip().lower(): h for h in reader.fieldnames}

            # 第一列也可以不是严格叫 file_name，但要求能找到下面几个字段。
            file_col = headers.get("file_name")
            if file_col is None:
                file_col = headers.get("filename")
            if file_col is None:
                file_col = headers.get("image")

            required = ["temperature", "do", "ph", "turbidity"]
            missing = [x for x in required if x not in headers]

            if file_col is None or missing:
                raise RuntimeError(
                    "CSV 列名不符合要求。\n"
                    f"当前列：{reader.fieldnames}\n"
                    "至少需要：file_name, temperature, do, ph, turbidity"
                )

            result = {}

            for line_no, row in enumerate(rows, start=2):
                raw_name = row.get(file_col, "")
                key = normalize_filename(raw_name)

                if not key:
                    continue

                try:
                    item = {
                        "file_name": str(raw_name).strip(),
                        "temperature": float(row[headers["temperature"]]),
                        "do": float(row[headers["do"]]),
                        "ph": float(row[headers["ph"]]),
                        "turbidity": float(row[headers["turbidity"]]),
                    }
                except (TypeError, ValueError, KeyError) as exc:
                    raise RuntimeError(
                        f"CSV 第 {line_no} 行水质数据无法转换为数字：{row}\n"
                        f"原因：{exc}"
                    ) from exc

                # 同名文件重复时，直接报错，避免选到错误的一行。
                if key in result:
                    raise RuntimeError(
                        f"CSV 中发现重复文件名：{raw_name}\n"
                        "请保证一个 JPG 文件名只对应一行水质数据。"
                    )

                result[key] = item

            print(f"CSV 读取成功：{csv_path}")
            print(f"CSV 有效记录：{len(result)}")
            return result

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"读取 CSV 失败：{csv_path}\n最后一次错误：{last_error}"
    )


# ============================================================
# Dataset 辅助函数
# ============================================================

def find_ann_file(obj: Any, max_depth: int = 8, depth: int = 0) -> Optional[str]:
    """尽量从 DataLoader / Dataset 中寻找 COCO annotation 文件。"""

    if obj is None or depth > max_depth:
        return None

    for attr in (
        "ann_file",
        "anno_file",
        "annotation_file",
        "json_file",
        "ann_path",
    ):
        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if isinstance(value, str) and os.path.isfile(value):
            return value

    for attr in ("dataset", "datasets"):
        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if attr == "dataset":
            result = find_ann_file(
                value,
                max_depth=max_depth,
                depth=depth + 1,
            )
            if result is not None:
                return result
        elif isinstance(value, (list, tuple)):
            for item in value:
                result = find_ann_file(
                    item,
                    max_depth=max_depth,
                    depth=depth + 1,
                )
                if result is not None:
                    return result

    return None


def find_dataset_image_index(
    dataset: Any,
    target_filename: str,
) -> Tuple[int, int, str]:
    """
    在 cfg.val_dataloader.dataset 中寻找指定图片。

    返回：
        dataset_idx, image_id, coco_file_name
    """

    target_key = normalize_filename(target_filename)

    # --------------------------------------------------------
    # 1. 常见 COCO Dataset：dataset.ids + dataset.coco
    # --------------------------------------------------------
    if hasattr(dataset, "ids") and hasattr(dataset, "coco"):
        try:
            ids = list(dataset.ids)
            coco = dataset.coco

            for idx, image_id in enumerate(ids):
                info = coco.loadImgs([int(image_id)])[0]
                file_name = str(info.get("file_name", ""))

                if normalize_filename(file_name) == target_key:
                    return idx, int(image_id), file_name
        except Exception:
            pass

    # --------------------------------------------------------
    # 2. 有 img_ids / image_ids + coco
    # --------------------------------------------------------
    for id_attr in ("img_ids", "image_ids"):
        if hasattr(dataset, id_attr) and hasattr(dataset, "coco"):
            try:
                ids = list(getattr(dataset, id_attr))
                coco = dataset.coco

                for idx, image_id in enumerate(ids):
                    info = coco.loadImgs([int(image_id)])[0]
                    file_name = str(info.get("file_name", ""))

                    if normalize_filename(file_name) == target_key:
                        return idx, int(image_id), file_name
            except Exception:
                pass

    # --------------------------------------------------------
    # 3. Subset：映射到原始 Dataset
    # --------------------------------------------------------
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        try:
            indices = list(dataset.indices)
            base_idx, image_id, file_name = find_dataset_image_index(
                dataset.dataset,
                target_filename,
            )
            # base_idx -> subset local idx
            for local_idx, original_idx in enumerate(indices):
                if int(original_idx) == int(base_idx):
                    return local_idx, image_id, file_name
        except Exception:
            pass

    # --------------------------------------------------------
    # 4. ConcatDataset / 多数据集
    # --------------------------------------------------------
    if hasattr(dataset, "datasets"):
        try:
            offset = 0
            for sub_dataset in dataset.datasets:
                try:
                    local_idx, image_id, file_name = find_dataset_image_index(
                        sub_dataset,
                        target_filename,
                    )
                    return offset + local_idx, image_id, file_name
                except Exception:
                    try:
                        offset += len(sub_dataset)
                    except Exception:
                        pass
        except Exception:
            pass

    # --------------------------------------------------------
    # 5. 最后兜底：利用 COCO annotation 得到 image_id，
    #    再遍历 Dataset 的 target['image_id']。
    # --------------------------------------------------------
    ann_file = find_ann_file(dataset)
    target_image_id = None
    target_coco_name = target_filename

    if ann_file is not None:
        with open(ann_file, "r", encoding="utf-8") as f:
            coco_json = json.load(f)

        for info in coco_json.get("images", []):
            file_name = str(info.get("file_name", ""))
            if normalize_filename(file_name) == target_key:
                target_image_id = int(info["id"])
                target_coco_name = file_name
                break

    if target_image_id is not None:
        try:
            for idx in range(len(dataset)):
                _, target = dataset[idx]
                image_id = target.get("image_id")

                if torch.is_tensor(image_id):
                    image_id = int(image_id.detach().cpu().item())
                else:
                    image_id = int(image_id)

                if image_id == target_image_id:
                    return idx, target_image_id, target_coco_name
        except Exception as exc:
            raise RuntimeError(
                "遍历 Dataset 查找图片时失败。\n"
                f"原因：{repr(exc)}"
            ) from exc

    raise FileNotFoundError(
        "这张图片没有在 cfg.val_dataloader.dataset 中找到。\n"
        f"目标图片：{target_filename}\n"
        "请确认：\n"
        "1. YAML 的 val_dataloader 指向你的测试集；\n"
        "2. COCO JSON 的 images 中包含这张 JPG；\n"
        "3. JPG 文件名与 COCO JSON / CSV 中的文件名一致。"
    )


# ============================================================
# 类别名称
# ============================================================

def get_category_names(ann_file: str) -> Tuple[List[str], List[int]]:
    """从 COCO JSON 中读取类别名和 category_id。"""

    with open(ann_file, "r", encoding="utf-8") as f:
        coco_json = json.load(f)

    categories = sorted(
        coco_json.get("categories", []),
        key=lambda x: int(x["id"]),
    )

    names = [str(x["name"]) for x in categories]
    category_ids = [int(x["id"]) for x in categories]

    return names, category_ids


# ============================================================
# Checkpoint
# ============================================================

def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """按当前 eval_curves.py 相同方式加载 EMA/model。"""

    print("\n========== 加载模型 ==========")
    print(f"Config    : {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = YAMLConfig(
        config_path,
        resume=checkpoint_path,
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "ema" in ckpt:
        state = ckpt["ema"]["module"]
        print("Checkpoint: 使用 EMA 模型")
    elif "model" in ckpt:
        state = ckpt["model"]
        print("Checkpoint: 使用 model 权重")
    else:
        raise KeyError("checkpoint 中不存在 'ema' 或 'model'。")

    model = cfg.model
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    post = cfg.postprocessor
    post = post.to(device)
    post.eval()

    print(f"Device    : {device}")
    print("模型加载完成。")

    return cfg, model, post


# ============================================================
# Target 转 device
# ============================================================

def move_target_to_device(target: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result = {}

    for key, value in target.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value

    return result


# ============================================================
# 画框
# ============================================================

def safe_font(size: int = 22):
    """Windows / Linux 尝试几个常见字体；失败则使用默认字体。"""

    font_candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    for path in font_candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass

    return ImageFont.load_default()


def draw_box(
    draw: ImageDraw.ImageDraw,
    box,
    label_text: str,
    color: str = "lime",
    width: int = 4,
):
    x1, y1, x2, y2 = [float(v) for v in box]

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=width,
    )

    font = safe_font(22)

    # 文字背景
    bbox = draw.textbbox((0, 0), label_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = max(0, x1)
    ty = max(0, y1 - th - 6)

    draw.rectangle(
        [tx, ty, tx + tw + 8, ty + th + 6],
        fill="black",
    )

    draw.text(
        (tx + 4, ty + 2),
        label_text,
        fill="white",
        font=font,
    )


# ============================================================
# 单张推理
# ============================================================

@torch.no_grad()
def run_single_inference(
    model,
    post,
    dataset,
    image_path: str,
    water_row: Dict[str, float],
    device: torch.device,
    names: List[str],
    output_dir: str,
    conf_thr: float,
    csv_path: str,
):
    """执行一张图片的真正推理。"""

    # --------------------------------------------------------
    # 找到该图片在 val dataset 中的位置
    # --------------------------------------------------------
    dataset_idx, image_id, coco_name = find_dataset_image_index(
        dataset,
        os.path.basename(image_path),
    )

    print("\n========== 图片信息 ==========")
    print(f"输入图片  : {image_path}")
    print(f"COCO 名称 : {coco_name}")
    print(f"Dataset idx: {dataset_idx}")
    print(f"image_id  : {image_id}")

    # --------------------------------------------------------
    # 用现有 val dataset 取数据。
    # 这样图像预处理与验证阶段一致。
    # --------------------------------------------------------
    sample, original_target = dataset[dataset_idx]

    if not torch.is_tensor(sample):
        raise TypeError(
            "Dataset 返回的 image 不是 Tensor。\n"
            f"实际类型：{type(sample)}\n"
            "请检查 val_dataloader 的 transforms。"
        )

    target = move_target_to_device(original_target, device)

    # --------------------------------------------------------
    # 用 CSV 中的数据覆盖水质输入
    # 顺序固定：temperature, do, ph, turbidity
    # --------------------------------------------------------
    water = torch.tensor(
        [
            water_row["temperature"],
            water_row["do"],
            water_row["ph"],
            water_row["turbidity"],
        ],
        dtype=torch.float32,
        device=device,
    )

    target["water_quality"] = water
    target["image_id"] = torch.tensor(
        image_id,
        dtype=torch.int64,
        device=device,
    )

    if "orig_size" not in target:
        raise KeyError(
            "target 中没有 orig_size，无法正确把预测框还原到原图尺寸。"
        )

    print("\n========== CSV 水质数据 ==========")
    print(f"temperature = {water_row['temperature']}")
    print(f"do          = {water_row['do']}")
    print(f"ph          = {water_row['ph']}")
    print(f"turbidity   = {water_row['turbidity']}")
    print(f"water tensor= {water.detach().cpu().tolist()}")

    if sample.dim() == 3:
        samples = sample.unsqueeze(0).to(device)
    elif sample.dim() == 4 and sample.shape[0] == 1:
        samples = sample.to(device)
    else:
        raise ValueError(
            f"Dataset 返回的 image Tensor 形状异常：{tuple(sample.shape)}"
        )
    targets = [target]

    # --------------------------------------------------------
    # Query-Water 模型推理
    # 与当前 eval_curves.py 保持一致：
    #     model(samples, targets)
    # --------------------------------------------------------
    outputs = model(
        samples,
        targets,
    )

    orig_sizes = torch.stack([
        target["orig_size"]
    ]).to(device)

    result = post(
        outputs,
        orig_sizes,
    )

    # --------------------------------------------------------
    # 兼容两种 DEIM postprocessor 返回格式
    # --------------------------------------------------------
    if (
        isinstance(result, tuple)
        and len(result) == 3
        and torch.is_tensor(result[0])
    ):
        labels = result[0][0]
        boxes = result[1][0]
        scores = result[2][0]
    elif isinstance(result, list):
        item = result[0]
        labels = item["labels"]
        boxes = item["boxes"]
        scores = item["scores"]
    else:
        raise TypeError(
            "无法识别 postprocessor 返回格式："
            f"{type(result)}"
        )

    labels = labels.detach().cpu().numpy().astype(np.int64)
    boxes = boxes.detach().cpu().numpy().astype(np.float64)
    scores = scores.detach().cpu().numpy().astype(np.float64)

    # --------------------------------------------------------
    # 置信度过滤
    # --------------------------------------------------------
    keep = scores >= conf_thr

    labels_keep = labels[keep]
    boxes_keep = boxes[keep]
    scores_keep = scores[keep]

    print("\n========== 推理结果 ==========")
    print(f"置信度阈值：{conf_thr:.4f}")
    print(f"保留预测框：{len(labels_keep)}")

    predictions = []

    for i, (label, box, score) in enumerate(
        zip(labels_keep, boxes_keep, scores_keep),
        start=1,
    ):
        label_id = int(label)

        if 0 <= label_id < len(names):
            label_name = names[label_id]
        else:
            label_name = f"class_{label_id}"

        x1, y1, x2, y2 = [float(v) for v in box]

        predictions.append({
            "index": i,
            "class_id": label_id,
            "class_name": label_name,
            "confidence": float(score),
            "bbox_xyxy": [x1, y1, x2, y2],
        })

        print(
            f"[{i:02d}] {label_name:>10s}  "
            f"conf={float(score):.4f}  "
            f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    # --------------------------------------------------------
    # 打开原图并画框
    # --------------------------------------------------------
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        # 这里统一使用绿色框，方便观察
        draw_box(
            draw,
            pred["bbox_xyxy"],
            f"{pred['class_name']} {pred['confidence']:.2f}",
            color="lime",
            width=4,
        )

    # 左上角信息
    info_lines = [
        f"file: {os.path.basename(image_path)}",
        f"T={water_row['temperature']:.2f}",
        f"DO={water_row['do']:.2f}",
        f"pH={water_row['ph']:.2f}",
        f"Tur={water_row['turbidity']:.2f}",
        f"conf>={conf_thr:.3f}",
        f"detections={len(predictions)}",
    ]

    font = safe_font(22)
    y = 10

    for line in info_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle(
            [10, y, 18 + tw, y + th + 6],
            fill="black",
        )

        draw.text(
            (14, y + 2),
            line,
            fill="white",
            font=font,
        )

        y += th + 8

    # --------------------------------------------------------
    # 保存图片
    # --------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(image_path))[0]

    output_image = os.path.join(
        output_dir,
        f"{stem}_prediction.jpg",
    )

    image.save(
        output_image,
        quality=95,
    )

    # --------------------------------------------------------
    # 保存 JSON
    # --------------------------------------------------------
    output_json = os.path.join(
        output_dir,
        f"{stem}_prediction.json",
    )

    result_json = {
        "image": os.path.abspath(image_path),
        "csv": os.path.abspath(csv_path),
        "image_id": int(image_id),
        "water_quality": {
            "temperature": float(water_row["temperature"]),
            "do": float(water_row["do"]),
            "ph": float(water_row["ph"]),
            "turbidity": float(water_row["turbidity"]),
        },
        "confidence_threshold": float(conf_thr),
        "predictions": predictions,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print("\n========== 保存结果 ==========")
    print(f"预测图片：{output_image}")
    print(f"预测 JSON：{output_json}")

    return output_image, output_json


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # 路径检查
    # --------------------------------------------------------
    image_path = os.path.abspath(args.image)
    csv_path = os.path.abspath(args.csv)
    config_path = os.path.abspath(args.config)
    checkpoint_path = os.path.abspath(args.resume)
    output_dir = os.path.abspath(args.output)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在：{image_path}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"YAML 不存在：{config_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    # --------------------------------------------------------
    # device
    # --------------------------------------------------------
    device_name = args.device

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换 CPU。")
        device_name = "cpu"

    device = torch.device(device_name)

    # --------------------------------------------------------
    # 读取 CSV
    # --------------------------------------------------------
    csv_map = read_sensor_csv(csv_path)

    image_key = normalize_filename(image_path)

    if image_key not in csv_map:
        raise KeyError(
            f"CSV 中没有找到这张图片：{os.path.basename(image_path)}\n"
            "请确认 CSV 第一列文件名包含 .jpg，并且与图片文件名一致。"
        )

    water_row = csv_map[image_key]

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------
    cfg, model, post = load_model(
        config_path,
        checkpoint_path,
        device,
    )

    # --------------------------------------------------------
    # 获取 val dataset
    # --------------------------------------------------------
    loader = cfg.val_dataloader
    dataset = loader.dataset

    ann_file = find_ann_file(dataset)

    if ann_file is None:
        raise RuntimeError(
            "无法自动找到 val dataset 的 COCO annotation 文件。"
        )

    ann_file = os.path.abspath(ann_file)

    print("\n========== Dataset ==========")
    print(f"val dataset: {type(dataset)}")
    print(f"COCO JSON  : {ann_file}")

    names, category_ids = get_category_names(ann_file)

    print(f"类别名称   : {names}")
    print(f"类别 ID    : {category_ids}")

    # --------------------------------------------------------
    # 推理
    # --------------------------------------------------------
    run_single_inference(
        model=model,
        post=post,
        dataset=dataset,
        image_path=image_path,
        water_row=water_row,
        device=device,
        names=names,
        output_dir=output_dir,
        conf_thr=args.conf,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
