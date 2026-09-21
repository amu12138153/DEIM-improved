# -*- coding: utf-8 -*-
"""python eval_curves.py -c configs/deimv2/visdrone_pico.yml -r ./outputs/deimv2_hgnetv2_pico_coco/best_stg1.pth --outdir eval_out1
DEIMv2 测试集评估 + 结果总结 + 错误样本/疑似标注问题自动分析

运行示例：



======================================================================
一、指标计算方式
======================================================================

1. mAP@0.5
   在 IoU = 0.50 的条件下，对每个类别计算 AP，再对类别取平均：

       mAP@0.5 = mean(AP_class@IoU=0.50)

2. mAP@0.5:0.95
   在 IoU = 0.50, 0.55, ..., 0.95 共 10 个阈值下计算 AP，
   再对所有 IoU 阈值和类别取平均：

       mAP@0.5:0.95 = mean(AP_class,IoU)

3. AP@0.75
   在 IoU = 0.75 时，对各类别 AP 取平均。

4. IoU（交并比）

       IoU = 交集面积 / 并集面积

   本脚本的 Precision / Recall / F1 和错误样本分析默认使用：
       IoU = 0.50

5. Precision（精确率）

       Precision = TP / (TP + FP)

   TP：预测类别正确，且预测框与 GT 的 IoU >= 0.50
   FP：预测框没有匹配到任何 GT，或预测类别错误。

6. Recall（召回率）

       Recall = TP / (TP + FN)

   FN：真实 GT 没有被任何符合条件的预测框匹配。

7. F1

       F1 = 2 × Precision × Recall / (Precision + Recall)

8. Best Confidence
   本脚本先在不同 confidence 下形成 Precision-Recall 点，
   然后选择 F1 最大的位置：

       Best Confidence = argmax(F1(confidence))

   之后用该 confidence + IoU=0.50 计算最终 Precision / Recall / F1
   以及混淆矩阵。

9. 混淆矩阵
   行 = Predicted，列 = True，最后一行/列为 background。

   例如：
       Pred normal  / True hypoxia  -> hypoxia 被错分成 normal
       Pred background / True hypoxia -> hypoxia 漏检
       Pred hypoxia / True normal  -> normal 被错分成 hypoxia
       Pred hypoxia / True background -> 误检 hypoxia

10. Parameters
    所有模型参数的 numel() 总和：

       Parameters = sum(p.numel())

11. GFLOPs
    使用 THOP：

       FLOPs = 2 × MACs

    输入尺寸默认为：

       1 × 3 × 640 × 640

12. 测试集预测结果图片
    测试结束后，会将测试集每一张图片保存为带模型预测框的 JPG。
    默认使用 Best F1 对应的 confidence，只显示最终评估阈值以上的预测框。

13. 疑似标注问题
    本脚本不会武断地认定某张图片“绝对标注错误”。
    它会输出“疑似标注问题”，供人工复核。重点条件包括：

    A. 疑似类别标注问题：
       GT 类别 != 高置信度预测类别
       且预测框与 GT 的 IoU >= 0.50

    B. 疑似框标注问题：
       GT 与同类别高置信度预测有明显重叠，
       但 IoU < 0.50，提示 GT 框可能偏移、过大或过小。

    同时会把这些图片画框保存，方便直接人工检查。

13. Query-Water Fusion
    模型推理保持：

       model(samples, targets)

    因此保留 water_quality 输入。
======================================================================
"""

import argparse
import json
import os
import csv

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from torchvision.ops import box_iou

from engine.core import YAMLConfig


# ============================================================
# 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='DEIMv2 YAML 配置文件'
    )

    parser.add_argument(
        '-r',
        '--resume',
        required=True,
        help='checkpoint 路径，例如 best_stg1.pth'
    )

    parser.add_argument(
        '--outdir',
        default='eval_out',
        help='输出目录'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='Precision/Recall/F1/混淆矩阵的 IoU 阈值'
    )

    parser.add_argument(
        '--device',
        default='cuda',
        help='cuda 或 cpu'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='GFLOPs 输入尺寸'
    )

    parser.add_argument(
        '--image-dir',
        default=None,
        help='测试集 JPG 所在目录；不指定时自动尝试从 Dataset 推断'
    )

    parser.add_argument(
        '--label-error-conf',
        type=float,
        default=0.80,
        help='疑似标注类别错误的高置信度阈值，默认 0.80'
    )

    return parser.parse_args()


# ============================================================
# 查找 ann_file
# ============================================================

def find_ann_file(obj, max_depth=8, depth=0):
    """
    递归寻找 Dataset 中的 COCO annotation 文件。
    """

    if obj is None:
        return None

    if depth > max_depth:
        return None

    # --------------------------------------------------------
    # 直接属性
    # --------------------------------------------------------

    possible_attrs = [
        'ann_file',
        'anno_file',
        'annotation_file',
        'json_file',
        'ann_path'
    ]

    for attr in possible_attrs:

        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if isinstance(value, str):
            return value

    # --------------------------------------------------------
    # dataset
    # --------------------------------------------------------

    if hasattr(obj, 'dataset'):

        try:
            result = find_ann_file(
                obj.dataset,
                max_depth=max_depth,
                depth=depth + 1
            )

            if result is not None:
                return result

        except Exception:
            pass

    # --------------------------------------------------------
    # datasets
    # --------------------------------------------------------

    if hasattr(obj, 'datasets'):

        try:
            datasets = getattr(
                obj,
                'datasets'
            )

            if isinstance(
                datasets,
                (list, tuple)
            ):

                for dataset in datasets:

                    result = find_ann_file(
                        dataset,
                        max_depth=max_depth,
                        depth=depth + 1
                    )

                    if result is not None:
                        return result

        except Exception:
            pass

    return None


# ============================================================
# 查找图片目录
# ============================================================

def find_image_root(obj, max_depth=8, depth=0):
    """
    尝试从 Dataset / DataLoader 中找到图片目录。
    支持常见属性：img_folder / image_folder / image_dir / img_dir / root。
    """

    if obj is None or depth > max_depth:
        return None

    possible_attrs = [
        'img_folder',
        'image_folder',
        'image_dir',
        'img_dir',
        'root',
    ]

    for attr in possible_attrs:
        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if isinstance(value, str) and os.path.isdir(value):
            return value

    for attr in ['dataset', 'datasets']:
        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if attr == 'dataset':
            result = find_image_root(value, max_depth=max_depth, depth=depth + 1)
            if result is not None:
                return result

        elif isinstance(value, (list, tuple)):
            for item in value:
                result = find_image_root(item, max_depth=max_depth, depth=depth + 1)
                if result is not None:
                    return result

    return None


def resolve_image_path(image_root, file_name, ann_file=None):
    """尽量稳妥地解析 COCO image.file_name 对应的 JPG 路径。"""

    if not file_name:
        return None

    candidates = []

    if os.path.isabs(file_name):
        candidates.append(file_name)

    if image_root:
        candidates.append(os.path.join(image_root, file_name))

    if ann_file:
        ann_dir = os.path.dirname(os.path.abspath(ann_file))
        candidates.append(os.path.join(ann_dir, file_name))
        candidates.append(os.path.join(os.path.dirname(ann_dir), file_name))
        candidates.append(os.path.join(os.path.dirname(ann_dir), 'images', file_name))

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    return None


# ============================================================
# 加载模型
# ============================================================

def load_everything(args):

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    cfg = YAMLConfig(
        args.config,
        resume=args.resume
    )

    # --------------------------------------------------------
    # checkpoint
    # --------------------------------------------------------

    ckpt = torch.load(
        args.resume,
        map_location='cpu'
    )

    if 'ema' in ckpt:

        state = ckpt['ema']['module']

        print(
            'Checkpoint: 使用 EMA 模型'
        )

    elif 'model' in ckpt:

        state = ckpt['model']

        print(
            'Checkpoint: 使用 model 权重'
        )

    else:

        raise KeyError(
            'checkpoint 中不存在 ema 或 model'
        )

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = cfg.model

    model.load_state_dict(
        state,
        strict=True
    )

    model = model.to(
        args.device
    )

    model.eval()

    # --------------------------------------------------------
    # postprocessor
    # --------------------------------------------------------

    post = cfg.postprocessor

    post = post.to(
        args.device
    )

    post.eval()

    # --------------------------------------------------------
    # validation loader
    # --------------------------------------------------------

    loader = cfg.val_dataloader

    # --------------------------------------------------------
    # dataset
    # --------------------------------------------------------

    dataset = loader.dataset

    print()
    print('=' * 70)
    print('Validation Dataset')
    print('=' * 70)

    print(
        'DataLoader type :',
        type(loader)
    )

    print(
        'Dataset type    :',
        type(dataset)
    )

    # --------------------------------------------------------
    # ann file
    # --------------------------------------------------------

    ann_file = find_ann_file(
        dataset
    )

    if ann_file is None:

        print()
        print(
            '无法自动找到 ann_file。'
        )

        print(
            '\nDataset 属性：'
        )

        try:
            for key, value in vars(
                dataset
            ).items():
                print(
                    f'{key} = {value}'
                )
        except Exception:
            pass

        raise RuntimeError(
            '\n请根据上面的 Dataset 属性检查 '
            'COCO JSON 路径。'
        )

    ann_file = os.path.abspath(
        ann_file
    )

    if not os.path.isfile(
        ann_file
    ):

        raise FileNotFoundError(
            f'\nCOCO annotation 不存在：\n'
            f'{ann_file}'
        )

    print(
        'COCO annotation :',
        ann_file
    )

    # --------------------------------------------------------
    # coco json
    # --------------------------------------------------------

    with open(
        ann_file,
        'r',
        encoding='utf-8'
    ) as f:

        coco_json = json.load(f)

    # --------------------------------------------------------
    # categories
    # --------------------------------------------------------

    categories = sorted(
        coco_json['categories'],
        key=lambda x: x['id']
    )

    names = [
        c['name']
        for c in categories
    ]

    category_ids = [
        int(c['id'])
        for c in categories
    ]

    print(
        'Categories      :',
        names
    )

    print(
        'Category IDs    :',
        category_ids
    )

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    id2gt = {}

    for ann in coco_json['annotations']:

        image_id = int(
            ann['image_id']
        )

        category_id = int(
            ann['category_id']
        )

        x, y, w, h = ann['bbox']

        box = [
            float(x),
            float(y),
            float(x + w),
            float(y + h)
        ]

        if image_id not in id2gt:

            id2gt[image_id] = {
                'labels': [],
                'boxes': []
            }

        id2gt[image_id][
            'labels'
        ].append(
            category_id
        )

        id2gt[image_id][
            'boxes'
        ].append(
            box
        )

    # --------------------------------------------------------
    # numpy
    # --------------------------------------------------------

    for image_id in id2gt:

        id2gt[image_id][
            'labels'
        ] = np.asarray(
            id2gt[image_id]['labels'],
            dtype=np.int64
        )

        id2gt[image_id][
            'boxes'
        ] = np.asarray(
            id2gt[image_id]['boxes'],
            dtype=np.float64
        ).reshape(
            -1,
            4
        )

    print(
        'COCO images    :',
        len(coco_json['images'])
    )

    print(
        'GT images      :',
        len(id2gt)
    )

    print(
        'GT annotations :',
        len(coco_json['annotations'])
    )

    print('=' * 70)

    return (
        cfg,
        model,
        post,
        loader,
        names,
        category_ids,
        id2gt,
        coco_json,
        ann_file
    )


# ============================================================
# 收集预测
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    post,
    loader,
    device,
    id2gt
):

    all_preds = []
    all_gts = []

    print()
    print('=' * 70)
    print('开始验证集推理')
    print('=' * 70)

    for batch_idx, (
        samples,
        targets
    ) in enumerate(loader):

        # ----------------------------------------------------
        # image
        # ----------------------------------------------------

        samples = samples.to(
            device
        )

        # ----------------------------------------------------
        # targets
        #
        # 保留 water_quality
        # ----------------------------------------------------

        new_targets = []

        for target in targets:

            item = {}

            for key, value in target.items():

                if torch.is_tensor(value):

                    item[key] = value.to(
                        device
                    )

                else:

                    item[key] = value

            new_targets.append(
                item
            )

        targets = new_targets

        # ----------------------------------------------------
        # 原图尺寸
        # ----------------------------------------------------

        orig_sizes = torch.stack([
            target['orig_size']
            for target in targets
        ]).to(device)

        # ----------------------------------------------------
        # forward
        #
        # Query-Water:
        # model(samples, targets)
        # ----------------------------------------------------

        outputs = model(
            samples,
            targets
        )

        # ----------------------------------------------------
        # postprocess
        # ----------------------------------------------------

        result = post(
            outputs,
            orig_sizes
        )

        # ----------------------------------------------------
        # DEIM 返回形式 1：
        #
        # labels, boxes, scores
        # ----------------------------------------------------

        if (
            isinstance(result, tuple)
            and len(result) == 3
            and torch.is_tensor(result[0])
        ):

            labels_b, boxes_b, scores_b = result

            per_image = []

            for i in range(
                len(targets)
            ):

                per_image.append({

                    'labels':
                        labels_b[i],

                    'boxes':
                        boxes_b[i],

                    'scores':
                        scores_b[i]

                })

        # ----------------------------------------------------
        # DEIM 返回形式 2：
        #
        # list[dict]
        # ----------------------------------------------------

        else:

            per_image = result

        # ----------------------------------------------------
        # 保存每张图
        # ----------------------------------------------------

        for i, target in enumerate(
            targets
        ):

            item = per_image[i]

            labels = (
                item['labels']
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            boxes = (
                item['boxes']
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            scores = (
                item['scores']
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            image_id = int(
                target['image_id']
            )

            gt = id2gt.get(
                image_id
            )

            if gt is None:

                print(
                    f'WARNING: '
                    f'image_id={image_id} '
                    f'没有找到 GT'
                )

                gt = {

                    'labels':
                        np.zeros(
                            0,
                            dtype=np.int64
                        ),

                    'boxes':
                        np.zeros(
                            (0, 4),
                            dtype=np.float64
                        )

                }

            all_preds.append({

                'image_id':
                    image_id,

                'labels':
                    labels,

                'boxes':
                    boxes,

                'scores':
                    scores

            })

            all_gts.append(
                gt
            )

    print(
        f'完成，共处理 '
        f'{len(all_preds)} 张图'
    )

    return (
        all_preds,
        all_gts
    )


# ============================================================
# 参数量
# ============================================================

def calculate_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    non_trainable = (
        total
        - trainable
    )

    result = {

        'Parameters':
            int(total),

        'Parameters_M':
            float(total / 1e6),

        'Trainable_Parameters':
            int(trainable),

        'Trainable_Parameters_M':
            float(trainable / 1e6),

        'Non_trainable_Parameters':
            int(non_trainable),

        'Non_trainable_Parameters_M':
            float(non_trainable / 1e6)

    }

    return result


# ============================================================
# GFLOPs
# ============================================================

def calculate_gflops(
    model,
    device,
    imgsz=640
):
    """
    使用 THOP 计算 GFLOPs。

    FLOPs = 2 × MACs

    输入：
        [1, 3, imgsz, imgsz]
    """

    try:

        from thop import profile

    except ImportError:

        print()
        print(
            '没有安装 thop。'
        )

        print(
            '请运行：'
        )

        print(
            'pip install thop'
        )

        return {
            'GFLOPs': None,
            'MACs': None,
            'input_size': imgsz
        }

    model.eval()

    dummy_image = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        device=device
    )

    # --------------------------------------------------------
    # Query-Water 的 dummy water
    #
    # 顺序：
    # temperature
    # do
    # ph
    # turbidity
    # --------------------------------------------------------

    water = torch.tensor(
        [
            25.0,
            5.0,
            7.0,
            10.0
        ],
        dtype=torch.float32,
        device=device
    )

    dummy_target = [{
        'water_quality':
            water,

        'orig_size':
            torch.tensor(
                [
                    imgsz,
                    imgsz
                ],
                dtype=torch.int64,
                device=device
            ),

        'image_id':
            torch.tensor(
                0,
                dtype=torch.int64,
                device=device
            )
    }]

    try:

        macs, params = profile(
            model,
            inputs=(
                dummy_image,
                dummy_target
            ),
            verbose=False
        )

        # THOP 输出 MACs
        #
        # 这里采用：
        # FLOPs = 2 * MACs

        flops = (
            2.0
            * float(macs)
        )

        gflops = (
            flops
            / 1e9
        )

        return {

            'GFLOPs':
                float(gflops),

            'MACs':
                float(macs),

            'MACs_G':
                float(
                    macs / 1e9
                ),

            'input_size':
                imgsz,

            'FLOPs_convention':
                'FLOPs = 2 * MACs'

        }

    except Exception as e:

        print()
        print('=' * 70)
        print('GFLOPs 计算失败')
        print('=' * 70)

        print(
            repr(e)
        )

        print(
            'Parameters 仍然有效。'
        )

        print('=' * 70)

        return {

            'GFLOPs':
                None,

            'MACs':
                None,

            'MACs_G':
                None,

            'input_size':
                imgsz,

            'FLOPs_convention':
                'FLOPs = 2 * MACs',

            'error':
                repr(e)

        }



# ============================================================
# 保存完整测试集预测结果图片
# ============================================================

def save_prediction_visualizations(
    all_preds,
    coco_json,
    names,
    image_root,
    ann_file,
    outdir,
    conf_thr,
):
    """
    将测试集中每一张图片的模型预测结果画出来并保存。

    只绘制最终评估时使用的 prediction：
        score >= conf_thr

    图像中的框：
        绿色 = 模型预测框

    标签格式：
        Pred: 类别 0.92

    该目录用于快速浏览模型在整个测试集上的实际检测结果。
    如果要判断“标注是否可能有问题”，请同时查看
        error_images/00_suspected_annotation_errors
    以及
        suspected_label_errors.csv
    """

    image_info_map = {
        int(item['id']): item
        for item in coco_json.get('images', [])
    }

    output_dir = os.path.join(
        outdir,
        'prediction_images'
    )

    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    missing = 0

    for pred in all_preds:
        image_id = int(pred['image_id'])

        info = image_info_map.get(image_id, {})
        file_name = info.get(
            'file_name',
            f'{image_id}.jpg'
        )

        image_path = resolve_image_path(
            image_root,
            file_name,
            ann_file
        )

        if image_path is None:
            print(
                f'WARNING: prediction 可视化找不到图片：'
                f'{file_name}'
            )
            missing += 1
            continue

        try:
            image = Image.open(
                image_path
            ).convert('RGB')
        except Exception as e:
            print(
                f'WARNING: 无法打开图片：'
                f'{image_path}\n{repr(e)}'
            )
            missing += 1
            continue

        draw = ImageDraw.Draw(image)
        font = _safe_font(18)
        small_font = _safe_font(15)

        keep = (
            pred['scores']
            >= conf_thr
        )

        labels = pred['labels'][keep]
        boxes = pred['boxes'][keep]
        scores = pred['scores'][keep]

        # ----------------------------------------------------
        # 绘制预测框
        # ----------------------------------------------------

        for label, box, score in zip(
            labels,
            boxes,
            scores
        ):
            label_id = int(label)
            label_name = _label_name(
                label_id,
                names
            )

            _draw_box(
                draw,
                box,
                'lime',
                width=3,
                text=(
                    f'Pred: {label_name} '
                    f'{float(score):.2f}'
                ),
                font=small_font
            )

        # ----------------------------------------------------
        # 左上角显示图片信息
        # ----------------------------------------------------

        info_text = (
            f'Predictions >= {conf_thr:.3f}: '
            f'{len(labels)}'
        )

        bbox = draw.textbbox(
            (0, 0),
            info_text,
            font=font
        )

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle(
            [
                8,
                8,
                18 + text_w,
                18 + text_h
            ],
            fill='black'
        )

        draw.text(
            (13, 11),
            info_text,
            fill='white',
            font=font
        )

        output_path = os.path.join(
            output_dir,
            os.path.basename(file_name)
        )

        image.save(
            output_path,
            quality=95
        )

        saved += 1

    print()
    print('=' * 70)
    print('测试集预测结果图片已保存')
    print('=' * 70)
    print('保存目录：', output_dir)
    print('成功保存：', saved)
    print('找不到/无法打开：', missing)
    print('使用 confidence：', f'{conf_thr:.4f}')
    print('=' * 70)

    return {
        'directory': output_dir,
        'saved': int(saved),
        'missing': int(missing),
        'confidence': float(conf_thr),
    }


# ============================================================
# 单类别 PR 数据
# ============================================================

def pr_for_class(
    class_id,
    all_preds,
    all_gts,
    iou_thr
):

    scores = []
    tps = []
    fps = []

    npos = 0

    for preds, gts in zip(
        all_preds,
        all_gts
    ):

        # ----------------------------------------------------
        # GT
        # ----------------------------------------------------

        gt_mask = (
            gts['labels']
            == class_id
        )

        gt_boxes = gts[
            'boxes'
        ][gt_mask]

        npos += len(
            gt_boxes
        )

        # ----------------------------------------------------
        # prediction
        # ----------------------------------------------------

        pred_mask = (
            preds['labels']
            == class_id
        )

        pred_boxes = preds[
            'boxes'
        ][pred_mask]

        pred_scores = preds[
            'scores'
        ][pred_mask]

        if len(pred_boxes) == 0:

            continue

        # ----------------------------------------------------
        # 按置信度降序
        # ----------------------------------------------------

        order = np.argsort(
            -pred_scores,
            kind='stable'
        )

        pred_boxes = (
            pred_boxes[order]
        )

        pred_scores = (
            pred_scores[order]
        )

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        if len(gt_boxes) > 0:

            ious = box_iou(
                torch.from_numpy(
                    pred_boxes
                ).float(),

                torch.from_numpy(
                    gt_boxes
                ).float()
            ).numpy()

        else:

            ious = None

        used = np.zeros(
            len(gt_boxes),
            dtype=bool
        )

        # ----------------------------------------------------
        # matching
        # ----------------------------------------------------

        for i in range(
            len(pred_boxes)
        ):

            scores.append(
                float(
                    pred_scores[i]
                )
            )

            matched = False

            if ious is not None:

                row = ious[i].copy()

                row[
                    used
                ] = -1.0

                best_gt = int(
                    np.argmax(row)
                )

                if (
                    row[best_gt]
                    >= iou_thr
                ):

                    used[
                        best_gt
                    ] = True

                    matched = True

            if matched:

                tps.append(1)
                fps.append(0)

            else:

                tps.append(0)
                fps.append(1)

    if npos == 0:
        return None

    if len(scores) == 0:

        return {

            'scores':
                np.array([1.0]),

            't':
                np.array([0]),

            'f':
                np.array([0]),

            'p':
                np.array([0.0]),

            'r':
                np.array([0.0]),

            'npos':
                npos

        }

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    tps = np.asarray(
        tps,
        dtype=np.int64
    )

    fps = np.asarray(
        fps,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # 全局排序
    # --------------------------------------------------------

    order = np.argsort(
        -scores,
        kind='stable'
    )

    scores = scores[
        order
    ]

    tps = tps[
        order
    ]

    fps = fps[
        order
    ]

    tp_c = np.cumsum(
        tps
    )

    fp_c = np.cumsum(
        fps
    )

    precision = (
        tp_c
        /
        np.maximum(
            tp_c + fp_c,
            1e-12
        )
    )

    recall = (
        tp_c
        /
        max(
            npos,
            1
        )
    )

    return {

        'scores':
            scores,

        't':
            tps,

        'f':
            fps,

        'p':
            precision,

        'r':
            recall,

        'npos':
            npos

    }


# ============================================================
# 汇总所有类别
# ============================================================

def pool_curves(curves):

    score_list = []
    tp_list = []
    fp_list = []

    npos = 0

    for curve in curves:

        if curve is None:
            continue

        score_list.append(
            curve['scores']
        )

        tp_list.append(
            curve['t']
        )

        fp_list.append(
            curve['f']
        )

        npos += curve[
            'npos'
        ]

    if not score_list:

        return {

            'scores':
                np.array([1.0]),

            'p':
                np.array([0.0]),

            'r':
                np.array([0.0]),

            't':
                np.array([0]),

            'f':
                np.array([0]),

            'npos':
                0

        }

    scores = np.concatenate(
        score_list
    )

    tps = np.concatenate(
        tp_list
    )

    fps = np.concatenate(
        fp_list
    )

    order = np.argsort(
        -scores,
        kind='stable'
    )

    scores = scores[
        order
    ]

    tps = tps[
        order
    ]

    fps = fps[
        order
    ]

    tp_c = np.cumsum(
        tps
    )

    fp_c = np.cumsum(
        fps
    )

    precision = (
        tp_c
        /
        np.maximum(
            tp_c + fp_c,
            1e-12
        )
    )

    recall = (
        tp_c
        /
        max(
            npos,
            1
        )
    )

    return {

        'scores':
            scores,

        't':
            tps,

        'f':
            fps,

        'p':
            precision,

        'r':
            recall,

        'npos':
            npos

    }


# ============================================================
# 最佳 F1
# ============================================================

def get_best_f1(pooled):

    p = pooled['p']
    r = pooled['r']

    f1 = (
        2
        * p
        * r
        /
        np.maximum(
            p + r,
            1e-12
        )
    )

    index = int(
        np.argmax(f1)
    )

    return {

        'f1':
            float(
                f1[index]
            ),

        'confidence':
            float(
                pooled['scores'][index]
            ),

        'precision':
            float(
                p[index]
            ),

        'recall':
            float(
                r[index]
            ),

        'index':
            index

    }


# ============================================================
# 精确率 / 召回率 / F1
# ============================================================

def calculate_prf1(
    all_preds,
    all_gts,
    conf_thr,
    iou_thr
):
    """
    在固定 conf + IoU 阈值下：

        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2PR / (P + R)
    """

    TP = 0
    FP = 0
    FN = 0

    for preds, gts in zip(
        all_preds,
        all_gts
    ):

        # ----------------------------------------------------
        # confidence
        # ----------------------------------------------------

        keep = (
            preds['scores']
            >= conf_thr
        )

        pred_boxes = preds[
            'boxes'
        ][keep]

        pred_labels = preds[
            'labels'
        ][keep]

        pred_scores = preds[
            'scores'
        ][keep]

        gt_boxes = gts[
            'boxes'
        ]

        gt_labels = gts[
            'labels'
        ]

        # ----------------------------------------------------
        # 按置信度排序
        # ----------------------------------------------------

        order = np.argsort(
            -pred_scores,
            kind='stable'
        )

        pred_boxes = (
            pred_boxes[order]
        )

        pred_labels = (
            pred_labels[order]
        )

        # ----------------------------------------------------
        # GT 使用情况
        # ----------------------------------------------------

        used = np.zeros(
            len(gt_boxes),
            dtype=bool
        )

        # ----------------------------------------------------
        # 匹配 prediction
        # ----------------------------------------------------

        for k in range(
            len(pred_boxes)
        ):

            pred_box = (
                pred_boxes[
                    k:k + 1
                ]
            )

            pred_label = int(
                pred_labels[k]
            )

            matched = False

            if len(gt_boxes) > 0:

                ious = box_iou(
                    torch.from_numpy(
                        pred_box
                    ).float(),

                    torch.from_numpy(
                        gt_boxes
                    ).float()
                ).numpy()[0]

                ious[
                    used
                ] = -1.0

                best_gt = int(
                    np.argmax(
                        ious
                    )
                )

                # IoU + 类别
                if (
                    ious[best_gt]
                    >= iou_thr
                    and
                    pred_label
                    ==
                    int(
                        gt_labels[
                            best_gt
                        ]
                    )
                ):

                    used[
                        best_gt
                    ] = True

                    matched = True

            if matched:

                TP += 1

            else:

                FP += 1

        # ----------------------------------------------------
        # 漏检
        # ----------------------------------------------------

        FN += int(
            (~used).sum()
        )

    precision = (
        TP
        /
        max(
            TP + FP,
            1
        )
    )

    recall = (
        TP
        /
        max(
            TP + FN,
            1
        )
    )

    f1 = (
        2
        * precision
        * recall
        /
        max(
            precision + recall,
            1e-12
        )
    )

    return {

        'TP':
            int(TP),

        'FP':
            int(FP),

        'FN':
            int(FN),

        'Precision':
            float(precision),

        'Recall':
            float(recall),

        'F1':
            float(f1),

        'Confidence':
            float(conf_thr),

        'IoU':
            float(iou_thr)

    }


# ============================================================
# 混淆矩阵
# ============================================================

def build_confusion_matrix(
    all_preds,
    all_gts,
    nc,
    conf_thr,
    iou_thr
):

    # 最后一行/列 background
    M = np.zeros(
        (
            nc + 1,
            nc + 1
        ),
        dtype=np.int64
    )

    for preds, gts in zip(
        all_preds,
        all_gts
    ):

        keep = (
            preds['scores']
            >= conf_thr
        )

        pred_boxes = preds[
            'boxes'
        ][keep]

        pred_labels = preds[
            'labels'
        ][keep]

        pred_scores = preds[
            'scores'
        ][keep]

        gt_boxes = gts[
            'boxes'
        ]

        gt_labels = gts[
            'labels'
        ]

        used = np.zeros(
            len(gt_boxes),
            dtype=bool
        )

        order = np.argsort(
            -pred_scores,
            kind='stable'
        )

        for k in order:

            pred_label = int(
                pred_labels[k]
            )

            # ------------------------------------------------
            # 非法类别
            # ------------------------------------------------

            if (
                pred_label < 0
                or pred_label >= nc
            ):
                continue

            matched = False

            if len(gt_boxes) > 0:

                ious = box_iou(
                    torch.from_numpy(
                        pred_boxes[
                            k:k + 1
                        ]
                    ).float(),

                    torch.from_numpy(
                        gt_boxes
                    ).float()
                ).numpy()[0]

                ious[
                    used
                ] = -1.0

                best_gt = int(
                    np.argmax(
                        ious
                    )
                )

                if (
                    ious[best_gt]
                    >= iou_thr
                ):

                    true_label = int(
                        gt_labels[
                            best_gt
                        ]
                    )

                    # ------------------------------------------------
                    # 预测类别和 GT 类别一致
                    # ------------------------------------------------

                    if (
                        pred_label
                        == true_label
                    ):

                        M[
                            pred_label,
                            true_label
                        ] += 1

                        used[
                            best_gt
                        ] = True

                        matched = True

                    # ------------------------------------------------
                    # 类别错误
                    # ------------------------------------------------

                    else:

                        M[
                            pred_label,
                            true_label
                        ] += 1

                        used[
                            best_gt
                        ] = True

                        matched = True

            # ----------------------------------------------------
            # prediction FP / background
            # ----------------------------------------------------

            if not matched:

                M[
                    pred_label,
                    nc
                ] += 1

        # --------------------------------------------------------
        # FN
        # --------------------------------------------------------

        for j in range(
            len(gt_boxes)
        ):

            if not used[j]:

                true_label = int(
                    gt_labels[j]
                )

                if (
                    0 <= true_label < nc
                ):

                    M[
                        nc,
                        true_label
                    ] += 1

    return M



# ============================================================
# 错误样本与疑似标注问题分析
# ============================================================

def _label_name(label_id, names):
    if 0 <= int(label_id) < len(names):
        return names[int(label_id)]
    return f'class_{int(label_id)}'


def _safe_font(size=18):
    """尝试加载系统字体，找不到时使用 PIL 默认字体。"""
    candidates = [
        r'C:\Windows\Fonts\arial.ttf',
        r'C:\Windows\Fonts\segoeui.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]

    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass

    return ImageFont.load_default()


def _draw_box(draw, box, color, width=3, text=None, font=None):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)

    for offset in range(width):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color
        )

    if text:
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        ty = max(0, y1 - text_h - 4)
        draw.rectangle(
            [x1, ty, x1 + text_w + 8, ty + text_h + 6],
            fill=color
        )
        draw.text(
            (x1 + 4, ty + 3),
            text,
            fill='white',
            font=font
        )


def annotate_error_image(
    image_path,
    gt,
    pred_labels,
    pred_boxes,
    pred_scores,
    names,
    output_path,
    events=None
):
    """
    生成错误样本可视化：
        红色 = GT
        绿色 = prediction
    图中同时写明类别和 confidence。
    """

    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = _safe_font(18)
    small_font = _safe_font(15)

    # GT：红色
    for idx, (label, box) in enumerate(zip(gt['labels'], gt['boxes'])):
        label_name = _label_name(int(label), names)
        _draw_box(
            draw,
            box,
            'red',
            width=4,
            text=f'GT: {label_name}',
            font=font
        )

    # Prediction：绿色
    for idx, (label, box, score) in enumerate(
        zip(pred_labels, pred_boxes, pred_scores)
    ):
        label_name = _label_name(int(label), names)
        _draw_box(
            draw,
            box,
            'lime',
            width=3,
            text=f'Pred: {label_name} {float(score):.2f}',
            font=small_font
        )

    # 左上角添加错误类型说明
    if events:
        panel_x = 8
        panel_y = 8
        max_lines = 8
        lines = []

        for event in events[:max_lines]:
            lines.append(event)

        if len(events) > max_lines:
            lines.append(f'... +{len(events) - max_lines} events')

        # 计算文本框高度
        line_height = 22
        panel_h = line_height * len(lines) + 10
        draw.rectangle(
            [panel_x, panel_y, 780, panel_y + panel_h],
            fill=(0, 0, 0)
        )

        for i, line in enumerate(lines):
            draw.text(
                (panel_x + 5, panel_y + 5 + i * line_height),
                line,
                fill='white',
                font=small_font
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, quality=95)


def analyze_errors(
    all_preds,
    all_gts,
    coco_json,
    names,
    category_ids,
    image_root,
    ann_file,
    conf_thr,
    iou_thr,
    label_error_conf,
    outdir
):
    """
    按当前脚本使用的 conf + IoU 匹配规则，对每张测试图片进行错误分析。

    输出：
        1. test_predictions.json
        2. error_report.csv
        3. suspected_label_errors.csv
        4. error_images/ 下的带框图片
    """

    from collections import defaultdict

    image_info_map = {
        int(item['id']): item
        for item in coco_json.get('images', [])
    }

    name_to_index = {name: idx for idx, name in enumerate(names)}
    # 兼容类别名称不是固定顺序的情况
    normal_index = name_to_index.get('normal')
    hypoxia_index = name_to_index.get('hypoxia')

    events_by_image = defaultdict(list)
    report_rows = []
    suspected_rows = []

    category_mismatch_count = 0
    fn_count = 0
    fp_count = 0
    suspected_class_count = 0
    suspected_box_count = 0

    for preds, gts in zip(all_preds, all_gts):
        image_id = int(preds['image_id'])
        info = image_info_map.get(image_id, {})
        file_name = os.path.basename(info.get('file_name', f'{image_id}.jpg'))

        keep = preds['scores'] >= conf_thr
        pred_boxes = preds['boxes'][keep]
        pred_labels = preds['labels'][keep]
        pred_scores = preds['scores'][keep]

        gt_boxes = gts['boxes']
        gt_labels = gts['labels']
        used = np.zeros(len(gt_boxes), dtype=bool)

        if len(pred_scores) > 0:
            order = np.argsort(-pred_scores, kind='stable')
        else:
            order = np.array([], dtype=np.int64)

        for k in order:
            pred_label = int(pred_labels[k])
            pred_box = pred_boxes[k:k + 1]
            pred_score = float(pred_scores[k])

            matched = False
            best_gt = -1
            best_iou = 0.0

            if len(gt_boxes) > 0:
                ious = box_iou(
                    torch.from_numpy(pred_box).float(),
                    torch.from_numpy(gt_boxes).float()
                ).numpy()[0]
                ious[used] = -1.0
                best_gt = int(np.argmax(ious))
                best_iou = float(ious[best_gt])

                if best_iou >= iou_thr:
                    true_label = int(gt_labels[best_gt])
                    used[best_gt] = True
                    matched = True

                    if pred_label != true_label:
                        category_mismatch_count += 1
                        event_type = 'class_mismatch'
                        gt_name = _label_name(true_label, names)
                        pred_name = _label_name(pred_label, names)
                        text = (
                            f'class_mismatch | GT={gt_name} -> '
                            f'Pred={pred_name} | IoU={best_iou:.3f} | '
                            f'Conf={pred_score:.3f}'
                        )
                        events_by_image[image_id].append(text)

                        report_rows.append([
                            'class_mismatch', image_id, file_name,
                            gt_name, pred_name,
                            f'{best_iou:.6f}',
                            f'{pred_score:.6f}'
                        ])

                        # 高置信度 + 高 IoU 的类别冲突，仅标记为“疑似标注类别错误”
                        if pred_score >= label_error_conf:
                            suspected_class_count += 1
                            suspected_rows.append([
                                'suspected_class_label',
                                image_id,
                                file_name,
                                gt_name,
                                pred_name,
                                f'{best_iou:.6f}',
                                f'{pred_score:.6f}',
                                'GT类别与高置信度预测类别冲突，请人工复核'
                            ])

            if not matched:
                fp_count += 1
                pred_name = _label_name(pred_label, names)
                text = (
                    f'false_positive/background | Pred={pred_name} | '
                    f'Conf={pred_score:.3f} | bestIoU={best_iou:.3f}'
                )
                events_by_image[image_id].append(text)
                report_rows.append([
                    'false_positive_background', image_id, file_name,
                    'background', pred_name,
                    f'{best_iou:.6f}',
                    f'{pred_score:.6f}'
                ])

        # 未匹配 GT => FN / background
        for j in range(len(gt_boxes)):
            if used[j]:
                continue

            fn_count += 1
            true_label = int(gt_labels[j])
            true_name = _label_name(true_label, names)
            gt_box = gt_boxes[j:j + 1]

            # 尝试寻找最高 IoU 的同类别 prediction，判断是否可能存在框标注偏差
            same_class_best_iou = 0.0
            same_class_best_score = 0.0

            if len(preds['boxes']) > 0:
                same_mask = (
                    (preds['labels'] == true_label)
                    & (preds['scores'] >= label_error_conf)
                )
                candidate_boxes = preds['boxes'][same_mask]
                candidate_scores = preds['scores'][same_mask]

                if len(candidate_boxes) > 0:
                    ious = box_iou(
                        torch.from_numpy(gt_box).float(),
                        torch.from_numpy(candidate_boxes).float()
                    ).numpy()[0]
                    best_idx = int(np.argmax(ious))
                    same_class_best_iou = float(ious[best_idx])
                    same_class_best_score = float(candidate_scores[best_idx])

                    if 0.20 <= same_class_best_iou < iou_thr:
                        suspected_box_count += 1
                        suspected_rows.append([
                            'suspected_box_annotation',
                            image_id,
                            file_name,
                            true_name,
                            true_name,
                            f'{same_class_best_iou:.6f}',
                            f'{same_class_best_score:.6f}',
                            '同类别高置信度预测与GT有明显重叠但IoU<0.50，请人工复核框位置/尺寸'
                        ])

                        events_by_image[image_id].append(
                            f'possible_box_issue | GT={true_name} | '
                            f'sameClassIoU={same_class_best_iou:.3f} | '
                            f'Conf={same_class_best_score:.3f}'
                        )

            text = (
                f'FN/background | GT={true_name} | '
                f'bestSameClassIoU={same_class_best_iou:.3f}'
            )
            events_by_image[image_id].append(text)
            report_rows.append([
                'false_negative_background', image_id, file_name,
                true_name, 'background',
                f'{same_class_best_iou:.6f}',
                f'{same_class_best_score:.6f}'
            ])

    # --------------------------------------------------------
    # 输出 COCO prediction JSON
    # --------------------------------------------------------
    coco_results = []
    label_to_category_id = {
        idx: int(category_ids[idx])
        for idx in range(len(category_ids))
    }

    for pred in all_preds:
        image_id = int(pred['image_id'])
        for label, box, score in zip(
            pred['labels'],
            pred['boxes'],
            pred['scores']
        ):
            label = int(label)
            if label not in label_to_category_id:
                continue

            x1, y1, x2, y2 = [float(x) for x in box]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            coco_results.append({
                'image_id': image_id,
                'category_id': label_to_category_id[label],
                'bbox': [x1, y1, w, h],
                'score': float(score)
            })

    pred_json_path = os.path.join(outdir, 'test_predictions.json')
    with open(pred_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_results, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # 输出报告 CSV
    # --------------------------------------------------------
    report_path = os.path.join(outdir, 'error_report.csv')
    with open(report_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'type', 'image_id', 'file_name',
            'true_label', 'pred_label', 'iou', 'confidence'
        ])
        writer.writerows(report_rows)

    suspected_path = os.path.join(outdir, 'suspected_label_errors.csv')
    with open(suspected_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'type', 'image_id', 'file_name',
            'true_label', 'pred_label', 'iou', 'confidence', 'reason'
        ])
        writer.writerows(suspected_rows)

    # --------------------------------------------------------
    # 生成错误样本图片
    # --------------------------------------------------------
    folders = {
        'class_mismatch': os.path.join(outdir, 'error_images', '01_class_mismatch'),
        'false_negative_background': os.path.join(outdir, 'error_images', '02_false_negative_background'),
        'false_positive_background': os.path.join(outdir, 'error_images', '03_false_positive_background'),
    }

    # 疑似标注错误 / 框错误单独输出
    suspected_dir = os.path.join(outdir, 'error_images', '00_suspected_annotation_errors')
    os.makedirs(suspected_dir, exist_ok=True)
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # 为每张图建立 pred / gt 快速索引
    pred_map = {int(x['image_id']): x for x in all_preds}
    gt_map = {int(x['image_id']): x for x in all_gts}

    image_events_types = defaultdict(set)
    for row in report_rows:
        image_events_types[int(row[1])].add(row[0])

    suspected_image_ids = set()
    for row in suspected_rows:
        suspected_image_ids.add(int(row[1]))

    for image_id, event_types in image_events_types.items():
        info = image_info_map.get(image_id, {})
        file_name = os.path.basename(info.get('file_name', f'{image_id}.jpg'))
        image_path = resolve_image_path(image_root, info.get('file_name'), ann_file)

        if image_path is None:
            print(f'WARNING: 找不到图片：{info.get("file_name", file_name)}')
            continue

        pred_item = pred_map[image_id]
        pred_keep = pred_item['scores'] >= conf_thr

        gt_item = gt_map[image_id]

        events = events_by_image.get(image_id, [])

        # 针对每一种错误类型复制一份标注图
        for event_type in event_types:
            folder = folders.get(event_type)
            if folder is None:
                continue

            out_path = os.path.join(folder, file_name)
            annotate_error_image(
                image_path,
                gt_item,
                pred_item['labels'][pred_keep],
                pred_item['boxes'][pred_keep],
                pred_item['scores'][pred_keep],
                names,
                out_path,
                events=events
            )

        # 疑似标注问题单独输出
        if image_id in suspected_image_ids:
            out_path = os.path.join(suspected_dir, file_name)
            annotate_error_image(
                image_path,
                gt_item,
                pred_item['labels'][pred_keep],
                pred_item['boxes'][pred_keep],
                pred_item['scores'][pred_keep],
                names,
                out_path,
                events=events
            )

    return {
        'num_class_mismatch': int(category_mismatch_count),
        'num_false_negative': int(fn_count),
        'num_false_positive': int(fp_count),
        'num_suspected_class_label': int(suspected_class_count),
        'num_suspected_box_annotation': int(suspected_box_count),
        'prediction_json': pred_json_path,
        'error_report': report_path,
        'suspected_label_errors': suspected_path,
    }


# ============================================================
# 单类别 PR 数据
# ============================================================


# ============================================================
# 主程序
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # device
    # --------------------------------------------------------
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA 不可用，自动切换 CPU')
        args.device = 'cpu'

    device = torch.device(args.device)

    # --------------------------------------------------------
    # output
    # --------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)

    # ========================================================
    # 1. 加载模型 + 测试数据
    # ========================================================
    (
        cfg,
        model,
        post,
        loader,
        names,
        category_ids,
        id2gt,
        coco_json,
        ann_file
    ) = load_everything(args)

    dataset = loader.dataset

    # 优先使用命令行指定的图片目录，否则自动探测
    image_root = args.image_dir
    if image_root is None:
        image_root = find_image_root(dataset)

    if image_root is not None:
        image_root = os.path.abspath(image_root)
        print('Image directory:', image_root)
    else:
        print('WARNING: 无法自动确定图片目录。')
        print('          建议运行时增加 --image-dir "你的test图片目录"')

    # ========================================================
    # 2. Parameters
    # ========================================================
    parameter_info = calculate_parameters(model)

    # ========================================================
    # 3. GFLOPs
    # ========================================================
    complexity = calculate_gflops(
        model,
        device,
        args.imgsz
    )

    # ========================================================
    # 4. 测试集推理
    # ========================================================
    all_preds, all_gts = collect_predictions(
        model,
        post,
        loader,
        device,
        id2gt
    )

    n_images = len(all_preds)
    n_gt = sum(len(x['labels']) for x in all_gts)
    n_pred = sum(len(x['labels']) for x in all_preds)

    # ========================================================
    # 5. COCO prediction + COCOeval
    # ========================================================
    coco_results = []

    for pred in all_preds:
        image_id = int(pred['image_id'])

        for label, box, score in zip(
            pred['labels'],
            pred['boxes'],
            pred['scores']
        ):
            label = int(label)

            if label < 0 or label >= len(category_ids):
                continue

            category_id = int(category_ids[label])
            x1, y1, x2, y2 = [float(v) for v in box]

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            if w <= 0 or h <= 0:
                continue

            coco_results.append({
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x1, y1, w, h],
                'score': float(score)
            })

    coco_metrics = {
        'mAP@0.5': None,
        'mAP@0.5:0.95': None,
        'AP@0.75': None
    }

    if len(coco_results) > 0:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt_tmp = os.path.join(args.outdir, '_gt_tmp.json')
        pred_tmp = os.path.join(args.outdir, '_pred_tmp.json')

        with open(gt_tmp, 'w', encoding='utf-8') as f:
            json.dump(coco_json, f, ensure_ascii=False)

        with open(pred_tmp, 'w', encoding='utf-8') as f:
            json.dump(coco_results, f, ensure_ascii=False)

        coco_gt = COCO(gt_tmp)
        coco_dt = coco_gt.loadRes(pred_tmp)
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        coco_metrics = {
            'mAP@0.5': float(evaluator.stats[1]),
            'mAP@0.5:0.95': float(evaluator.stats[0]),
            'AP@0.75': float(evaluator.stats[2])
        }

        for tmp in [gt_tmp, pred_tmp]:
            try:
                os.remove(tmp)
            except Exception:
                pass
    else:
        print('警告：没有预测框，COCOeval 无法计算。')

    # ========================================================
    # 6. Best F1 + Precision/Recall/F1
    # ========================================================
    curves = []
    for class_id in category_ids:
        curves.append(
            pr_for_class(
                class_id,
                all_preds,
                all_gts,
                args.iou
            )
        )

    pooled = pool_curves(curves)
    best_f1_info = get_best_f1(pooled)

    prf1 = calculate_prf1(
        all_preds,
        all_gts,
        conf_thr=best_f1_info['confidence'],
        iou_thr=args.iou
    )

    # ========================================================
    # 7. 混淆矩阵（只保存数值，不再绘制图片）
    # ========================================================
    M = build_confusion_matrix(
        all_preds,
        all_gts,
        len(names),
        best_f1_info['confidence'],
        args.iou
    )

    confusion_csv = os.path.join(args.outdir, 'confusion_matrix.csv')
    with open(confusion_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted \\ True'] + names + ['background'])
        labels = names + ['background']
        for i, row in enumerate(M):
            writer.writerow([labels[i]] + [int(x) for x in row])

    # ========================================================
    # 8. 保存整个测试集的模型预测结果图片
    # ========================================================
    visualization_info = save_prediction_visualizations(
        all_preds,
        coco_json,
        names,
        image_root,
        ann_file,
        args.outdir,
        conf_thr=best_f1_info['confidence']
    )

    # ========================================================
    # 9. 错误分析 + 带框图片 + prediction JSON
    # ========================================================
    error_info = analyze_errors(
        all_preds,
        all_gts,
        coco_json,
        names,
        category_ids,
        image_root,
        ann_file,
        conf_thr=best_f1_info['confidence'],
        iou_thr=args.iou,
        label_error_conf=args.label_error_conf,
        outdir=args.outdir
    )

    # ========================================================
    # 9. 总结 JSON
    # ========================================================
    summary = {
        'checkpoint': os.path.abspath(args.resume),
        'config': os.path.abspath(args.config),
        'annotation': os.path.abspath(ann_file),
        'image_dir': image_root,
        'num_images': int(n_images),
        'num_gt': int(n_gt),
        'num_predictions': int(n_pred),
        'model_complexity': {
            **parameter_info,
            **complexity
        },
        'COCO': {
            'mAP@0.5': None if coco_metrics['mAP@0.5'] is None else round(coco_metrics['mAP@0.5'], 6),
            'mAP@0.5:0.95': None if coco_metrics['mAP@0.5:0.95'] is None else round(coco_metrics['mAP@0.5:0.95'], 6),
            'AP@0.75': None if coco_metrics['AP@0.75'] is None else round(coco_metrics['AP@0.75'], 6),
        },
        'DetectionMetrics': {
            'IoU': float(prf1['IoU']),
            'Confidence': float(prf1['Confidence']),
            'TP': int(prf1['TP']),
            'FP': int(prf1['FP']),
            'FN': int(prf1['FN']),
            'Precision': round(prf1['Precision'], 6),
            'Recall': round(prf1['Recall'], 6),
            'F1': round(prf1['F1'], 6),
        },
        'BestF1': {
            'F1': round(best_f1_info['f1'], 6),
            'confidence': round(best_f1_info['confidence'], 6),
            'Precision': round(best_f1_info['precision'], 6),
            'Recall': round(best_f1_info['recall'], 6),
        },
        'ConfusionMatrix': {
            'labels': names + ['background'],
            'matrix_pred_rows_true_cols': M.tolist()
        },
        'Visualization': {
            **visualization_info
        },
        'ErrorAnalysis': {
            **error_info,
            'label_error_conf': float(args.label_error_conf)
        },
        'Categories': names,
        'Category_IDs': category_ids,
    }

    summary_path = os.path.join(args.outdir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ========================================================
    # 10. 最终打印
    # ========================================================
    print()
    print('=' * 70)
    print('最终测试集评估结果')
    print('=' * 70)

    print(f'Images = {n_images}')
    print(f'GT = {n_gt}')
    print(f'Predictions = {n_pred}')
    print()

    print(f"Parameters = {parameter_info['Parameters_M']:.4f} M")
    print(
        f"GFLOPs = {complexity['GFLOPs']:.4f}"
        if complexity['GFLOPs'] is not None
        else 'GFLOPs = None'
    )
    print()

    print(f"mAP@0.5 = {coco_metrics['mAP@0.5']}")
    print(f"mAP@0.5:0.95 = {coco_metrics['mAP@0.5:0.95']}")
    print(f"AP@0.75 = {coco_metrics['AP@0.75']}")
    print()

    print(f"IoU = {prf1['IoU']:.2f}")
    print(f"Best Confidence = {best_f1_info['confidence']:.4f}")
    print(f"TP = {prf1['TP']}")
    print(f"FP = {prf1['FP']}")
    print(f"FN = {prf1['FN']}")
    print(f"Precision = {prf1['Precision']:.4f}")
    print(f"Recall = {prf1['Recall']:.4f}")
    print(f"F1 = {prf1['F1']:.4f}")
    print()

    print('---------------- Error Analysis ----------------')
    print(f"Class mismatch             = {error_info['num_class_mismatch']}")
    print(f"False negative / background = {error_info['num_false_negative']}")
    print(f"False positive / background = {error_info['num_false_positive']}")
    print(f"Suspected class label      = {error_info['num_suspected_class_label']}")
    print(f"Suspected box annotation   = {error_info['num_suspected_box_annotation']}")
    print()

    print(f'Summary              : {summary_path}')
    print(f'Prediction JSON      : {error_info["prediction_json"]}')
    print(f'Error report         : {error_info["error_report"]}')
    print(f'Suspected labels     : {error_info["suspected_label_errors"]}')
    print(f'Confusion matrix CSV : {confusion_csv}')
    print(f'全部预测结果图片     : {visualization_info["directory"]}')
    print()
    print('带框错误图片目录：')
    print(os.path.join(args.outdir, 'error_images'))
    print('=' * 70)


# ============================================================

if __name__ == '__main__':
    main()
