# -*- coding: utf-8 -*-
"""
DEIMv2 验证集评估可视化(YOLO 风格):
PR 曲线 / F1-置信度曲线 / P-置信度曲线 / R-置信度曲线 / 混淆矩阵(计数+归一化) / 指标汇总 summary.json

在 DEIMv2 项目根目录运行:
    python eval_curves.py -c configs/deimv2/visdrone_pico.yml -r outputs/xxx/best_stg1.pth --outdir eval_out

说明:
- 类别名自动从验证集 coco json 的 categories 读取(按 id 排序)
- 预测框置信度阈值取 all-classes F1 最大值处(与 YOLO 混淆矩阵口径一致)
- AP 为 IoU=0.5 单阈值, 与 YOLO 的 mAP@0.5 同口径; mAP@0.5:0.95 请看训练日志
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

from engine.core import YAMLConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True, help='训练用的 yml 配置')
    p.add_argument('-r', '--resume', required=True, help='权重路径(best_stg1.pth 等)')
    p.add_argument('--outdir', default='eval_out', help='图片输出目录')
    p.add_argument('--iou', type=float, default=0.5, help='匹配 IoU 阈值')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


# ---------------- 模型与数据 ----------------

def load_everything(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    ckpt = torch.load(args.resume, map_location='cpu')
    state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    cfg.model.load_state_dict(state)
    model = cfg.model.to(args.device).eval()
    post = cfg.postprocessor.to(args.device).eval()
    loader = cfg.val_dataloader

    ann_file = cfg.yaml_cfg['val_dataset']['ann_file']
    with open(ann_file, 'r', encoding='utf-8') as f:
        js = json.load(f)
    cats = sorted(js['categories'], key=lambda c: c['id'])
    names = [c['name'] for c in cats]
    id2wh = {im['id']: (im['width'], im['height']) for im in js['images']}
    return model, post, loader, names, id2wh


@torch.no_grad()
def collect(model, post, loader, device, id2wh):
    """跑完整个验证集, 返回每张图的预测与真值(绝对坐标 xyxy)"""
    all_preds, all_gts = [], []
    for samples, targets in loader:
        samples = samples.to(device)
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                   for t in targets]
        orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        try:
            outputs = model(samples, targets)  # eval 也传 targets: 取 water_quality
        except TypeError:
            outputs = model(samples)
        res = post(outputs, orig_sizes)

        if isinstance(res, tuple) and len(res) == 3 and torch.is_tensor(res[0]):
            labels_b, boxes_b, scores_b = res
            per_image = [{'labels': labels_b[j], 'boxes': boxes_b[j], 'scores': scores_b[j]}
                         for j in range(len(targets))]
        else:
            per_image = res  # list of dict

        for i, t in enumerate(targets):
            item = per_image[i]
            pl = item['labels'].cpu().numpy()
            pb = item['boxes'].cpu().numpy()
            ps = item['scores'].cpu().numpy()

            w, h = id2wh[int(t['image_id'])]
            gb = t['boxes'].cpu().numpy()  # 归一化 cxcywh
            gx1 = (gb[:, 0] - gb[:, 2] / 2) * w
            gy1 = (gb[:, 1] - gb[:, 3] / 2) * h
            gx2 = (gb[:, 0] + gb[:, 2] / 2) * w
            gy2 = (gb[:, 1] + gb[:, 3] / 2) * h

            all_preds.append({'labels': pl, 'boxes': pb, 'scores': ps})
            all_gts.append({'labels': t['labels'].cpu().numpy(),
                            'boxes': np.stack([gx1, gy1, gx2, gy2], 1)})
    return all_preds, all_gts


# ---------------- 指标计算 ----------------

def pr_for_class(c, all_preds, all_gts, iou_thr):
    """按置信度降序匹配, 返回 scores/p/r/t/f 数组与该类 GT 数"""
    scores, tps, fps, npos = [], [], [], 0
    for preds, gts in zip(all_preds, all_gts):
        npos += int((gts['labels'] == c).sum())
        m = preds['labels'] == c
        pb, ps = preds['boxes'][m], preds['scores'][m]
        if len(pb) == 0:
            continue
        order = np.argsort(-ps, kind='stable')
        pb, ps = pb[order], ps[order]
        gb = gts['boxes'][gts['labels'] == c]
        used = np.zeros(len(gb), dtype=bool)
        ious = box_iou(torch.from_numpy(pb), torch.from_numpy(gb)).numpy() if len(gb) else None
        for k in range(len(pb)):
            scores.append(float(ps[k]))
            hit = False
            if ious is not None:
                row = ious[k].copy()
                row[used] = -1.0
                j = int(np.argmax(row))
                if row[j] >= iou_thr:
                    used[j] = True
                    hit = True
            tps.append(int(hit))
            fps.append(int(not hit))
    if npos == 0:
        return None
    if len(scores) == 0:
        return {'scores': np.array([1.0]), 'p': np.array([0.0]), 'r': np.array([0.0]),
                't': np.array([0]), 'f': np.array([0]), 'npos': npos, 'ap': 0.0}
    s = np.array(scores)
    t = np.array(tps)
    f = np.array(fps)
    o = np.argsort(-s, kind='stable')
    s, t, f = s[o], t[o], f[o]
    tp_c = np.cumsum(t)
    fp_c = np.cumsum(f)
    p = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    r = tp_c / npos
    return {'scores': s, 'p': p, 'r': r, 't': t, 'f': f, 'npos': npos,
            'ap': compute_ap(r, p)}


def compute_ap(r, p):
    """VOC/COCO 风格 AP(包络 + 梯形)"""
    mrec = np.concatenate(([0.0], r, [1.0]))
    mpre = np.concatenate(([0.0], p, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def confusion_matrix(all_preds, all_gts, nc, conf_thr, iou_thr):
    """YOLO 风格混淆矩阵: 行=预测, 列=真值, 最后一行/列=background"""
    M = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    for preds, gts in zip(all_preds, all_gts):
        keep = preds['scores'] >= conf_thr
        pb, pl = preds['boxes'][keep], preds['labels'][keep]
        psc = preds['scores'][keep]
        gb, gl = gts['boxes'], gts['labels']
        used = np.zeros(len(gb), dtype=bool)
        order = np.argsort(-psc, kind='stable')
        for k in order:
            if len(gb):
                row = box_iou(torch.from_numpy(pb[k:k + 1]), torch.from_numpy(gb)).numpy()[0]
                row[used] = -1.0
                j = int(np.argmax(row))
                if row[j] >= iou_thr:
                    M[pl[k], gl[j]] += 1
                    used[j] = True
                    continue
            M[pl[k], nc] += 1  # 预测了目标但真值是背景
        for j in range(len(gb)):
            if not used[j]:
                M[nc, gl[j]] += 1  # 真值目标被漏检
    return M


# ---------------- 画图 ----------------

def plot_conf_metric(curves, pooled, names, which, outdir):
    """which in {'f1','p','r'}: 画 指标-置信度 曲线"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = plt.cm.tab10.colors

    def metric(c):
        if which == 'p':
            return c['p']
        if which == 'r':
            return c['r']
        return 2 * c['p'] * c['r'] / np.maximum(c['p'] + c['r'], 1e-9)

    for i, c in enumerate(curves):
        if c is not None:
            ax.plot(c['scores'], metric(c), lw=1, color=colors[i % 10], label=names[i])
    yall = metric(pooled)
    ax.plot(pooled['scores'], yall, lw=2.5, color='blue',
            label=f'all classes {yall.max():.2f} at {pooled["scores"][int(np.argmax(yall))]:.3f}')
    title = {'f1': 'F1-Confidence Curve', 'p': 'Precision-Confidence Curve',
             'r': 'Recall-Confidence Curve'}[which]
    ax.set_title(title)
    ax.set_xlabel('Confidence')
    ax.set_ylabel({'f1': 'F1', 'p': 'Precision', 'r': 'Recall'}[which])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, f'Box{which.upper()}_curve.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_pr(curves, pooled, names, mAP50, outdir):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i, c in enumerate(curves):
        if c is not None:
            ax.plot(c['r'], c['p'], lw=1, color=colors[i % 10], label=f"{names[i]} {c['ap']:.3f}")
    ax.plot(pooled['r'], pooled['p'], lw=2.5, color='blue',
            label=f'all classes {mAP50:.3f} mAP@0.5')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, 'BoxPR_curve.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_cm(M, names, outdir, normalize=False):
    labels = names + ['background']
    disp = M.astype(float)
    fmt = '{:d}'
    if normalize:
        colsum = M.sum(0, keepdims=True)
        disp = np.divide(M.astype(float), colsum, out=np.zeros((M.shape), dtype=float),
                         where=colsum > 0)
        fmt = '{:.2f}'
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(disp, cmap='Blues', vmin=0, vmax=1.0 if normalize else None)
    thresh = 0.5 if normalize else (disp.max() / 2 if disp.max() > 0 else 0.5)
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            v = disp[i, j]
            txt = fmt.format(v) if normalize else fmt.format(int(v))
            ax.text(j, i, txt, ha='center', va='center', fontsize=11,
                    color='white' if v > thresh else 'black')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix' + (' Normalized' if normalize else ''))
    fig.colorbar(im)
    fig.tight_layout()
    out = os.path.join(outdir, 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------- 主流程 ----------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    model, post, loader, names, id2wh = load_everything(args)
    nc = len(names)
    print(f'类别({nc}): {names}')

    all_preds, all_gts = collect(model, post, loader, args.device, id2wh)
    n_img = len(all_preds)
    n_gt = sum(len(g['labels']) for g in all_gts)
    print(f'验证集: {n_img} 张图, {n_gt} 个目标')

    # 每类 PR + 汇总
    curves = []
    pool_s, pool_t, pool_f, pool_npos = [], [], [], 0
    for c in range(nc):
        cur = pr_for_class(c, all_preds, all_gts, args.iou)
        curves.append(cur)
        if cur is not None and len(cur['scores']) and cur['t'].sum() + cur['f'].sum() > 0:
            pool_s.append(cur['scores'])
            pool_t.append(cur['t'])
            pool_f.append(cur['f'])
            pool_npos += cur['npos']

    aps = [c['ap'] if c is not None else 0.0 for c in curves]
    mAP50 = float(np.mean(aps))

    if pool_s:
        s = np.concatenate(pool_s)
        t = np.concatenate(pool_t)
        f = np.concatenate(pool_f)
        o = np.argsort(-s, kind='stable')
        s, t, f = s[o], t[o], f[o]
        tp_c = np.cumsum(t)
        fp_c = np.cumsum(f)
        pooled = {'scores': s,
                  'p': tp_c / np.maximum(tp_c + fp_c, 1e-9),
                  'r': tp_c / max(pool_npos, 1)}
    else:
        pooled = {'scores': np.array([1.0]), 'p': np.array([0.0]), 'r': np.array([0.0])}

    # 最佳 F1 阈值
    f1 = 2 * pooled['p'] * pooled['r'] / np.maximum(pooled['p'] + pooled['r'], 1e-9)
    best_i = int(np.argmax(f1))
    best_conf = float(pooled['scores'][best_i])
    best_f1 = float(f1[best_i])

    # 出图
    outs = []
    outs.append(plot_pr(curves, pooled, names, mAP50, args.outdir))
    for which in ['f1', 'p', 'r']:
        outs.append(plot_conf_metric(curves, pooled, names, which, args.outdir))
    M = confusion_matrix(all_preds, all_gts, nc, best_conf, args.iou)
    outs.append(plot_cm(M, names, args.outdir, normalize=False))
    outs.append(plot_cm(M, names, args.outdir, normalize=True))

    # 汇总
    summary = {
        'note': 'AP 为 IoU=0.5 单阈值, 与 YOLO mAP@0.5 同口径',
        'checkpoint': os.path.abspath(args.resume),
        'num_images': n_img,
        'num_gt': int(n_gt),
        'mAP@0.5': round(mAP50, 4),
        'per_class_AP50': {names[i]: round(aps[i], 4) for i in range(nc)},
        'best_F1': {'f1': round(best_f1, 4), 'conf': round(best_conf, 3)},
    }
    with open(os.path.join(args.outdir, 'summary.json'), 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print('\n===== 评估结果 =====')
    print(f"mAP@0.5 = {summary['mAP@0.5']}")
    for k, v in summary['per_class_AP50'].items():
        print(f'  {k}: AP50 = {v}')
    print(f"best F1 = {best_f1:.3f} @ conf {best_conf:.3f}")
    print('\n已生成:')
    for o in outs:
        print(' ', o)
    print(' ', os.path.join(args.outdir, 'summary.json'))


if __name__ == '__main__':
    main()
