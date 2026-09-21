# -*- coding: utf-8 -*-
"""
把 DEIMv2 训练日志(log.txt 或控制台输出重定向文件)解析成 YOLO 风格训练曲线图。

用法:
    python plot_training_log.py --log outputs/xxx/log.txt --out 训练曲线.png
"""
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt


def parse_log(path):
    """返回 (epoch_losses, evals)
    epoch_losses: {epoch: [每次迭代的loss, ...]}
    evals: [{'ap5095':..,'ap50':..,'ap75':..,'ar':..}, ...] 按评估顺序
    """
    epoch_losses = {}
    evals = []
    pending = {}

    pat_epoch = re.compile(r'Epoch:\s*\[(\d+)\]')
    pat_loss = re.compile(r'\bloss:\s*([\d.]+)')
    pat_val = re.compile(r'=\s*(-?\d+\.\d+|nan)\s*$')

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat_epoch.search(line)
            if m:
                lm = pat_loss.search(line)
                if lm:
                    ep = int(m.group(1))
                    epoch_losses.setdefault(ep, []).append(float(lm.group(1)))
            if 'Average Precision' in line or 'Average Recall' in line:
                vm = pat_val.search(line)
                if vm is None:
                    continue
                val = float(vm.group(1))
                if 'Precision' in line and '0.50:0.95' in line:
                    pending = {'ap5095': val}
                elif 'Precision' in line and re.search(r'IoU=0\.50\s*\|', line):
                    pending['ap50'] = val
                elif 'Precision' in line and '0.75' in line:
                    pending['ap75'] = val
                elif 'Recall' in line and 'ap5095' in pending:
                    pending['ar'] = val
                    evals.append(pending)
                    pending = {}
    return epoch_losses, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='训练日志路径')
    ap.add_argument('--out', default='training_curves.png', help='输出图片路径')
    args = ap.parse_args()

    epoch_losses, evals = parse_log(args.log)
    assert epoch_losses, '日志里没有找到 Epoch 训练行, 请确认传入的是完整训练日志'

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- loss 曲线(每次迭代 + 每轮均值) ----
    it_x, it_y = [], []
    for ep in sorted(epoch_losses):
        losses = epoch_losses[ep]
        xs = ep + (np.arange(len(losses)) + 1) / (len(losses) + 1)
        it_x.extend(xs)
        it_y.extend(losses)
    axes[0].plot(it_x, it_y, '.', color='steelblue', alpha=0.25, ms=3, label='per-iter')
    ep_x = sorted(epoch_losses)
    ep_y = [float(np.mean(epoch_losses[e])) for e in ep_x]
    axes[0].plot(ep_x, ep_y, 'o-', color='tab:blue', lw=2, ms=4, label='epoch mean')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('train/loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ---- 指标曲线 ----
    if evals:
        x = np.arange(len(evals))
        for key, label, color in [
            ('ap50',   'mAP@0.5',      'tab:orange'),
            ('ap5095', 'mAP@0.5:0.95', 'tab:blue'),
            ('ap75',   'AP@0.75',      'tab:green'),
            ('ar',     'AR@0.5:0.95',  'tab:red'),
        ]:
            y = [e.get(key, np.nan) for e in evals]
            axes[1].plot(x, y, 'o-', ms=3, lw=1.8, color=color, label=label)
        ap50_list = [e.get('ap50', np.nan) for e in evals]
        best = int(np.nanargmax(ap50_list))
        axes[1].set_title(f'metrics (best mAP@0.5 = {ap50_list[best]:.4f} @ epoch {best})')
        axes[1].set_ylim(0, 1.0)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'no eval blocks found in log', ha='center', va='center')
    axes[1].set_xlabel('Epoch')

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f'已保存: {args.out}')
    print(f'解析到 {sum(len(v) for v in epoch_losses.values())} 条迭代记录, {len(evals)} 次评估')


if __name__ == '__main__':
    main()
