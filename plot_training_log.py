# -*- coding: utf-8 -*-
"""
把 DEIMv2 训练日志解析成 YOLO 风格的多子图训练曲线。

兼容两种日志格式:
  A. 每轮一行 JSON (det_solver.py 里 json.dumps(log_stats) 写出的 log.txt)
  B. 控制台文本格式 (Epoch: [0] ... loss: x.xxx + Average Precision 行)

用法:
    python plot_training_log.py --log outputs/xxx/log.txt --out 训练曲线.png
"""

import argparse
import ast
import json
import re

import numpy as np
import matplotlib.pyplot as plt


# ---------------- 中文字体 ----------------

def setup_chinese_font():
    from matplotlib import font_manager, rcParams

    candidates = [
        'Microsoft YaHei', 'SimHei', 'DengXian', 'SimSun',
        'KaiTi', 'FangSong',
        'PingFang SC', 'Hiragino Sans GB',
        'Source Han Sans SC', 'Source Han Sans CN',
        'Noto Sans CJK SC', 'Noto Sans SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Arial Unicode MS',
    ]

    installed = {f.name for f in font_manager.fontManager.ttflist}
    rcParams['axes.unicode_minus'] = False

    for name in candidates:
        if name in installed:
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.sans-serif'] = [name] + list(
                rcParams.get('font.sans-serif', [])
            )
            return True

    return False


USE_CN = setup_chinese_font()


def T(zh, en):
    """有中文字体用中文，没有就用英文。"""
    return zh if USE_CN else en


# =========================================================
# 格式 A：每轮一行 JSON
# =========================================================

def parse_json_lines(path):
    records = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            if not line.startswith('{'):
                continue

            try:
                d = json.loads(line)
            except Exception:
                try:
                    d = ast.literal_eval(line)
                except Exception:
                    continue

            if (
                isinstance(d, dict)
                and 'epoch' in d
                and ('train_loss' in d or 'train_main_loss' in d)
            ):
                records.append(d)

    records.sort(key=lambda d: d['epoch'])

    return records


# =========================================================
# YOLO 风格 F1 计算
# =========================================================

def calculate_f1(precision, recall):
    """
    按 YOLO 常用公式计算 F1：

        F1 = 2 * P * R / (P + R)

    对 NaN / 0 做安全处理。
    """
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)

    denominator = precision + recall

    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0
    )

    return f1


def plot_json_records(records, out):

    ep = [r['epoch'] for r in records]

    def get(key):
        return [r.get(key, np.nan) for r in records]

    # -----------------------------------------------------
    # 计算 YOLO 风格 P / R / F1
    # -----------------------------------------------------

    precision = np.array(
        get('test_precision_50_at_f1'),
        dtype=float
    )

    recall = np.array(
        get('test_recall_50_at_f1'),
        dtype=float
    )

    # 不再直接读取 test_f1_50_max
    # 而是按照 YOLO 公式重新计算
    f1 = calculate_f1(precision, recall)

    # -----------------------------------------------------
    # 创建 2×3 子图
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        2, 3,
        figsize=(17, 8.5)
    )

    def panel(ax, keys_labels, title, ylabel=None):

        drew = False

        for key, label in keys_labels:

            y = np.array(
                get(key),
                dtype=float
            )

            if np.isnan(y).all():
                continue

            ax.plot(
                ep,
                y,
                'o-',
                ms=3,
                lw=1.6,
                label=label
            )

            drew = True

        ax.set_title(title)
        ax.set_xlabel('epoch')

        if ylabel:
            ax.set_ylabel(ylabel)

        if drew:
            ax.legend(fontsize=8)

        else:
            ax.text(
                0.5,
                0.5,
                'no data',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='gray'
            )

        ax.grid(alpha=0.3)

    # =====================================================
    # 1. train / test 主损失
    # =====================================================

    panel(
        axes[0, 0],
        [
            ('train_main_loss', 'train/main_loss'),
            ('test_main_loss', 'test/main_loss')
        ],
        T(
            '主损失对比 (train vs test, 同口径)',
            'main loss (train vs test)'
        ),
        'loss'
    )

    # =====================================================
    # 2. train 主损失分项
    # =====================================================

    panel(
        axes[0, 1],
        [
            (
                'train_main_loss_mal',
                T('mal(分类)', 'mal(cls)')
            ),
            ('train_main_loss_bbox', 'bbox'),
            ('train_main_loss_giou', 'giou')
        ],
        T(
            'train 主损失分项',
            'train main loss parts'
        ),
        'loss'
    )

    # =====================================================
    # 3. test 主损失
    # =====================================================

    panel(
        axes[0, 2],
        [
            ('test_main_loss', 'test/main'),
            ('test_main_loss_mal', 'mal'),
            ('test_main_loss_bbox', 'bbox'),
            ('test_main_loss_giou', 'giou')
        ],
        T(
            'test 主损失',
            'test main loss'
        ),
        'loss'
    )

    # =====================================================
    # 4. mAP
    # =====================================================

    panel(
        axes[1, 0],
        [
            ('test_map_50', 'mAP@0.5'),
            ('test_map_50_95', 'mAP@0.5:0.95'),
            ('test_map_75', 'AP@0.75')
        ],
        'test mAP',
        'AP'
    )

    axes[1, 0].set_ylim(bottom=0)

    # =====================================================
    # 5. YOLO 风格 P / R / F1
    # =====================================================

    ax = axes[1, 1]

    valid = (
        np.isfinite(precision)
        & np.isfinite(recall)
    )

    if valid.any():

        ax.plot(
            ep,
            precision,
            'o-',
            ms=3,
            lw=1.8,
            label='Precision'
        )

        ax.plot(
            ep,
            recall,
            'o-',
            ms=3,
            lw=1.8,
            label='Recall'
        )

        ax.plot(
            ep,
            f1,
            'o-',
            ms=3,
            lw=2.0,
            label='F1'
        )

        # 找 F1 最大值
        best_idx = np.nanargmax(
            np.where(valid, f1, np.nan)
        )

        best_epoch = ep[best_idx]
        best_f1 = f1[best_idx]
        best_p = precision[best_idx]
        best_r = recall[best_idx]

        # 在最佳点标记
        ax.scatter(
            [best_epoch],
            [best_f1],
            s=45,
            zorder=5
        )

        ax.annotate(
            f'best F1={best_f1:.4f}\n'
            f'Epoch={best_epoch}',
            xy=(best_epoch, best_f1),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            arrowprops=dict(
                arrowstyle='->',
                lw=0.8
            )
        )

        ax.set_title(
            T(
                'YOLO 风格 Precision / Recall / F1',
                'Precision / Recall / F1'
            )
        )

        ax.set_xlabel('epoch')
        ax.set_ylabel('value')
        ax.set_ylim(0, 1.02)

        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    else:

        ax.text(
            0.5,
            0.5,
            'no precision / recall data',
            ha='center',
            va='center',
            transform=ax.transAxes,
            color='gray'
        )

        ax.set_title(
            T(
                'YOLO 风格 Precision / Recall / F1',
                'Precision / Recall / F1'
            )
        )

    # =====================================================
    # 6. learning rate
    # =====================================================

    panel(
        axes[1, 2],
        [
            ('train_lr', 'lr')
        ],
        'learning rate',
        'lr'
    )

    axes[1, 2].set_yscale('log')

    # =====================================================
    # 总标题
    # =====================================================

    ap50 = np.array(
        get('test_map_50'),
        dtype=float
    )

    if not np.isnan(ap50).all():

        i = int(np.nanargmax(ap50))

        fig.suptitle(
            f'best mAP@0.5 = {ap50[i]:.4f} '
            f'@ epoch {ep[i]}',
            y=1.0
        )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=200,
        bbox_inches='tight'
    )

    print(f'已保存: {out}')
    print(f'解析到 {len(records)} 轮记录 (JSON 格式)')

    if not np.isnan(ap50).all():

        i = int(np.nanargmax(ap50))

        print(
            f'最佳 mAP@0.5 = {ap50[i]:.4f} '
            f'(epoch {ep[i]})'
        )

    # -----------------------------------------------------
    # 打印 YOLO 风格最佳 F1
    # -----------------------------------------------------

    if valid.any():

        best_idx = np.nanargmax(
            np.where(valid, f1, np.nan)
        )

        print(
            f'最佳 F1 = {f1[best_idx]:.4f} '
            f'(epoch {ep[best_idx]})'
        )

        print(
            f'对应 Precision = '
            f'{precision[best_idx]:.4f}'
        )

        print(
            f'对应 Recall = '
            f'{recall[best_idx]:.4f}'
        )


# =========================================================
# 格式 B：控制台文本
# =========================================================

def parse_console_text(path):

    epoch_losses = {}
    evals = []
    pending = {}

    pat_epoch = re.compile(
        r'Epoch:\s*\[(\d+)\]'
    )

    pat_loss = re.compile(
        r'\bloss:\s*([\d.]+)'
    )

    pat_val = re.compile(
        r'=\s*(-?\d+\.\d+|nan)\s*$'
    )

    with open(
        path,
        'r',
        encoding='utf-8',
        errors='ignore'
    ) as f:

        for line in f:

            m = pat_epoch.search(line)

            if m:

                lm = pat_loss.search(line)

                if lm:

                    epoch_losses.setdefault(
                        int(m.group(1)),
                        []
                    ).append(
                        float(lm.group(1))
                    )

            if (
                'Average Precision' in line
                or 'Average Recall' in line
            ):

                vm = pat_val.search(line)

                if vm is None:
                    continue

                val = float(vm.group(1))

                if (
                    'Precision' in line
                    and '0.50:0.95' in line
                ):

                    pending = {
                        'ap5095': val
                    }

                elif (
                    'Precision' in line
                    and re.search(
                        r'IoU=0\.50\s*\|',
                        line
                    )
                ):

                    pending['ap50'] = val

                elif (
                    'Precision' in line
                    and '0.75' in line
                ):

                    pending['ap75'] = val

                elif (
                    'Recall' in line
                    and 'ap5095' in pending
                ):

                    pending['ar'] = val

                    evals.append(
                        pending
                    )

                    pending = {}

    return epoch_losses, evals


# =========================================================
# 绘制控制台文本
# =========================================================

def plot_console_records(
    epoch_losses,
    evals,
    out
):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 4.8)
    )

    # -----------------------------------------------------
    # train loss
    # -----------------------------------------------------

    it_x, it_y = [], []

    for ep in sorted(epoch_losses):

        losses = epoch_losses[ep]

        xs = (
            ep
            + (np.arange(len(losses)) + 1)
            / (len(losses) + 1)
        )

        it_x.extend(xs)
        it_y.extend(losses)

    axes[0].plot(
        it_x,
        it_y,
        '.',
        alpha=0.25,
        ms=3,
        label='per-iter'
    )

    ep_x = sorted(epoch_losses)

    ep_y = [
        float(np.mean(epoch_losses[e]))
        for e in ep_x
    ]

    axes[0].plot(
        ep_x,
        ep_y,
        'o-',
        lw=2,
        ms=4,
        label='epoch mean'
    )

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('train/loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # -----------------------------------------------------
    # eval metrics
    # -----------------------------------------------------

    if evals:

        x = np.arange(len(evals))

        ap50 = np.array(
            [
                e.get('ap50', np.nan)
                for e in evals
            ],
            dtype=float
        )

        ap5095 = np.array(
            [
                e.get('ap5095', np.nan)
                for e in evals
            ],
            dtype=float
        )

        ap75 = np.array(
            [
                e.get('ap75', np.nan)
                for e in evals
            ],
            dtype=float
        )

        # -------------------------------------------------
        # 控制台格式通常没有 P / R
        # 因此这里只画 AP / AR
        # -------------------------------------------------

        axes[1].plot(
            x,
            ap50,
            'o-',
            ms=3,
            lw=1.8,
            label='mAP@0.5'
        )

        axes[1].plot(
            x,
            ap5095,
            'o-',
            ms=3,
            lw=1.8,
            label='mAP@0.5:0.95'
        )

        axes[1].plot(
            x,
            ap75,
            'o-',
            ms=3,
            lw=1.8,
            label='AP@0.75'
        )

        if not np.isnan(ap50).all():

            best = int(
                np.nanargmax(ap50)
            )

            axes[1].set_title(
                f'metrics '
                f'(best mAP@0.5 = '
                f'{ap50[best]:.4f} '
                f'@ epoch {best})'
            )

        axes[1].set_ylim(
            0,
            1.0
        )

        axes[1].legend()
        axes[1].grid(alpha=0.3)

    else:

        axes[1].text(
            0.5,
            0.5,
            'no eval blocks found in log',
            ha='center',
            va='center'
        )

    axes[1].set_xlabel('Epoch')

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=200
    )

    print(f'已保存: {out}')

    print(
        f'解析到 '
        f'{sum(len(v) for v in epoch_losses.values())} '
        f'条迭代记录, '
        f'{len(evals)} 次评估 (文本格式)'
    )


# =========================================================
# 入口：自动识别格式
# =========================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--log',
        required=True,
        help='训练日志路径'
    )

    ap.add_argument(
        '--out',
        default='training_curves.png',
        help='输出图片路径'
    )

    args = ap.parse_args()

    records = parse_json_lines(
        args.log
    )

    if records:

        plot_json_records(
            records,
            args.out
        )

        return

    epoch_losses, evals = parse_console_text(
        args.log
    )

    assert (
        epoch_losses
        or evals
    ), (
        '日志里既没有每轮 JSON 记录, '
        '也没有 Epoch 文本行, '
        '请确认传入的是完整训练日志'
    )

    plot_console_records(
        epoch_losses,
        evals,
        args.out
    )


if __name__ == '__main__':
    main()