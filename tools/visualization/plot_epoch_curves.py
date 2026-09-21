
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 配置
# ============================================================
LOG_PATH = Path(
    r"./outputs/你的实验目录/log.txt"
)

OUTPUT_DIR = LOG_PATH.parent / "paper_figures"

# EMA 平滑跨度
# 5：轻微平滑
# 7：比较常用
# 9：更加平滑
SMOOTH_SPAN = 7


# ============================================================
# 2. 论文风格
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,

    "axes.linewidth": 0.8,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "figure.dpi": 150,
    "savefig.dpi": 600,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 3. 读取日志
# ============================================================
def read_log(log_path: Path) -> pd.DataFrame:
    records = []

    with log_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"跳过第 {line_number} 行："
                    "不是合法 JSON"
                )
                continue

            epoch = row.get("epoch")

            if epoch is None:
                continue

            # 兼容旧日志：
            # 旧日志可能只有 test_coco_eval_bbox
            bbox_stats = row.get(
                "test_coco_eval_bbox"
            )

            if not isinstance(
                bbox_stats,
                (list, tuple)
            ):
                bbox_stats = None

            map_50_95 = row.get(
                "test_map_50_95"
            )

            map_50 = row.get(
                "test_map_50"
            )

            map_75 = row.get(
                "test_map_75"
            )

            if bbox_stats is not None:
                if (
                    map_50_95 is None
                    and len(bbox_stats) > 0
                ):
                    map_50_95 = bbox_stats[0]

                if (
                    map_50 is None
                    and len(bbox_stats) > 1
                ):
                    map_50 = bbox_stats[1]

                if (
                    map_75 is None
                    and len(bbox_stats) > 2
                ):
                    map_75 = bbox_stats[2]

            records.append({
                "epoch": int(epoch),

                "train_loss": row.get(
                    "train_loss",
                    np.nan
                ),

                "val_loss": row.get(
                    "test_loss",
                    np.nan
                ),

                "map_50_95": (
                    map_50_95
                    if map_50_95 is not None
                    else np.nan
                ),

                "map_50": (
                    map_50
                    if map_50 is not None
                    else np.nan
                ),

                "map_75": (
                    map_75
                    if map_75 is not None
                    else np.nan
                ),

                "f1_50_max": row.get(
                    "test_f1_50_max",
                    np.nan
                ),

                "precision_50": row.get(
                    "test_precision_50_at_f1",
                    np.nan
                ),

                "recall_50": row.get(
                    "test_recall_50_at_f1",
                    np.nan
                ),
            })

    if not records:
        raise RuntimeError(
            "日志中没有找到有效 epoch 数据。"
        )

    data = pd.DataFrame(records)

    data = (
        data
        .drop_duplicates(
            subset="epoch",
            keep="last"
        )
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    return data


# ============================================================
# 4. 平滑函数
# ============================================================
def smooth_series(
    values: pd.Series,
    span: int
) -> pd.Series:
    return values.ewm(
        span=span,
        adjust=False,
        min_periods=1
    ).mean()


def plot_raw_and_smooth(
    ax,
    epochs,
    values,
    label,
    color,
    span=7,
):
    values = pd.Series(
        values,
        dtype="float64"
    )

    valid = values.notna()

    if valid.sum() == 0:
        return

    valid_epochs = np.asarray(epochs)[valid]
    valid_values = values[valid]

    smooth_values = smooth_series(
        valid_values,
        span=span
    )

    # 原始曲线：细且透明
    ax.plot(
        valid_epochs,
        valid_values,
        color=color,
        linewidth=0.7,
        alpha=0.20
    )

    # 平滑曲线：论文中主要展示的曲线
    ax.plot(
        valid_epochs,
        smooth_values,
        color=color,
        linewidth=1.8,
        label=label
    )


def format_axis(
    ax,
    xlabel="Epoch",
    ylabel=None,
    title=None,
):
    if title is not None:
        ax.set_title(
            title,
            pad=7
        )

    ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.minorticks_on()

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.5,
        alpha=0.20
    )

    ax.tick_params(
        which="major",
        length=4,
        width=0.8
    )

    ax.tick_params(
        which="minor",
        length=2,
        width=0.6
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ============================================================
# 5. 绘图
# ============================================================
def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data = read_log(LOG_PATH)

    # 保存整理后的指标
    data.to_csv(
        OUTPUT_DIR / "epoch_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.4),
        constrained_layout=True
    )

    epochs = data["epoch"].to_numpy()

    # --------------------------------------------------------
    # (a) mAP50:95
    # --------------------------------------------------------
    plot_raw_and_smooth(
        axes[0, 0],
        epochs,
        data["map_50_95"],
        label=r"$mAP_{50:95}$",
        color="#1f77b4",
        span=SMOOTH_SPAN,
    )

    format_axis(
        axes[0, 0],
        ylabel=r"$mAP_{50:95}$",
        title="(a) Detection accuracy"
    )

    axes[0, 0].legend(
        frameon=False,
        loc="lower right"
    )

    # --------------------------------------------------------
    # (b) mAP50 与 mAP75
    # --------------------------------------------------------
    plot_raw_and_smooth(
        axes[0, 1],
        epochs,
        data["map_50"],
        label=r"$mAP_{50}$",
        color="#d62728",
        span=SMOOTH_SPAN,
    )

    plot_raw_and_smooth(
        axes[0, 1],
        epochs,
        data["map_75"],
        label=r"$mAP_{75}$",
        color="#2ca02c",
        span=SMOOTH_SPAN,
    )

    format_axis(
        axes[0, 1],
        ylabel="Average Precision",
        title="(b) AP at different IoU"
    )

    axes[0, 1].legend(
        frameon=False,
        loc="lower right"
    )

    # --------------------------------------------------------
    # (c) F1、Precision、Recall
    # --------------------------------------------------------
    plot_raw_and_smooth(
        axes[1, 0],
        epochs,
        data["f1_50_max"],
        label=r"Max $F_1$@0.5",
        color="#9467bd",
        span=SMOOTH_SPAN,
    )

    plot_raw_and_smooth(
        axes[1, 0],
        epochs,
        data["precision_50"],
        label="Precision",
        color="#ff7f0e",
        span=SMOOTH_SPAN,
    )

    plot_raw_and_smooth(
        axes[1, 0],
        epochs,
        data["recall_50"],
        label="Recall",
        color="#17becf",
        span=SMOOTH_SPAN,
    )

    format_axis(
        axes[1, 0],
        ylabel="Metric value",
        title="(c) Precision, recall and F1"
    )

    axes[1, 0].legend(
        frameon=False,
        loc="lower right"
    )

    # --------------------------------------------------------
    # (d) 训练损失与验证损失
    # --------------------------------------------------------
    plot_raw_and_smooth(
        axes[1, 1],
        epochs,
        data["train_loss"],
        label="Training loss",
        color="#1f77b4",
        span=SMOOTH_SPAN,
    )

    plot_raw_and_smooth(
        axes[1, 1],
        epochs,
        data["val_loss"],
        label="Validation loss",
        color="#d62728",
        span=SMOOTH_SPAN,
    )

    format_axis(
        axes[1, 1],
        ylabel="Loss",
        title="(d) Training and validation loss"
    )

    axes[1, 1].legend(
        frameon=False,
        loc="upper right"
    )

    # 保存位图
    png_path = (
        OUTPUT_DIR
        / "training_curves_paper_style.png"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    # 保存矢量图
    pdf_path = (
        OUTPUT_DIR
        / "training_curves_paper_style.pdf"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("绘图完成：")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()