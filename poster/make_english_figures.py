#!/usr/bin/env python3
"""Build English-labeled poster charts from the Day 07/08 experiment metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "poster" / "runs" / "figures"
DETR_RUNS = ROOT / "day07_detr" / "runs"
RTDETR_RUNS = ROOT / "day08_rtdetr" / "runs"

INK = "#1f2933"
GRID = "#d7dde3"
BLUE = "#2f6f9f"
CORAL = "#c1553b"
GREEN = "#4a7c59"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
    }
)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11.5, color=INK, pad=9)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=180)
    plt.close(fig)
    print(f"saved {name}")


def load_metrics(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def box_giou_loss(box_a, box_b):
    def corners(box):
        cx, cy, width, height = box
        return np.array(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
        )

    a, b = corners(box_a), corners(box_b)
    intersection_wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = intersection_wh[0] * intersection_wh[1]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    iou = intersection / max(union, 1e-12)
    enclosing_wh = np.maximum(a[2:], b[2:]) - np.minimum(a[:2], b[:2])
    enclosing = enclosing_wh[0] * enclosing_wh[1]
    giou = iou - (enclosing - union) / max(enclosing, 1e-12)
    return 1 - giou


def chart_l1_giou():
    small = np.array([0.24, 0.21, 0.10, 0.08])
    large = np.array([0.50, 0.55, 0.90, 0.80])
    errors = np.linspace(0, 0.40, 41)
    values = {"small_l1": [], "large_l1": [], "small_giou": [], "large_giou": []}
    for error in errors:
        pred_small, pred_large = small.copy(), large.copy()
        pred_small[2:] *= 1 + error
        pred_large[2:] *= 1 + error
        values["small_l1"].append(np.abs(small - pred_small).sum())
        values["large_l1"].append(np.abs(large - pred_large).sum())
        values["small_giou"].append(box_giou_loss(small, pred_small))
        values["large_giou"].append(box_giou_loss(large, pred_large))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(errors * 100, values["large_l1"], color=BLUE, linewidth=2,
            label="L1 · large (h,w)=(0.90,0.80)")
    ax.plot(errors * 100, values["small_l1"], color=BLUE, linewidth=2,
            linestyle="--", label="L1 · small (h,w)=(0.10,0.08)")
    style(ax, "L1 scales with box size at equal relative error",
          "Relative error (%)", "L1 loss")
    ax.legend(fontsize=7.5, frameon=False)
    ratio = values["large_l1"][15] / values["small_l1"][15]
    ax.annotate(f"{ratio:.2f}× at 15%", xy=(15, values["large_l1"][15]),
                xytext=(20, values["large_l1"][15] * 0.75), fontsize=9, color=CORAL,
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2))

    ax = axes[1]
    ax.plot(errors * 100, values["large_giou"], color=GREEN, linewidth=2,
            label="GIoU · large")
    ax.plot(errors * 100, values["small_giou"], color=GREEN, linewidth=2,
            linestyle="--", label="GIoU · small")
    style(ax, "GIoU is scale-invariant (the curves overlap)",
          "Relative error (%)", "GIoU loss")
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.52, 0.86, "Large / small = 1.000×\nat equal relative error",
            transform=ax.transAxes, fontsize=9, color=GREEN)
    save(fig, "l1-giou-scale-en.png")


def chart_nms(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rows, x_key, title, xlabel in (
        (axes[0], metrics["A_nms_cost"]["conf_sweep"], "conf",
         "Higher confidence threshold → faster NMS, lower AP proxy",
         "Confidence threshold"),
        (axes[1], metrics["A_nms_cost"]["iou_sweep"], "iou",
         "Higher IoU threshold → slower NMS, higher AP proxy",
         "NMS IoU threshold"),
    ):
        x = [row[x_key] for row in rows]
        ax.plot(x, [row["ms"] for row in rows], marker="o", color=BLUE,
                linewidth=2, label="NMS time")
        if x_key == "conf":
            ax.set_xscale("log")
        style(ax, title, xlabel, "NMS time (ms)")
        ax2 = ax.twinx()
        ax2.plot(x, [row["ap_proxy"] for row in rows], marker="s", color=CORAL,
                 linestyle="--", linewidth=1.8, label="AP proxy")
        ax2.set_ylabel("AP proxy (experiment)", fontsize=9.5, color=CORAL)
        ax2.set_ylim(0.85, 1.02)
        ax2.spines["top"].set_visible(False)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, frameon=False,
                  loc="center right")
    fig.subplots_adjust(wspace=0.48)
    save(fig, "nms-tradeoff-en.png")


def chart_detr_ap():
    models = ["Faster R-CNN-FPN+", "DETR", "DETR-DC5", "DETR-R101", "DETR-DC5-R101"]
    ap = [42.0, 42.0, 43.3, 43.5, 44.9]
    aps = [26.6, 20.5, 22.5, 21.9, 23.7]
    apl = [53.4, 61.1, 61.1, 61.8, 62.3]
    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - width, ap, width, color=BLUE, label="AP")
    ax.bar(x, aps, width, color=CORAL, label="AP$_S$ (small)")
    ax.bar(x + width, apl, width, color=GREEN, label="AP$_L$ (large)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    style(ax, "DETR vs Faster R-CNN: AP breakdown (DETR paper, Table 1)",
          "Model (R50 / R101)", "AP")
    ax.legend(fontsize=8.5, frameon=False)
    ax.annotate("DETR: AP$_S$ −5.5 · AP$_L$ +7.8\n(41M vs 42M parameters)",
                xy=(1, 61.1), xytext=(0.02, 0.94), textcoords="axes fraction",
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    save(fig, "detr-ap-scale-en.png")


def chart_encoder(metrics):
    rows = metrics["C_encoder_variants"]["rows"]
    labels = [row["variant"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, [row["ms"] for row in rows], 0.4, color=BLUE,
                label="Mini implementation (CPU)")
    axes[0].bar(x + 0.2, [row["paper_ms"] for row in rows], 0.4, color=CORAL,
                label="Paper Table 3 (T4)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    style(axes[0], "Encoder variant latency (compare order, not absolute values)",
          "Variant", "Latency (ms)")
    axes[0].legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    paper_latency = [row["paper_ms"] for row in rows]
    paper_ap = [row["paper_ap"] for row in rows]
    colors = [BLUE, BLUE, BLUE, GREEN, CORAL, CORAL]
    ax.scatter(paper_latency, paper_ap, s=110, c=colors, zorder=3,
               edgecolors="white", linewidths=1.2)
    for row in rows:
        ax.annotate(row["variant"], (row["paper_ms"], row["paper_ap"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9, color=INK)
    style(ax, "Paper Table 3: accuracy vs latency",
          "Latency (ms)", "AP")
    ax.annotate("D → D$_{S5}$\n−35% latency · +0.4 AP", xy=(7.9, 46.8),
                xytext=(9.2, 44.2), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    save(fig, "encoder-variants-en.png")


def chart_attention(metrics):
    data = metrics["B_attention_scaling"]
    rows = data["scaling"]
    tokens = np.array([row["tokens"] for row in rows], dtype=float)
    times = np.array([row["ms"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.loglog(tokens, times, marker="o", color=BLUE, linewidth=2,
              label="Measured (CPU, d=256, heads=8)")
    ax.loglog(tokens, times[0] * (tokens / tokens[0]) ** 2, linestyle="--",
              color=CORAL, linewidth=1.6, label="Quadratic reference (∝ M²)")
    style(ax, "Attention cost vs token count — why AIFI uses S5 (400 tokens)",
          "Token count M", "Attention time (ms)")
    ax.legend(fontsize=8, frameon=False)
    ax.annotate(
        f"400 → 3,200 tokens (8×) = {data['ratio_8x']:.1f}× time\n"
        "(theory: 64×; measured scaling is lower)",
        xy=(tokens[-1], times[-1]), xytext=(0.06, 0.62), textcoords="axes fraction",
        fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1),
    )
    ax.text(
        0.06, 0.12,
        f"RT-DETR v1: 8,400 candidates vs S5 400 = "
        f"{data['token_ratio_8400_over_400']:.0f}× tokens\n"
        f"(1×1 conv at 80×80×256 = {data['conv_ms_80x80']:.2f} ms)",
        transform=ax.transAxes, fontsize=8.5, color=GREEN,
    )
    save(fig, "attention-scaling-en.png")


def chart_uncertainty(metrics):
    data = metrics["D_uncertainty_degeneracy"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    steps = [row["step"] for row in data["traj"]]
    ax.plot(steps, [row["s"] for row in data["traj"]], color=CORAL, linewidth=2,
            label="U-only: classification score s")
    ax.plot(steps, [row["q"] for row in data["traj"]], color=CORAL, linestyle="--",
            linewidth=1.8, label="U-only: localization quality q")
    steps_with_cls = [row["step"] for row in data["traj_with_cls"]]
    ax.plot(steps_with_cls, [row["s"] for row in data["traj_with_cls"]],
            color=BLUE, linewidth=2, label="With classification loss: s")
    ax.plot(steps_with_cls, [row["q"] for row in data["traj_with_cls"]],
            color=BLUE, linestyle="--", linewidth=1.8,
            label="With classification loss: q")
    style(ax, "Degenerate uncertainty objective: U alone vs classification supervision",
          "Optimization step", "Score / quality (0–1)")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    save(fig, "uncertainty-objective-en.png")


def chart_decoder_depth(metrics):
    results = metrics["results"]
    depths = sorted(int(depth) for depth in results["vanilla"]["depth_curve"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, color, label in (
        ("vanilla", BLUE, "Vanilla"),
        ("uncertainty", CORAL, "Uncertainty-minimal"),
    ):
        curve = results[key]["depth_curve"]
        ax.plot(depths, [curve[str(depth)] for depth in depths], marker="o",
                color=color, linewidth=2, label=label)
    style(ax, "Validation AP as decoder depth is reduced (no retraining)",
          "Decoder layers used", "AP@0.5 (validation)")
    ax.invert_xaxis()
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.03, 0.06,
            "In this mini-demo, shallower settings sometimes score higher.\n"
            "The paper's RT-DETR Table 5 favors six layers.",
            transform=ax.transAxes, fontsize=8.2, color=INK)
    save(fig, "decoder-depth-en.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rtdetr = load_metrics(RTDETR_RUNS / "rtdetr_demo_metrics.json")
    detr_pipeline = load_metrics(DETR_RUNS / "detr_pipeline_metrics.json")
    chart_l1_giou()
    chart_nms(rtdetr)
    chart_detr_ap()
    chart_encoder(rtdetr)
    chart_attention(rtdetr)
    chart_uncertainty(rtdetr)
    chart_decoder_depth(detr_pipeline)


if __name__ == "__main__":
    main()
