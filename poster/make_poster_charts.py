#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poster-facing charts — unified colorful palette, English labels
==============================================================
Regenerates the figures used on the poster (from the Day 07/08 measurements) in the
poster palette so the whole sheet reads as one design system.

Outputs (poster/figs/):
  fig_nms_cost.png        NMS cost model (conf / IoU sweeps, our measurement)
  fig_attention.png       attention cost ~ M^2 scaling (our measurement)
  fig_encoder.png         hybrid-encoder variants: latency order vs paper
  fig_uncertainty.png     uncertainty degenerate optimum (our measurement)
  fig_l1_giou.png         L1 scales with size, GIoU does not (our measurement)
  fig_ap_vs_fps.png       accuracy vs real-time speed landscape (paper numbers)

Data provenance (kept honest):
  * NMS / attention / encoder / uncertainty / L1-GIoU  -> our own CPU measurement
  * AP vs FPS scatter                                    -> paper Table 2 numbers
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

NAVY = "#12324f"
TEAL = "#1f7a8c"
AMBER = "#e0a526"
CORAL = "#d1495b"
VIOLET = "#6b4e9b"
MINT = "#3f9d6d"
SKY = "#3d8fc1"
INK = "#16202a"
MUTED = "#5d6d7c"
GRID = "#e3e9ee"


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12.5, color=INK, weight="bold", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def fig_nms_cost(m):
    conf = m["A_nms_cost"]["conf_sweep"]
    iou = m["A_nms_cost"]["iou_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot([r["conf"] for r in conf], [r["ms"] for r in conf], marker="o", color=CORAL,
            linewidth=2.4, label="NMS time (ours)")
    ax2 = ax.twinx()
    ax2.plot([r["conf"] for r in conf], [r["ap_proxy"] for r in conf], marker="s", color=AMBER,
             linestyle="--", linewidth=2, label="AP proxy (ours)")
    ax.set_xscale("log")
    ax.set_ylim(0, max(r["ms"] for r in conf) * 1.25)
    ax2.set_ylim(0.9, 1.02)
    style(ax, "Confidence threshold up:  faster, less accurate", "confidence threshold", "NMS time (ms)")
    ax2.set_ylabel("AP proxy", fontsize=10, color=AMBER)
    ax2.tick_params(colors=AMBER, labelsize=9)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    ax.annotate("conf 0.001 -> 0.25\ntime 57.0 -> 21.3 ms",
                xy=(0.25, conf[-1]["ms"]), xytext=(0.012, conf[-1]["ms"] * 0.72),
                fontsize=9, color=CORAL, arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2))

    ax = axes[1]
    ax.plot([r["iou"] for r in iou], [r["ms"] for r in iou], marker="o", color=TEAL,
            linewidth=2.4, label="NMS time (ours)")
    ax2 = ax.twinx()
    ax2.plot([r["iou"] for r in iou], [r["ap_proxy"] for r in iou], marker="s", color=AMBER,
             linestyle="--", linewidth=2, label="AP proxy (ours)")
    ax.set_ylim(0, max(r["ms"] for r in iou) * 1.25)
    ax2.set_ylim(0.9, 1.02)
    style(ax, "IoU threshold up:  more accurate, slower", "NMS IoU threshold", "NMS time (ms)")
    ax2.set_ylabel("AP proxy", fontsize=10, color=AMBER)
    ax2.tick_params(colors=AMBER, labelsize=9)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_nms_cost.png")
    fig.savefig(p)
    plt.close(fig)
    print("  saved", os.path.basename(p))


def fig_attention(m):
    rows = m["B_attention_scaling"]["scaling"]
    M = np.array([r["tokens"] for r in rows], dtype=float)
    T = np.array([r["ms"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.loglog(M, T, marker="o", markersize=7, color=VIOLET, linewidth=2.4, label="measured (CPU)")
    ax.loglog(M, T[0] * (M / M[0]) ** 2, linestyle="--", color=MUTED, linewidth=1.6,
              label=r"theory $\propto M^2$")
    style(ax, "Attention cost scales with token count", "tokens M", "attention time (ms)")
    ax.legend(fontsize=9, frameon=False)
    ax.annotate("8x tokens -> 42.3x time\n(theory 64x)", xy=(M[-1], T[-1]), xytext=(0.08, 0.72),
                textcoords="axes fraction", fontsize=9, color=VIOLET,
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.2))
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_attention.png")
    fig.savefig(p)
    plt.close(fig)
    print("  saved", os.path.basename(p))


def fig_encoder(m):
    rows = m["C_encoder_variants"]["rows"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(rows))
    ours = [r["ms"] for r in rows]
    paper = [r["paper_ms"] for r in rows]
    b1 = ax.bar(x - 0.2, ours, 0.38, color=TEAL, label="ours (mini impl.)")
    b2 = ax.bar(x + 0.2, paper, 0.38, color=AMBER, label="paper Table 3 (T4)")
    for r, rect in zip(rows, b1):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3, f"{r['paper_ap']}",
                ha="center", fontsize=8.4, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([r["variant"] for r in rows])
    style(ax, "Hybrid encoder variants: same latency order", "variant (paper AP above ours)", "latency (ms)")
    ax.legend(fontsize=8.6, frameon=False)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_encoder.png")
    fig.savefig(p)
    plt.close(fig)
    print("  saved", os.path.basename(p))


def fig_uncertainty(m):
    d = m["D_uncertainty_degeneracy"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    t = [r["step"] for r in d["traj"]]
    ax.plot(t, [r["s"] for r in d["traj"]], color=CORAL, linewidth=2.4, label="U only: class score s")
    ax.plot(t, [r["q"] for r in d["traj"]], color=CORAL, linestyle="--", linewidth=2,
            label="U only: location quality q")
    t2 = [r["step"] for r in d["traj_with_cls"]]
    ax.plot(t2, [r["s"] for r in d["traj_with_cls"]], color=MINT, linewidth=2.4,
            label="U + class loss: s")
    ax.plot(t2, [r["q"] for r in d["traj_with_cls"]], color=MINT, linestyle="--", linewidth=2,
            label="U + class loss: q")
    style(ax, "Uncertainty alone has a degenerate optimum", "optimization step", "score / quality")
    ax.legend(fontsize=8.2, frameon=False, ncol=2)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_uncertainty.png")
    fig.savefig(p)
    plt.close(fig)
    print("  saved", os.path.basename(p))


def fig_l1_giou():
    import torch
    import sys
    sys.path.insert(0, os.path.join(ROOT, "day07_detr"))
    from detr_demo import giou_loss  # type: ignore

    small = torch.tensor([[0.24, 0.21, 0.10, 0.08]])
    large = torch.tensor([[0.50, 0.55, 0.90, 0.80]])
    errs = np.linspace(0, 0.40, 41)
    l1s, l1l, gs, gl = [], [], [], []
    for e in errs:
        ps, pl = small.clone(), large.clone()
        ps[0, 2:] *= 1 + e
        pl[0, 2:] *= 1 + e
        l1s.append(float(torch.abs(small - ps).sum()))
        l1l.append(float(torch.abs(large - pl).sum()))
        gs.append(float(giou_loss(small, ps)))
        gl.append(float(giou_loss(large, pl)))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(errs * 100, l1l, color=SKY, linewidth=2.4, label="L1, large box (0.90x0.80)")
    ax.plot(errs * 100, l1s, color=SKY, linestyle="--", linewidth=2, label="L1, small box (0.10x0.08)")
    style(ax, "L1 penalizes big boxes harder", "relative error (%)", "L1 loss")
    ax.legend(fontsize=8.6, frameon=False)
    ax.annotate("9.44x at 15% error", xy=(15, l1l[15]), xytext=(17, l1l[15] * 0.6),
                fontsize=9, color=CORAL, arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2))
    ax = axes[1]
    ax.plot(errs * 100, gl, color=MINT, linewidth=2.4, label="GIoU, large box")
    ax.plot(errs * 100, gs, color=MINT, linestyle="--", linewidth=2, label="GIoU, small box")
    style(ax, "GIoU is scale-invariant (ratio 1.000x)", "relative error (%)", "GIoU loss")
    ax.legend(fontsize=8.6, frameon=False)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_l1_giou.png")
    fig.savefig(p)
    plt.close(fig)
    print("  saved", os.path.basename(p))


def fig_ap_vs_fps():
    """Accuracy vs real-time speed landscape (paper Table 2 numbers, clearly labeled)."""
    pts = [
        ("DINO-Deformable\n(R50)", 50.9, 5, VIOLET),
        ("Faster R-CNN-FPN+\n(R50)", 42.0, 26, MUTED),
        ("YOLOv8-L", 52.9, 71, SKY),
        ("RT-DETR (R50)", 53.1, 108, CORAL),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    for name, ap, fps, c in pts:
        ax.scatter(fps, ap, s=210, color=c, zorder=3, edgecolors="white", linewidths=1.6)
        ax.annotate(name, (fps, ap), textcoords="offset points", xytext=(8, 6),
                    fontsize=8.6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(3, 200)
    ax.set_ylim(40, 55)
    style(ax, "Accuracy vs real-time speed (paper Table 2)", "FPS (T4, TensorRT FP16, bs=1)", "COCO AP")
    ax.annotate("RT-DETR: best accuracy\nat 108 FPS", xy=(108, 53.1), xytext=(22, 47.4),
                fontsize=9, color=CORAL, weight="bold",
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.3))
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_ap_vs_fps.png")
    fig.savefig(p)
    plt.close(fig)
    print("  saved", os.path.basename(p))


def main() -> None:
    import json
    m = json.load(open(os.path.join(ROOT, "day08_rtdetr/runs/rtdetr_demo_metrics.json")))
    print("[poster charts] rendering")
    fig_nms_cost(m)
    fig_attention(m)
    fig_encoder(m)
    fig_uncertainty(m)
    fig_l1_giou()
    fig_ap_vs_fps()
    print("[poster charts] done ->", os.path.relpath(FIGS, ROOT))


if __name__ == "__main__":
    main()
