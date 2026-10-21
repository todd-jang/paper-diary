#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Poster Builder — DETR -> RT-DETR (colorful, all-English)
============================================================
Renders a full-color A0/A1 poster (and 16:9 slides) from the Day 07/08 artifacts.

Design: 3-column layout, tinted panels, colored metric cards, a rendered results
table, a custom pipeline flow diagram, and a framed chart gallery.

Usage:
  .venv/bin/python poster/build_poster.py               # A1 + PNG preview
  .venv/bin/python poster/build_poster.py --size A0     # A0 (print)
  .venv/bin/python poster/build_poster.py --slides      # 4 slides (16:9)
"""
import argparse
import os
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "runs")
os.makedirs(OUT, exist_ok=True)

NAVY = "#12324f"
TEAL = "#1f7a8c"
AMBER = "#e0a526"
CORAL = "#d1495b"
VIOLET = "#6b4e9b"
MINT = "#3f9d6d"
SKY = "#3d8fc1"
BG = "#fbfcfd"
LINE = "#dfe6ec"
INK = "#16202a"
MUTED = "#5d6d7c"
WHITE = "#ffffff"

TINT = {
    "problem": "#fdf1ee",
    "method": "#eef6f8",
    "result": "#eef4fb",
    "exp": "#f0f7f1",
    "limit": "#fdf6ea",
    "flow": "#f5f0fa",
}

SIZES = {"A0": (33.1, 46.8), "A1": (23.4, 33.1), "A2": (16.5, 23.4)}


def load(rel: str):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"  [skip] {rel}")
        return None
    return plt.imread(path)


def panel(ax, x, y, w, h, title: str, accent: str, tint: str, num: str = "") -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.005",
                                linewidth=0.8, edgecolor=LINE, facecolor=tint, transform=ax.transAxes))
    ax.add_patch(Rectangle((x, y + h - 0.006), w, 0.006, facecolor=accent, transform=ax.transAxes))
    tx = x + 0.016
    if num:
        ax.add_patch(FancyBboxPatch((tx, y + h - 0.036), 0.026, 0.022,
                                    boxstyle="round,pad=0.001,rounding_size=0.008",
                                    facecolor=accent, transform=ax.transAxes))
        ax.text(tx + 0.013, y + h - 0.025, num, transform=ax.transAxes, fontsize=11.5,
                color=WHITE, ha="center", va="center", weight="bold")
        tx += 0.036
    ax.text(tx, y + h - 0.025, title, transform=ax.transAxes, fontsize=15.5,
            color=accent, weight="bold", va="center")


def rich(ax, x, y, rows: Sequence[Tuple[str, str]], size: float = 11.0, step: float = 0.0155) -> None:
    for i, (c, t) in enumerate(rows):
        yy = y - i * (size * step)
        ax.add_patch(Rectangle((x, yy - 0.0035), 0.007, 0.007, facecolor=c, transform=ax.transAxes))
        ax.text(x + 0.013, yy, t, transform=ax.transAxes, fontsize=size, color=INK, va="top")


def card(ax, x, y, w, h, big: str, label: str, color: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.01",
                                linewidth=0, facecolor=color, transform=ax.transAxes))
    ax.text(x + w / 2, y + h * 0.62, big, transform=ax.transAxes, fontsize=25, color=WHITE,
            ha="center", va="center", weight="bold")
    ax.text(x + w / 2, y + h * 0.26, label, transform=ax.transAxes, fontsize=9.4, color=WHITE,
            ha="center", va="center")


def chip(ax, x, y, w, h, text: str, color: str, fontsize: float = 9.2) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.001,rounding_size=0.012",
                                linewidth=1.1, edgecolor=color, facecolor=WHITE,
                                transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, fontsize=fontsize, color=color,
            ha="center", va="center", weight="bold")


def framed_image(ax, img, extent, color: str, placed: list) -> None:
    x0, x1, y0, y1 = extent
    ax.add_patch(Rectangle((x0 - 0.004, y0 - 0.004), (x1 - x0) + 0.008, (y1 - y0) + 0.008,
                           facecolor=WHITE, edgecolor=color, linewidth=1.4, transform=ax.transAxes))
    ax.imshow(img, extent=extent, transform=ax.transAxes, aspect="auto")
    placed.append(tuple(extent))


def results_table(ax, x, y, w, h, placed: list) -> None:
    rows = [
        ["Model", "AP", "AP_S", "AP_L", "Params", "FPS"],
        ["DETR (R50)", "42.0", "20.5", "61.1", "41M", "28"],
        ["Faster R-CNN-FPN+", "42.0", "26.6", "53.4", "42M", "26"],
        ["RT-DETR (R50)", "53.1", "34.8", "70.0", "42M", "108"],
    ]
    highlights = {(3, 2): CORAL, (3, 3): AMBER, (3, 1): AMBER, (3, 5): AMBER}
    widths = [0.36, 0.13, 0.13, 0.13, 0.13, 0.12]
    rh = 0.028
    for ri, row in enumerate(rows):
        ry = y + h - (ri + 1) * rh
        cx = x
        if ri == 0:
            ax.add_patch(Rectangle((x, ry), w, rh, facecolor=NAVY, transform=ax.transAxes))
        else:
            bg = "#fdf3e3" if ri == 3 else ("#eaf3fa" if ri % 2 else WHITE)
            ax.add_patch(Rectangle((x, ry), w, rh, facecolor=bg, transform=ax.transAxes))
        for ci, cell in enumerate(row):
            cw = w * widths[ci]
            color = WHITE if ri == 0 else INK
            if (ri, ci) in highlights:
                color = highlights[(ri, ci)]
            weight = "bold" if (ri == 0 or ri == 3) else "normal"
            ax.text(cx + cw / 2, ry + rh * (0.60 if ri else 0.5), cell, transform=ax.transAxes,
                    fontsize=10.0, color=color, ha="center", va="center", weight=weight)
            if ri and ci in (2, 3):
                v = float(cell)
                bar_c = CORAL if ci == 2 else AMBER
                bw = cw * 0.62 * (v / 70.0)
                ax.add_patch(Rectangle((cx + cw * 0.19, ry + rh * 0.14), bw, rh * 0.16,
                                       facecolor=bar_c, transform=ax.transAxes))
            cx += cw
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=LINE, linewidth=1.0,
                           transform=ax.transAxes))
    placed.append((x, x + w, y, y + h))
    ax.text(x, y - 0.012,
            "DETR: same params as Faster R-CNN-FPN+, but AP_S -5.5 / AP_L +7.8   |   "
            "RT-DETR: +2.2 AP and 21x faster than DINO-Deformable (5 -> 108 FPS)",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, va="top")


def flow_diagram(ax, x, y, w, h) -> None:
    steps = [
        ("Backbone", "multi-scale\nfeatures", SKY),
        ("Hybrid Encoder", "AIFI (S5 only)\n+ CCFF (conv)", TEAL),
        ("Query Selection", "top-K by cls score\n+ uncertainty loss", AMBER),
        ("Decoder (+aux)", "6 layers, NMS-free\ndepth tunable", CORAL),
    ]
    gap = 0.024
    bw = (w - gap * (len(steps) - 1)) / len(steps)
    bh = h * 0.62
    for i, (title, sub, color) in enumerate(steps):
        bx = x + i * (bw + gap)
        by = y + h - bh
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                    boxstyle="round,pad=0.002,rounding_size=0.012",
                                    facecolor=color, edgecolor="none", transform=ax.transAxes))
        ax.text(bx + bw / 2, by + bh * 0.66, title, transform=ax.transAxes, fontsize=10.2,
                color=WHITE, ha="center", va="center", weight="bold")
        ax.text(bx + bw / 2, by + bh * 0.28, sub, transform=ax.transAxes, fontsize=7.9,
                color=WHITE, ha="center", va="center", alpha=0.94)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((bx + bw + 0.002, by + bh / 2),
                                         (bx + bw + gap - 0.002, by + bh / 2),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color=MUTED, linewidth=1.1, transform=ax.transAxes))
    ax.text(x, y + h - bh - 0.016,
            "One-to-one matching kills duplicates at training time -> no NMS at inference",
            transform=ax.transAxes, fontsize=8.6, color=MUTED, va="top")
    ax.text(x, y + h - bh - 0.032,
            "8400 vs 400 tokens = 441x attention gap -> keep attention only on S5",
            transform=ax.transAxes, fontsize=8.6, color=CORAL, va="top", weight="bold")


def timeline(ax, x, y, w, days: Sequence[Tuple[str, str, str]]) -> None:
    gap = 0.007
    cw = (w - gap * (len(days) - 1)) / len(days)
    for i, (d, label, color) in enumerate(days):
        cx = x + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((cx, y), cw, 0.050,
                                    boxstyle="round,pad=0.001,rounding_size=0.014",
                                    facecolor=color, transform=ax.transAxes))
        ax.text(cx + cw / 2, y + 0.035, d, transform=ax.transAxes, fontsize=8.2, color=WHITE,
                ha="center", va="center", weight="bold")
        ax.text(cx + cw / 2, y + 0.016, label, transform=ax.transAxes, fontsize=6.6, color=WHITE,
                ha="center", va="center", alpha=0.95)


def check_overlaps(fig, ax, placed) -> List[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    boxes = []
    for t in ax.texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width == 0 or bb.height == 0:
            continue
        c = inv.transform(bb)
        boxes.append(((float(c[:, 0].min()), float(c[:, 1].min()),
                       float(c[:, 0].max()), float(c[:, 1].max())), t.get_text()[:26]))
    issues = []
    for i, (b1, t1) in enumerate(boxes):
        for b2, t2 in boxes[i + 1:]:
            if b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]:
                area = (min(b1[2], b2[2]) - max(b1[0], b2[0])) * (min(b1[3], b2[3]) - max(b1[1], b2[1]))
                if area > 0.0015:
                    issues.append(f"TEXT x TEXT  {t1!r} <-> {t2!r} (area {area:.4f})")
    for ext in placed:
        for b, t in boxes:
            if b[0] < ext[1] and ext[0] < b[2] and b[1] < ext[3] and ext[2] < b[3]:
                area = (min(b[2], ext[1]) - max(b[0], ext[0])) * (min(b[3], ext[3]) - max(b[1], ext[2]))
                if area > 0.0015:
                    issues.append(f"TEXT x IMAGE {t!r} (area {area:.4f})")
    return issues


def build_poster(size: str = "A1") -> str:
    w_in, h_in = SIZES[size]
    placed: List[tuple] = []
    fig = plt.figure(figsize=(w_in, h_in), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=BG, transform=ax.transAxes))

    # header
    ax.add_patch(Rectangle((0, 0.885), 1, 0.115, facecolor=NAVY, transform=ax.transAxes))
    for i, c in enumerate([TEAL, AMBER, CORAL, VIOLET, MINT]):
        ax.add_patch(Rectangle((0.055 + i * 0.016, 0.885), 0.014, 0.004, facecolor=c,
                               transform=ax.transAxes))
    ax.text(0.030, 0.958, "NMS-Free Real-Time Detection:  DETR -> RT-DETR",
            transform=ax.transAxes, fontsize=29, color=WHITE, weight="bold", va="center")
    ax.text(0.030, 0.913,
            "Why moving duplicate suppression from post-processing into the loss "
            "buys accuracy and speed at once",
            transform=ax.transAxes, fontsize=12.6, color="#bcd4e4", va="center")
    ax.text(0.560, 0.958, "Carion et al., ECCV 2020  ·  arXiv:2005.12872",
            transform=ax.transAxes, fontsize=9.8, color="#9fc0d6", va="center", ha="right")
    ax.text(0.560, 0.933, "Lv et al., CVPR 2024  ·  arXiv:2304.08069",
            transform=ax.transAxes, fontsize=9.8, color="#9fc0d6", va="center", ha="right")
    for i, (big, lab, c) in enumerate([("42.0", "DETR AP (41M)", TEAL),
                                       ("53.1", "RT-DETR AP", AMBER),
                                       ("108", "FPS on T4 (FP16)", CORAL)]):
        card(ax, 0.600 + i * 0.133, 0.898, 0.125, 0.062, big, lab, c)

    colw = 0.305
    xs = [0.028, 0.348, 0.668]
    top = 0.862
    row1_h = 0.375
    row2_h = 0.215
    row3_h = 0.075

    # 1 problem
    panel(ax, xs[0], top - row1_h, colw, row1_h, "The NMS dilemma", CORAL, TINT["problem"], "1")
    rich(ax, xs[0] + 0.016, top - 0.058, [
        (CORAL, "Two thresholds move accuracy and time in opposite directions"),
        (MUTED, "conf 0.001 -> 0.05:  AP 52.9 -> 51.2,  NMS 2.36 -> 1.06 ms"),
        (MUTED, "IoU 0.5 -> 0.8:  AP 52.1 -> 52.8,  NMS 2.24 -> 2.46 ms"),
        (CORAL, "YOLO papers report model speed, excluding NMS time"),
    ], size=10.2, step=0.0152)
    img = load("poster/figs/fig_nms_cost.png")
    if img is not None:
        framed_image(ax, img, (xs[0] + 0.018, xs[0] + colw - 0.018,
                               top - row1_h + 0.020, top - row1_h + 0.150), CORAL, placed)
    ax.text(xs[0] + 0.016, top - row1_h + 0.006,
            "Fig. 1  NMS cost model (our measurement; direction matches paper Table 1)",
            transform=ax.transAxes, fontsize=8.0, color=MUTED, va="top")

    # 2 method
    panel(ax, xs[1], top - row1_h, colw, row1_h, "Three moves", TEAL, TINT["method"], "2")
    flow_diagram(ax, xs[1] + 0.014, top - 0.060, colw - 0.028, 0.152)
    rich(ax, xs[1] + 0.016, top - 0.216, [
        (TEAL, "1) Duplicates -> loss: optimal bipartite matching (one slot per GT)"),
        (MUTED, "     matching cost = probability, loss = log-likelihood"),
        (MUTED, "     box loss = scale-invariant GIoU + L1 (weights 2 / 5)"),
        (TEAL, "2) Compute -> design: encoder was 49% of GFLOPs, 11% of AP"),
        (MUTED, "     AIFI (S5 only, 400 tokens) + CCFF (conv fusion)"),
        (TEAL, "3) Init -> quality: U = ||P - C|| in the loss, AP 47.9 -> 48.7"),
    ], size=9.9, step=0.0150)

    # 3 results
    panel(ax, xs[2], top - row1_h, colw, row1_h, "Accuracy and speed together", SKY, TINT["result"], "3")
    results_table(ax, xs[2] + 0.016, top - 0.070, colw - 0.032, 0.104, placed)
    img = load("poster/figs/fig_ap_vs_fps.png")
    if img is not None:
        framed_image(ax, img, (xs[2] + 0.018, xs[2] + colw - 0.018,
                               top - row1_h + 0.020, top - row1_h + 0.140), SKY, placed)
    ax.text(xs[2] + 0.016, top - row1_h + 0.006,
            "Fig. 2  Accuracy vs speed (paper Table 2): RT-DETR is top-right",
            transform=ax.transAxes, fontsize=8.0, color=MUTED, va="top")
    ax.text(xs[2] + 0.016, top - 0.190,
            "Conditional comparison: 640x640 / 72 ep (RT-DETR) vs 800x1333 / 500 ep (DETR)",
            transform=ax.transAxes, fontsize=8.4, color=CORAL, va="top", weight="bold")

    # 4 our experiments
    r2 = top - row1_h - 0.022
    panel(ax, xs[0], r2 - row2_h, colw, row2_h, "Our own experiments", MINT, TINT["exp"], "4")
    gallery = [
        ("poster/figs/fig_l1_giou.png", TEAL, "L1 scales with size, GIoU does not (9.44x vs 1.000x)"),
        ("poster/figs/fig_attention.png", VIOLET, "Attention cost ~ M^2 (42.3x for 8x tokens)"),
        ("poster/figs/fig_encoder.png", SKY, "Encoder variant latency order matches the paper"),
        ("poster/figs/fig_uncertainty.png", CORAL, "Uncertainty alone has a degenerate optimum"),
    ]
    gw = (colw - 0.036 - 0.010) / 2
    gh = 0.076
    for i, (rel, color, cap) in enumerate(gallery):
        gx = xs[0] + 0.016 + (i % 2) * (gw + 0.010)
        gy = r2 - 0.056 - (i // 2) * (gh + 0.032)
        im = load(rel)
        if im is not None:
            framed_image(ax, im, (gx, gx + gw, gy - gh, gy), color, placed)
        ax.text(gx, gy - gh - 0.004, cap, transform=ax.transAxes, fontsize=7.4, color=MUTED, va="top")

    # 5 limitations
    panel(ax, xs[1], r2 - row2_h, colw, row2_h, "Limitations we kept honest", AMBER, TINT["limit"], "5")
    rich(ax, xs[1] + 0.016, r2 - 0.056, [
        (AMBER, "Small objects still weak: AP_S 34.8 (-0.5 vs YOLOv8-L; R101 -0.9)"),
        (AMBER, "49% GFLOPs / 11% AP is a second-hand statistic, not our measurement"),
        (AMBER, "138% / 120% are relative counts, not percentage-point gains"),
        (AMBER, "Selection != objective: uncertainty changes the loss, not the ranking"),
        (AMBER, "2D COCO only: depth, BEV and multi-view remain untested"),
    ], size=9.7, step=0.0150)

    # 6 flow
    panel(ax, xs[2], r2 - row2_h, colw, row2_h, "Study trail and what is next", VIOLET, TINT["flow"], "6")
    timeline(ax, xs[2] + 0.016, r2 - 0.058, colw - 0.032, [
        ("D1", "MAE", SKY), ("D2", "BEiT", SKY), ("D3", "iBOT", SKY),
        ("D4-6", "DeepSeek", TEAL), ("D7", "DETR", AMBER), ("D8", "RT-DETR", CORAL),
    ])
    rich(ax, xs[2] + 0.016, r2 - 0.130, [
        (VIOLET, "16 deep-dive docs, 4 runnable demos, 22 charts, 0 citation errors"),
        (VIOLET, "Next: GLIP + SAM (open-vocab detection -> prompt segmentation)"),
        (VIOLET, "Then: CLIP -> LLaVA (2 days) -> VL-JEPA   |   parallel 3D track"),
        (MUTED, "3D: DETR3D -> PETR -> StreamPETR -> Sparse4D -> occupancy"),
    ], size=9.7, step=0.0150)

    # footer
    fy = r2 - row2_h - 0.022
    ax.add_patch(Rectangle((0, fy - row3_h), 1, row3_h, facecolor=NAVY, transform=ax.transAxes))
    ax.text(0.030, fy - row3_h * 0.64,
            "Move duplicate removal into the loss, move compute into the design, "
            "move initialization into quality.",
            transform=ax.transAxes, fontsize=16.0, color=WHITE, weight="bold", va="center")
    ax.text(0.030, fy - row3_h * 0.22,
            "arXiv:2005.12872  ·  arXiv:2304.08069  ·  code: github.com/facebookresearch/detr, "
            "github.com/lyuwenyu/rt-detr  ·  figures: our Day 07-08 measurements (synthetic, CPU)",
            transform=ax.transAxes, fontsize=9.4, color="#9fc0d6", va="center")
    for i, (txt, c) in enumerate([("set prediction", TEAL), ("hybrid encoder", AMBER),
                                   ("uncertainty-minimal", VIOLET)]):
        chip(ax, 0.700 + i * 0.098, fy - row3_h * 0.68, 0.092, 0.030, txt, c, fontsize=8.6)

    issues = check_overlaps(fig, ax, placed)
    if issues:
        print(f"  [layout] overlaps: {len(issues)}")
        for m in issues[:14]:
            print("   -", m)
    else:
        print("  [layout] no overlaps")

    path = os.path.join(OUT, f"poster_det_rtdetr_{size}.pdf")
    fig.savefig(path)
    fig.savefig(os.path.join(OUT, f"poster_det_rtdetr_{size}.png"), dpi=60)
    plt.close(fig)
    print(f"  saved {os.path.relpath(path, ROOT)} (+ PNG preview)")
    return path


def build_slides() -> List[str]:
    slides = [
        ("NMS-Free Real-Time Detection: DETR -> RT-DETR",
         ["Moving duplicate suppression from post-processing into the loss",
          "buys accuracy and speed at the same time."], "poster/figs/fig_nms_cost.png", CORAL),
        ("Three moves",
         ["1) Duplicates -> loss: optimal bipartite matching (one slot per GT)",
          "2) Compute -> design: AIFI (S5 only) + CCFF (conv fusion)",
          "3) Init -> quality: U = ||P - C|| added to the loss",
          "42.0 AP (DETR, 41M)  ->  53.1 AP / 108 FPS (RT-DETR-R50)"],
         "poster/figs/fig_encoder.png", TEAL),
        ("What our own experiments add",
         ["Encoder variant latency order reproduces the paper exactly:",
          "A < D_S5 < E < B < D < C",
          "Attention cost ~ M^2:  8x tokens -> 42.3x time (theory 64x)",
          "Uncertainty alone has a degenerate optimum (s 0.90 -> 0.690)"],
         "poster/figs/fig_encoder.png", MINT),
        ("Limitations and what is next",
         ["AP_S 34.8 (-0.5 vs YOLOv8-L)  |  49/11 is 2nd-hand",
          "138/120 = relative counts, not %-points",
          "Selection != objective (loss changes, ranking does not)",
          "Next: GLIP + SAM -> CLIP -> LLaVA -> VL-JEPA  |  3D track"],
         "poster/figs/fig_uncertainty.png", AMBER),
    ]
    paths = []
    for i, (title, lines, img_rel, color) in enumerate(slides, start=1):
        placed: List[tuple] = []
        fig = plt.figure(figsize=(13.33, 7.5), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=BG, transform=ax.transAxes))
        ax.add_patch(Rectangle((0, 0.86), 1, 0.14, facecolor=NAVY, transform=ax.transAxes))
        ax.add_patch(Rectangle((0, 0.855), 1, 0.005, facecolor=color, transform=ax.transAxes))
        ax.text(0.05, 0.930, title, transform=ax.transAxes, fontsize=27, color=WHITE,
                weight="bold", va="center")
        ax.text(0.055, 0.50, "\n".join(lines), transform=ax.transAxes, fontsize=15.0, color=INK,
                va="center", linespacing=1.9)
        im = load(img_rel)
        if im is not None:
            framed_image(ax, im, (0.575, 0.955, 0.16, 0.80), color, placed)
        issues = check_overlaps(fig, ax, placed)
        if issues:
            print(f"  [slide {i}] overlaps {len(issues)}")
            for m in issues[:6]:
                print("   -", m)
        p = os.path.join(OUT, f"slide_{i}.pdf")
        fig.savefig(p)
        fig.savefig(os.path.join(OUT, f"slide_{i}.png"), dpi=60)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="A1", choices=list(SIZES))
    ap.add_argument("--slides", action="store_true")
    args = ap.parse_args()
    if args.slides:
        for p in build_slides():
            print("  saved", os.path.relpath(p, ROOT))
    else:
        build_poster(args.size)


if __name__ == "__main__":
    main()
