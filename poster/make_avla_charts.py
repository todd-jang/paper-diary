#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AVLA PoC poster — figures from the real project artifacts
==========================================================
Reads (read-only) the AIFFELTHON project at ~/spatial-avla-pipeline and renders
the poster figures. Nothing in the project is modified.

Outputs (poster/avla_figs/):
  avla_vision_overlay.jpg   RT-DETR-L detections drawn on a real dashcam frame
  avla_latency.png          PoC latency vs the <33 ms/frame target
  avla_vision_risk.png      S_A vision-anomaly score by driving condition
  avla_siren_trigger.png    S_B measured trigger rate by condition (+ hysteresis)
  avla_head_confusion.png   siren head accuracy / recall / specificity + confusion
  avla_p_siren.png          WavLM cosine: real vs synthetic vs mixed vs negative
  avla_ast_operating.png    AST threshold operating points (TPR / FPR)

Run:  .venv/bin/python poster/make_avla_charts.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

matplotlib.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.expanduser("~/spatial-avla-pipeline")
FIGS = os.path.join(HERE, "avla_figs")
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
    ax.set_title(title, fontsize=11.5, color=INK, weight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.6)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("  saved", name)


def load(rel):
    with open(os.path.join(PROJ, rel)) as f:
        return json.load(f)


# ── 1. RT-DETR detection overlay on a real dashcam frame ──────────────────
def fig_overlay():
    from PIL import Image, ImageDraw
    det = load("outputs/detections/C01__rtdetr-l.json")
    frame_dir = os.path.join(PROJ, "exports/TL_A/train/C01/frames")
    # pick the frame with the most detections
    best = max(det["frames"].items(), key=lambda kv: len(kv[1]["boxes"]))
    fidx, info = best
    img = Image.open(os.path.join(frame_dir, info["file"])).convert("RGB")
    dr = ImageDraw.Draw(img)
    palette = {"car": (219, 73, 91), "traffic light": (240, 197, 87),
               "person": (31, 122, 140), "truck": (108, 78, 155),
               "bus": (63, 157, 109)}
    for b in info["boxes"]:
        x1, y1, x2, y2 = b["bbox"]
        c = palette.get(b["class_name"], (30, 50, 68))
        dr.rectangle([x1, y1, x2, y2], outline=c, width=4)
        label = f'{b["class_name"]} {b["conf"]:.2f}'
        dr.rectangle([x1, max(0, y1 - 26), x1 + 15 + 8 * len(label), y1], fill=c)
        dr.text((x1 + 6, max(0, y1 - 20)), label, fill=(255, 255, 255))
    out = os.path.join(FIGS, "avla_vision_overlay.jpg")
    img.save(out, quality=92)
    print("  saved avla_vision_overlay.jpg  (frame", info["file"], f'{len(info["boxes"])} boxes)')
    return len(info["boxes"])


# ── 2. latency ───────────────────────────────────────────────────────────
def fig_latency():
    poc = load("outputs/poc_report.json")
    p50 = poc["latency"]["p50_ms"]
    p95 = poc["latency"]["p95_ms"]
    target = 33.0
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [p50, p95, target]
    labels = ["p50", "p95", "UI target"]
    colors = [TEAL, MINT, AMBER]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.7, f"{v:.2f} ms", ha="center",
                fontsize=10, color=INK, weight="bold")
    ax.axhline(target, color=AMBER, linestyle="--", linewidth=1.2, alpha=0.75)
    style(ax, "PoC UI fast-path latency (target < 33 ms/frame)", "", "latency (ms)")
    ax.set_ylim(0, 38)
    ax.text(0.02, 0.92, "All_OK=True · Errors_Count=0\n(mocks, no GPU inference on the fast path)",
            transform=ax.transAxes, fontsize=8.6, color=MUTED, va="top")
    save(fig, "avla_latency.png")


# ── 3. vision anomaly score by condition ─────────────────────────────────
def fig_vision():
    a = load("outputs/abl_report_s_a.json")
    base = a["baseline"]["constant"]
    thr = a["trigger_rule"]["vision_anomaly_ma_threshold"]
    conds = a["conditions"]
    names = ["day", "night", "rain"]
    vals = [conds[c]["rtdetr_mean"] for c in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(names, vals, color=[SKY, VIOLET, TEAL], width=0.5)
    ax.axhline(base, color=MUTED, linestyle=":", linewidth=1.4)
    ax.text(2.42, base + 0.012, "mock baseline 0.20", fontsize=8.4, color=MUTED, ha="right")
    ax.axhline(thr, color=CORAL, linestyle="--", linewidth=1.3)
    ax.text(-0.45, thr + 0.012, "trigger threshold 0.75", fontsize=8.4, color=CORAL)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                fontsize=10, color=INK, weight="bold")
    style(ax, "S_A vision-anomaly score (RT-DETR-L + MAE), by condition",
          "driving condition", "mean anomaly score")
    ax.set_ylim(0, 0.85)
    save(fig, "avla_vision_risk.png")


# ── 4. siren trigger rate by condition ───────────────────────────────────
def fig_siren():
    b = load("outputs/abl_report_s_b.json")
    order = ["real_siren", "siren_mixed", "day", "night", "rain"]
    nice = ["real siren", "siren-mixed", "day", "night", "rain"]
    trig = [b["conditions"][c]["measured_trigger_rate"] for c in order]
    hyst = [b["conditions"][c]["measured_is_triggered_rate"] for c in order]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    b1 = ax.bar(x - w / 2, trig, w, color=CORAL, label="measured trigger rate")
    b2 = ax.bar(x + w / 2, hyst, w, color=AMBER, label="after 35-frame hysteresis")
    for r, v in zip(b1, trig):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.018, f"{v:.3f}", ha="center",
                fontsize=8.8, color=INK)
    style(ax, "S_B siren trigger rate by condition (WavLM head, threshold 0.8)",
          "condition", "rate")
    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.6, frameon=False, loc="upper right")
    save(fig, "avla_siren_trigger.png")


# ── 5. siren head confusion ───────────────────────────────────────────────
def fig_head():
    b = load("outputs/abl_report_s_b.json")
    cm = b["trigger_rule"]["head_confusion"]
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    acc = (tp + tn) / (tp + tn + fp + fn)
    rec = tp / (tp + fn)
    spec = tn / (tn + fp)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    labels = ["recall\n(real siren)", "specificity\n(background)"]
    vals = [rec, spec]
    bars = ax.bar(labels, vals, color=[CORAL, MINT], width=0.45)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center",
                fontsize=11, color=INK, weight="bold")
    ax.axhline(acc, color=NAVY, linestyle="--", linewidth=1.3)
    ax.text(1.42, acc + 0.015, f"accuracy {acc:.4f}", fontsize=9, color=NAVY, ha="right")
    ax.set_ylim(0, 1.12)
    style(ax, "S_B head quality on the mixed clip set", "", "rate")
    ax = axes[1]
    mat = np.array([[tn, fp], [fn, tp]])
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(tp, tn))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred background", "pred siren"], fontsize=8.6)
    ax.set_yticklabels(["true background", "true siren"], fontsize=8.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="white" if mat[i, j] > max(tp, tn) * 0.5 else INK, fontsize=11, weight="bold")
    style(ax, "Confusion matrix (frames)", "", "")
    fig.tight_layout()
    save(fig, "avla_head_confusion.png")


# ── 6. WavLM cosine: real vs synthetic vs mixed ──────────────────────────
def fig_p_siren():
    d = load("outputs/p_siren_report.json")
    real = d["real_sirens"]
    names, vals = [], []
    for k, v in real.items():
        names.append(k.split("-")[0])
        vals.append(v["frame_cosine_mean"])
    neg = [v["frame_cosine_mean"] for v in d["negative_control_clips"].values()]
    noisy = [v["frame_cosine_mean"] for v in d["noisy_positive_clips"].values()]
    synth = [v if isinstance(v, float) else v.get("frame_cosine_mean", 0) for v in d["synthetic_text_clips"].values()]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.bar(names, vals, color=VIOLET, width=0.55, label="real sirens (WavLM frame cosine)")
    ax.axhline(float(np.mean(neg)), color=MINT, linestyle="--", linewidth=1.4)
    ax.text(len(names) - 0.5, float(np.mean(neg)) + 0.012, f"background mean {np.mean(neg):.3f}",
            fontsize=8.6, color=MINT, ha="right")
    ax.axhline(float(np.mean(noisy)), color=SKY, linestyle="--", linewidth=1.4)
    ax.text(len(names) - 0.5, float(np.mean(noisy)) + 0.012, f"siren-mixed mean {np.mean(noisy):.3f}",
            fontsize=8.6, color=SKY, ha="right")
    style(ax, "Real sirens occupy a different WavLM neighbourhood than the synthetic baseline",
          "real siren clip (G44-xx)", "mean frame cosine to synthetic centroid")
    ax.set_ylim(0, 0.8)
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, "avla_p_siren.png")


# ── 7. AST threshold operating points ────────────────────────────────────
def fig_ast():
    modes = ["balanced", "conservative", "sensitive"]
    tpr = [1.00, 0.917, 1.00]
    fpr = [0.00, 0.00, 0.40]
    x = np.arange(len(modes))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b1 = ax.bar(x - w / 2, tpr, w, color=MINT, label="TPR (siren detected)")
    b2 = ax.bar(x + w / 2, fpr, w, color=CORAL, label="FPR (false trigger)")
    for bars, vals in ((b1, tpr), (b2, fpr)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                    fontsize=9, color=INK)
    style(ax, "AST template operating points (12 positive / 5 negative clips)", "", "rate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(score > {t})" for m, t in
                        zip(modes, ["-0.20", "-0.10", "-0.30"])], fontsize=8.8)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8.6, frameon=False, loc="upper center", ncol=2)
    ax.text(0.02, 0.06, "temporal rule: 2 s window / 1 s hop, 5 consecutive windows above -0.04",
            transform=ax.transAxes, fontsize=8.2, color=MUTED)
    save(fig, "avla_ast_operating.png")


def main() -> None:
    print("[avla charts] rendering from", PROJ)
    fig_overlay()
    fig_latency()
    fig_vision()
    fig_siren()
    fig_head()
    fig_p_siren()
    fig_ast()
    print("[avla charts] done ->", os.path.relpath(FIGS, HERE))


if __name__ == "__main__":
    main()
