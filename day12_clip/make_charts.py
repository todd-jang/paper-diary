#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 12 CLIP charts — all numbers from day12_clip/00_검증팩트시트.md only."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "runs", "figures")


def _final(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    print("saved", name)


def c01_training_scale():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels1 = ["(image, text) pairs", "queries", "max pairs/query", "batch size", "epochs"]
    vals1 = [400_000_000, 500_000, 20_000, 32_768, 32]
    y = np.arange(len(vals1))[::-1]
    ax1.barh(y, vals1, color="#4472C4")
    ax1.set_yticks(y, labels1)
    ax1.set_xscale("log")
    ax1.set_title("WIT dataset & training regime (log scale)")
    for yi, v in zip(y, vals1):
        ax1.text(v * 1.15, yi, f"{v:,}", va="center", fontsize=9)
    ax1.set_xlim(10, 4e9)

    labels2 = ["RN50x64\n592 V100 x 18d", "largest ViT\n256 V100 x 12d"]
    vals2 = [592 * 18, 256 * 12]
    y2 = np.arange(len(vals2))[::-1]
    ax2.barh(y2, vals2, color="#ED7D31")
    ax2.set_yticks(y2, labels2)
    ax2.set_title("pre-training cost (GPU-days, derived: GPUs x days)")
    for yi, v in zip(y2, vals2):
        ax2.text(v + 250, yi, f"{v:,} GPU-days", va="center", fontsize=9)
    ax2.set_xlim(0, 12_500)
    fig.suptitle("arXiv 2103.00020 - Sec 2.1 / Sec 4 setup (factsheet-verified)", fontsize=10)
    _final(fig, "01_clip_training_scale.png")


def c02_table1_vngrams():
    ds = ["aYahoo", "ImageNet", "SUN"]
    vn = [72.4, 11.5, 23.0]
    clip = [98.4, 76.2, 58.5]
    x = np.arange(len(ds))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w / 2, vn, w, label="Visual N-Grams", color="#A5A5A5")
    ax.bar(x + w / 2, clip, w, label="CLIP (concluding model)", color="#4472C4")
    for xi, (a, b) in enumerate(zip(vn, clip)):
        ax.text(xi - w / 2, a + 1, f"{a}", ha="center", fontsize=9)
        ax.text(xi + w / 2, b + 1, f"{b}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, ds)
    ax.set_ylabel("zero-shot top-1 accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Table 1: CLIP vs prior zero-shot transfer (Visual N-Grams)")
    ax.legend()
    _final(fig, "02_clip_table1_vngrams.png")


def c03_dataset_deltas():
    names = ["STL10\n(zero-shot only)", "Stanford Cars", "Food101",
             "Kinetics-700", "UCF101", "Flowers102", "FGVCAircraft"]
    vals = [99.3, 20, 20, 14.5, 7.7, -10, -10]
    colors = ["#70AD47" if v >= 0 else "#C00000" for v in vals]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(names))
    ax.bar(x, vals, color=colors)
    for xi, v in zip(x, vals):
        off = 2 if v >= 0 else -5
        ax.text(xi, v + off, ("{:+.1f}".format(v) if names[xi] != "STL10\n(zero-shot only)"
                              else f"{v}%"), ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylabel("percentage points vs supervised RN50 features (or absolute %)")
    ax.set_title("CLIP zero-shot on single datasets: wins and losses (Sec 3.1.2)")
    _final(fig, "03_clip_dataset_deltas.png")


def c04_robustness():
    sets = ["IN", "IN-V2", "IN-A", "IN-R", "ObjectNet", "IN-Sketch"]
    clip = [76.2, 70.1, 77.2, 88.9, 72.3, 60.2]
    ns = [88.3, 80.2, 84.9, 74.7, 68.5, 47.6]
    fix = [86.4, 78.0, 68.4, 80.0, 57.8, 59.1]
    x = np.arange(len(sets))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - w, clip, w, label="Zero-Shot CLIP", color="#4472C4")
    ax.bar(x, ns, w, label="NS EfficientNet-L2", color="#A5A5A5")
    ax.bar(x + w, fix, w, label="FixResNeXt101-32x48d V2", color="#ED7D31")
    ax.set_xticks(x, sets)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("Natural distribution shift: zero-shot CLIP vs supervised baselines (appendix table)")
    ax.legend(fontsize=8.5)
    _final(fig, "04_clip_robustness.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    c01_training_scale()
    c02_table1_vngrams()
    c03_dataset_deltas()
    c04_robustness()


if __name__ == "__main__":
    main()
