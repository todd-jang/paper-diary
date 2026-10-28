#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 13 LLaVA charts — all numbers from day13_llava/00_검증팩트시트.md only."""
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


def c01_training_stages():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels1 = ["Stage 1 (CC-595K)", "Stage 2 (158K instruct)"]
    data1 = [595.0, 158.0]   # K pairs
    y = np.arange(2)[::-1]
    ax1.barh(y, data1, color="#4472C4", height=0.5)
    ax1.set_yticks(y, labels1)
    ax1.set_title("Stage data volume (K pairs)")
    for yi, v in zip(y, data1):
        ax1.text(v + 8, yi, f"{v:.0f}K", va="center", fontsize=9)
    ax1.set_xlim(0, 700)
    stages2 = ["Stage 1", "Stage 2"]
    lrs = [2e-3, 2e-5]
    ep = [1, 3]
    y2 = np.arange(2)[::-1]
    ax2.barh(y2 + 0.15, lrs, 0.3, color="#ED7D31", label="learning rate (log)")
    ax2.barh(y2 - 0.15, ep, 0.3, color="#A5A5A5", label="epochs")
    ax2.set_xscale("log")
    ax2.set_yticks(y2, stages2)
    ax2.set_title("lr (2e-3 → 2e-5 = 100x down) / epochs")
    for yi, v in zip(y2, lrs):
        ax2.text(v * 1.4, yi + 0.15, f"{v:g}", va="center", fontsize=8.5)
    for yi, v in zip(y2, ep):
        ax2.text(v * 1.4, yi - 0.15, f"{v} ep", va="center", fontsize=8.5)
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.set_xlim(1e-5, 3e-2)
    fig.suptitle("arXiv 2304.08485 - Sec 4.2 / Sec 5 two-stage regime "
                 "(batch 128/32 - 8xA100 - <=4h + <=10h, factsheet-verified)", fontsize=10)
    _final(fig, "01_llava_training_stages.png")


def c02_table4_data_mix():
    rows = ["Full data", "Detail+Complex", "Conv+5%D+10%C", "Conv only", "No tuning"]
    conv = [83.1, 81.5, 81.0, 76.5, 22.0]
    detail = [75.3, 73.3, 68.4, 59.8, 24.0]
    comp = [96.5, 90.8, 91.5, 84.9, 18.5]
    allc = [85.1, 81.9, 80.5, 73.8, 21.5]
    x = np.arange(len(rows))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - 1.5 * w, conv, w, label="Conv", color="#8FAADC")
    ax.bar(x - 0.5 * w, detail, w, label="Detail", color="#A5A5A5")
    ax.bar(x + 0.5 * w, comp, w, label="Complex", color="#ED7D31")
    ax.bar(x + 1.5 * w, allc, w, label="All", color="#4472C4")
    for xi, v in zip(x, allc):
        ax.text(xi + 1.5 * w, v + 1.5, f"{v}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, rows)
    ax.set_ylabel("relative score (%) vs text-only GPT-4")
    ax.set_ylim(0, 105)
    ax.set_title("LLaVA-Bench (COCO, 90 questions): data-mixture ablation (Table 4)")
    ax.legend(fontsize=8.5, ncol=4)
    _final(fig, "02_llava_table4_data_mix.png")


def c03_table5_wild():
    models = ["OpenFlamingo", "BLIP-2", "LLaVA"]
    conv = [19.3, 54.6, 57.3]
    detail = [19.0, 29.1, 52.5]
    comp = [19.1, 32.9, 81.7]
    allc = [19.1, 38.1, 67.3]
    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 1.5 * w, conv, w, label="Conv", color="#8FAADC")
    ax.bar(x - 0.5 * w, detail, w, label="Detail", color="#A5A5A5")
    ax.bar(x + 0.5 * w, comp, w, label="Complex", color="#ED7D31")
    ax.bar(x + 1.5 * w, allc, w, label="All", color="#4472C4")
    for xi, v in zip(x, allc):
        ax.text(xi + 1.5 * w, v + 1.5, f"{v}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, models)
    ax.set_ylabel("relative score (%) vs text-only GPT-4 (mean, std omitted)")
    ax.set_ylim(0, 95)
    ax.set_title("LLaVA-Bench (In-the-Wild): harder bench, same judge protocol (Table 5)")
    ax.legend(fontsize=8.5, ncol=4)
    _final(fig, "03_llava_table5_in_the_wild.png")


def c04_scienceqa():
    models = ["Human", "GPT-3.5", "GPT-3.5\nw/ CoT", "GPT-4\n(text-only)",
              "LLaMA-Adapter", "MM-CoT\nLarge", "LLaVA", "LLaVA+\nGPT-4 judge"]
    vals = [88.40, 73.97, 75.17, 82.69, 85.19, 91.68, 90.92, 92.53]
    colors = ["#A5A5A5"] * 6 + ["#8FAADC", "#70AD47"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(models))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    ax.set_yticks(y, models)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(vals[6], color="#8FAADC", ls="--", lw=1)
    ax.axvline(82.69, color="#A5A5A5", ls=":", lw=1)
    ax.set_xlabel("All accuracy (%) - test 4241")
    ax.set_xlim(70, 95)
    ax.set_title("ScienceQA: LLaVA 90.92 → +GPT-4 judge 92.53 new SoTA (Table 7)")
    _final(fig, "04_llava_scienceqa.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    c01_training_stages()
    c02_table4_data_mix()
    c03_table5_wild()
    c04_scienceqa()


if __name__ == "__main__":
    main()
