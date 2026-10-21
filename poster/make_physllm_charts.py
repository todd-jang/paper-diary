#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysLLM combined-poster figures (G0 -> N02 -> N03)
=================================================
All numbers are read from the project artifacts (read-only):
  N02: ~/Downloads/physllmN02/N02_Pilot_Baseline_v1.0.md, N02_preflight_report_v1.0.3.json
  N03: ~/Downloads/physllmN03/.../N03_Architecture_Scope_v1.0.md, reports/posters/*.pdf
  G0 : ~/Downloads/physllm/output*.png

Outputs (poster/physllm_figs/):
  physllm_cosine.png      N02 layer-wise input-update cosine (GPT-2 vs Pythia-160M)
  physllm_branch.png      N02 final-block branch decomposition (attention vs MLP)
  physllm_n03_bands.png   N03 depth-band summary (residual ratio + normalized entropy)
  physllm_moe.png         N03 MoE anatomy diagram (native, no data)
  physllm_manifest.png    N02 main-evaluation manifest design
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "physllm_figs")
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
WHITE = "#ffffff"

# ── N02 layer-wise summary (N02_Pilot_Baseline_v1.0.md) ──
GPT2_COS = [0.0923, 0.4098, 0.2358, 0.8696, 0.8510, 0.7161, 0.5338, 0.3264, 0.2095, 0.1410, 0.1354, -0.7686]
PYTHIA_COS = [0.0729, 0.1981, 0.3768, 0.5628, 0.2774, 0.1245, 0.0220, 0.0779, -0.4252, -0.7207, -0.8402, -0.3785]
GPT2_UPD = [11.1945, 2.0234, 3.3507, 0.0797, 0.0646, 0.0516, 0.0446, 0.0500, 0.0572, 0.0761, 0.1552, 0.9656]
PYTHIA_UPD = [15.1212, 1.2162, 1.0953, 2.3851, 0.0963, 0.1044, 0.1056, 0.1383, 0.1631, 0.2600, 0.4998, 2.1735]
GPT2_OUT = [11.3307, 2.6171, 3.7178, 1.0683, 1.0535, 1.0332, 1.0205, 1.0144, 1.0125, 1.0150, 1.0390, 0.6309]
PYTHIA_OUT = [15.2271, 1.7361, 1.7529, 3.0631, 1.0269, 1.0156, 1.0074, 1.0210, 0.9529, 0.8486, 0.6456, 2.0271]

# branch decomposition, final block (GPT-2 / Pythia, 5 prompts)
# 실제 보고된 값만 사용 (N02_Pilot_Baseline_v1.0.md) — 추측 금지
BRANCH = {
    "GPT-2": {"attn": [2717.872, 2726.625, 2738.920, 2731.693, 3967.207],
              "mlp": [357.245, 347.067, 379.905, 404.561, 1371.641],
              "cos": [0.366, 0.366, 0.392, 0.374, 0.431], "err": 2.3e-5,
              "prompts": ["general", "technical", "academic", "question", "code+NL"]},
    "Pythia-160M": {"attn": [58.621, 57.949], "mlp": [330.531, 348.995],
                     "cos": [0.723, 0.757], "err": 3.0e-6,
                     "prompts": ["general", "technical"]},
}
PROMPTS = ["P1 general", "P2 technical", "P3 academic", "P4 question", "P5 code+NL"]

# N03 depth-band summary (N03 A1 landscape EN poster)
BANDS = ["Early", "Middle", "Late"]
RATIO = [1.351, 1.234, 1.137]
H_LN4 = [0.952, 0.979, 0.974]


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


def fig_cosine():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    x = np.arange(12)
    ax = axes[0]
    ax.plot(x, GPT2_COS, marker="o", color=SKY, linewidth=2.2, label="GPT-2")
    ax.plot(x, PYTHIA_COS, marker="s", color=CORAL, linewidth=2.2, label="Pythia-160M")
    ax.axhline(0, color=MUTED, linewidth=1.0, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7.6)
    style(ax, "N02 · layer-wise input-update cosine", "transformer block", "cos(h_l, dh_l)")
    ax.legend(fontsize=8.6, frameon=False, loc="lower left")
    ax = axes[1]
    ax.plot(x, GPT2_UPD, marker="o", color=SKY, linewidth=2.2, label="GPT-2")
    ax.plot(x, PYTHIA_UPD, marker="s", color=CORAL, linewidth=2.2, label="Pythia-160M")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7.6)
    style(ax, "N02 · length-adjusted update norm", "transformer block", "||dh_l||_F / sqrt(T)")
    ax.legend(fontsize=8.6, frameon=False)
    fig.tight_layout()
    save(fig, "physllm_cosine.png")


def fig_branch():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    x = np.arange(5)
    w = 0.38
    for ax, model, c1, c2 in ((axes[0], "GPT-2", SKY, MINT), (axes[1], "Pythia-160M", VIOLET, AMBER)):
        b = BRANCH[model]
        n = len(b["attn"])
        xs = np.arange(n)
        ax.bar(xs - w / 2, b["attn"], w, color=c1, label="attention norm")
        ax.bar(xs + w / 2, b["mlp"], w, color=c2, label="MLP norm")
        ax.set_xticks(xs)
        ax.set_xticklabels(b["prompts"], fontsize=7.4, rotation=18)
        style(ax, f"{model} · final-block branch split", "", "Frobenius norm (pre-residual)")
        ax.legend(fontsize=8.4, frameon=False)
        ax.text(0.01, 0.94, f"branch cosine {min(b['cos']):.3f}-{max(b['cos']):.3f} · "
                            f"max identity residual {b['err']:.1e}",
                transform=ax.transAxes, fontsize=7.6, color=MUTED, va="top")
        if n < 5:
            ax.text(0.99, 0.06, "only P1/P2 reported in the frozen pilot summary",
                    transform=ax.transAxes, fontsize=7.2, color=CORAL, ha="right")
    fig.tight_layout()
    save(fig, "physllm_branch.png")


def fig_bands():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    bars = ax.bar(BANDS, RATIO, color=[SKY, TEAL, VIOLET], width=0.5)
    for r, v in zip(bars, RATIO):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                fontsize=10.5, color=INK, weight="bold")
    style(ax, "N03 · mean output/input norm ratio by depth band", "", "ratio (dimensionless)")
    ax.set_ylim(0, 1.6)
    ax = axes[1]
    bars = ax.bar(BANDS, H_LN4, color=[AMBER, CORAL, MINT], width=0.5)
    for r, v in zip(bars, H_LN4):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center",
                fontsize=10.5, color=INK, weight="bold")
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(2.42, 1.003, "uniform over top-k", fontsize=8, color=MUTED, ha="right")
    style(ax, "N03 · normalized routing entropy H/ln k", "", "H / ln 4")
    ax.set_ylim(0.9, 1.02)
    fig.tight_layout()
    save(fig, "physllm_n03_bands.png")


def fig_moe():
    """Native diagram of the gpt-oss-120b MoE anatomy used in N03."""
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")
    # stack of blocks
    for i in range(6):
        y = 2 + i * 5
        col = NAVY if i % 2 == 0 else "#1d4666"
        ax.add_patch(FancyBboxPatch((3, y), 16, 4, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    facecolor=col, edgecolor="none"))
        ax.text(11, y + 2, f"block {i}" + (" (full attn)" if i % 2 == 0 else " (sliding 128)"),
                ha="center", va="center", color=WHITE, fontsize=8.2)
    # experts
    for i in range(8):
        for j in range(4):
            x = 26 + i * 4.2
            y = 3 + j * 7
            ax.add_patch(FancyBboxPatch((x, y), 3.4, 5.4, boxstyle="round,pad=0.2,rounding_size=0.6",
                                        facecolor=TEAL if (i + j) % 5 else AMBER, alpha=0.85, edgecolor="none"))
    ax.text(40, 31.5, "128 experts  ·  top-4 per token", ha="center", fontsize=9.4, color=TEAL, weight="bold")
    # routing arrow
    ax.add_patch(FancyBboxPatch((64, 10), 30, 13, boxstyle="round,pad=0.5,rounding_size=1.2",
                                facecolor="#f3f6f9", edgecolor="#d9e0e6"))
    ax.text(79, 20.5, "117B total  /  5.1B active", ha="center", fontsize=10.4, color=INK, weight="bold")
    ax.text(79, 16.6, "36 layers  ·  hidden 2880  ·  64 heads / 8 KV", ha="center", fontsize=8.8, color=MUTED)
    ax.text(79, 13.2, "MXFP4 weights (~60.8 GB)  ·  bf16 activations", ha="center", fontsize=8.8, color=MUTED)
    ax.annotate("", xy=(62, 16.5), xytext=(52, 16.5),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=AMBER, lw=1.6))
    ax.text(57, 18.4, "routing", ha="center", fontsize=8.6, color=AMBER, weight="bold")
    ax.text(50, 0.6, "per-token: top-4 expert ids + gate weights (softmax over the top-k logits, not the full 128)",
            ha="center", fontsize=8.2, color=MUTED)
    ax.text(50, 33.2, "PhysLLM N03 · M1 MoE routing-conditional target: openai/gpt-oss-120b",
            ha="center", fontsize=10, color=INK, weight="bold")
    save(fig, "physllm_moe.png")


def fig_manifest():
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    blocks = [
        (2, "400", "texts", SKY), (23, "100", "topic clusters", TEAL),
        (44, "4", "styles / topic", VIOLET), (65, "3", "length bands", AMBER),
    ]
    for x, big, lab, c in blocks:
        ax.add_patch(FancyBboxPatch((x, 14), 18, 11, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + 9, 21.4, big, ha="center", fontsize=17, color=WHITE, weight="bold")
        ax.text(x + 9, 16.6, lab, ha="center", fontsize=8.4, color=WHITE)
    ax.text(50, 11.2, "100 topics x 4 styles = 400 candidate texts   ·   length bands 16-64 / 65-128 / 129-256 tokens",
            ha="center", fontsize=8.8, color=INK)
    ax.text(50, 8.0, "two tokenizers (gpt2, pythia-160m)   ·   bootstrap 10,000 replicates   ·   "
                     "pilot prompts excluded from main evaluation",
            ha="center", fontsize=8.8, color=MUTED)
    ax.add_patch(FancyBboxPatch((2, 1.4), 96, 5.2, boxstyle="round,pad=0.3,rounding_size=1",
                                 facecolor="#fdf3e3", edgecolor="#efd9a8"))
    ax.text(50, 4.0, "preflight v1.0.3:  hard-check failures 0  |  warnings: 24 code texts without a domain word, "
                     "12 code texts share one template  |  frozen for confirmatory analysis: NO",
            ha="center", fontsize=8.4, color="#8a6d2b")
    ax.text(50, 28.4, "N02 · main-evaluation candidate manifest design (v1.0.3)", ha="center",
            fontsize=10.4, color=INK, weight="bold")
    save(fig, "physllm_manifest.png")


def main() -> None:
    print("[physllm figures] rendering")
    fig_cosine()
    fig_branch()
    fig_bands()
    fig_moe()
    fig_manifest()
    print("[physllm figures] done ->", os.path.relpath(FIGS, HERE))


if __name__ == "__main__":
    main()
