#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 08 — RT-DETR 차트 생성기 (paper-diary 시각화 규약 v1)
=========================================================
입력: runs/rtdetr_demo_metrics.json (자체 실측)
출력: runs/chart_nms_cost.png · chart_attention_scaling.png
      chart_encoder_variants.png · chart_uncertainty_degeneracy.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
INK = "#1f2933"
GRID = "#d7dde3"
C1 = "#2f6f9f"
C2 = "#c1553b"
C3 = "#4a7c59"


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
    fig.savefig(os.path.join(RUNS, name), dpi=150)
    plt.close(fig)
    print("  saved", name)


def chart_nms(m):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = m["A_nms_cost"]["conf_sweep"]
    x = [r["conf"] for r in rows]
    ax = axes[0]
    ax.plot(x, [r["ms"] for r in rows], marker="o", color=C1, linewidth=2, label="NMS 시간")
    ax.set_xscale("log")
    style(ax, "conf 임계값 ↑ → NMS 시간 ↓ (원문 Table 1)", "conf 임계값", "NMS 시간 (ms)")
    ax2 = ax.twinx()
    ax2.plot(x, [r["ap_proxy"] for r in rows], marker="s", color=C2, linestyle="--",
             linewidth=1.8, label="AP proxy")
    ax2.set_ylabel("AP proxy (자체 실측)", fontsize=9.5, color=C2)
    ax2.set_ylim(0.85, 1.02)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center right")

    rows = m["A_nms_cost"]["iou_sweep"]
    x2 = [r["iou"] for r in rows]
    ax = axes[1]
    ax.plot(x2, [r["ms"] for r in rows], marker="o", color=C1, linewidth=2, label="NMS 시간")
    style(ax, "IoU 임계값 ↑ → NMS 시간 ↑ · AP proxy ↑ (원문 Table 1)", "NMS IoU 임계값", "NMS 시간 (ms)")
    ax2 = ax.twinx()
    ax2.plot(x2, [r["ap_proxy"] for r in rows], marker="s", color=C2, linestyle="--",
             linewidth=1.8, label="AP proxy")
    ax2.set_ylabel("AP proxy (자체 실측)", fontsize=9.5, color=C2)
    ax2.set_ylim(0.85, 1.02)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center right")
    save(fig, "chart_nms_cost.png")


def chart_attention(m):
    b = m["B_attention_scaling"]
    rows = b["scaling"]
    M = np.array([r["tokens"] for r in rows], dtype=float)
    T = np.array([r["ms"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.loglog(M, T, marker="o", color=C1, linewidth=2, label="실측 (CPU, d=256, head=8)")
    theory = T[0] * (M / M[0]) ** 2
    ax.loglog(M, theory, linestyle="--", color=C2, linewidth=1.6, label="이차 법칙(∝ M²) 기준선")
    style(ax, "Attention 비용의 토큰 스케일링 — AIFI가 S5(400토큰)만 보는 이유",
          "토큰 수 M", "attention 시간 (ms)")
    ax.legend(fontsize=8.5, frameon=False)
    ratio = b["ratio_8x"]
    ax.annotate(f"400 → 3200 토큰(8배) = 시간 {ratio:.1f}배\n(이론 64배, 실측은 메모리 한계로 저하)",
                xy=(M[-1], T[-1]), xytext=(0.06, 0.62), textcoords="axes fraction",
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.text(0.06, 0.12, f"RT-DETR v1: 후보 8400 vs S5 400 = {b['token_ratio_8400_over_400']:.0f}배 토큰 차\n"
                        f"(1x1 conv 80x80x256 = {b['conv_ms_80x80']:.2f}ms — attention 대비 파레토)",
            transform=ax.transAxes, fontsize=9, color=C3)
    save(fig, "chart_attention_scaling.png")


def chart_variants(m):
    rows = m["C_encoder_variants"]["rows"]
    labels = [r["variant"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ours = [r["ms"] for r in rows]
    paper = [r["paper_ms"] for r in rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, ours, 0.4, color=C1, label="자체 미니 구현 (CPU)")
    ax.bar(x + 0.2, paper, 0.4, color=C2, label="논문 Table 3 (T4)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style(ax, "하이브리드 인코더 변형별 지연 (절대값 비교 불가, 순서 비교)", "변형", "지연 (ms)")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    aps = [r["paper_ap"] for r in rows]
    pms = [r["paper_ms"] for r in rows]
    cols = [C1, C1, C1, C3, C2, C2]
    ax.scatter(pms, aps, s=110, c=cols, zorder=3, edgecolors="white", linewidths=1.2)
    for r in rows:
        ax.annotate(r["variant"], (r["paper_ms"], r["paper_ap"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9.5, color=INK)
    style(ax, "논문 Table 3: 정확도 vs 지연 (D_S5가 지연 -35%, AP +0.4)", "지연 (ms)", "AP")
    ax.annotate("D → D_S5\n지연 -35% · AP +0.4", xy=(7.9, 46.8), xytext=(9.2, 44.2),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    save(fig, "chart_encoder_variants.png")


def chart_uncertainty(m):
    d = m["D_uncertainty_degeneracy"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    t = [r["step"] for r in d["traj"]]
    ax.plot(t, [r["s"] for r in d["traj"]], color=C2, linewidth=2, label="U만 최소화: 분류 점수 s")
    ax.plot(t, [r["q"] for r in d["traj"]], color=C2, linestyle="--", linewidth=1.8,
            label="U만 최소화: 위치 품질 q")
    t2 = [r["step"] for r in d["traj_with_cls"]]
    ax.plot(t2, [r["s"] for r in d["traj_with_cls"]], color=C1, linewidth=2,
            label="분류 손실 포함: s")
    ax.plot(t2, [r["q"] for r in d["traj_with_cls"]], color=C1, linestyle="--", linewidth=1.8,
            label="분류 손실 포함: q")
    style(ax, "uncertainty 항의 퇴화 방향 (U 단독 vs 분류 손실 병행)",
          "최적화 스텝", "점수 / 품질 (0~1)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.text(0.03, 0.06, f"U 단독: s {d['degenerate']['s_final']:.3f} · q {d['degenerate']['q_final']:.3f}"
                       f"  (만나는 지점으로 수렴)\n분류 손실 병행: s {d['with_cls']['s_final']:.3f}"
                       f" · q {d['with_cls']['q_final']:.3f}  (품질이 확신을 따라감)",
            transform=ax.transAxes, fontsize=8.5, color=INK)
    save(fig, "chart_uncertainty_degeneracy.png")


def main():
    m = json.load(open(os.path.join(RUNS, "rtdetr_demo_metrics.json")))
    print("[charts] 생성 시작")
    chart_nms(m)
    chart_attention(m)
    chart_variants(m)
    chart_uncertainty(m)
    print("[charts] 완료 ->", RUNS)


if __name__ == "__main__":
    main()
