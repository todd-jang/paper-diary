#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 10 · 차트 4종 — day10_glip_sam/00_검증팩트시트.md의 표 수치만 사용
규칙:
  · 모든 수치는 논문 표/본문 + 조건(분할·백본·해상도)을 그림 하단 출처행에 명시.
  · "61.5"는(COCO 편입 변형) 이 그림에서 다루지 않는다 — 조건 없는 인용 금지.
  · SAM 인간평가·1점 mIoU 등 본문 수치 없는 축은 그리지 않는다.
사용법:  .venv/bin/python day10_glip_sam/make_charts.py
결과:    day10_glip_sam/runs/figures/*.png (4종)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "runs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

_avail = {f.name for f in fontManager.ttflist}
for _font in ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"]:
    if _font in _avail:
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = "#4878CF", "#EE854A", "#79AF7C", "#D1615D", "#8C8C8C"


def src_note(ax, text):
    ax.figure.text(0.01, 0.01, text, fontsize=6.5, color=C_GRAY, va="bottom")


# ── ① GLIP COCO zero-shot (Table 2, 2017val 열) ───────────────────────────
def chart_coco_zeroshot():
    rows = [
        ("DyHead-T (O365)", 43.6, False),
        ("GLIP-T (A) O365", 42.9, False),
        ("GLIP-T (B) O365", 44.9, False),
        ("GLIP-T (C) O365+GoldG", 46.7, False),
        ("GLIP-T (O365,G,C,S)", 46.6, False),
        ("GLIP-L (FourODs,G,Cap24M)", 49.8, False),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bars = ax.bar(range(len(values)), values,
                  color=[C_RED] * 5 + [C_BLUE])
    bars[1].set_alpha(0.55)                       # (A)는 기준 DyHead보다 낮음 — 낮게 표시
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(values):
        ax.text(i, v + 0.3, f"{v}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 56)
    ax.set_ylabel("COCO 2017val AP (zero-shot)")
    ax.set_title("GLIP zero-shot COCO: 분류기 로짓을 말로 교체한 탐지", fontsize=11)
    ax.axhline(43.6, color=C_GRAY, lw=0.8, ls="--")
    src_note(ax, "로: GLIP arXiv 2112.03857, Table 2 zero-shot 2017val 열 · COCO는 사전학습에 미편입(COCO-free) · 평가: 로짓 스왑 S = O P^T")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "01_glip_coco_zeroshot.png"))
    plt.close(fig)


# ── ② GLIP ODinW 13-avg few-shot → fully supervised (Table 11) ────────────
def chart_odinw():
    shots = ["0-shot", "1-shot", "3-shot", "5-shot", "10-shot", "fully(\n모든 데이터)"]
    x = [0, 1, 3, 5, 10, 22]
    series = {
        "DyHead-T(O365, 지도)": [None, 33.8, 43.6, 46.4, 50.8, 60.8],
        "GLIP-T": [46.5, 51.1, 54.9, 56.4, 58.4, 64.9],
        "GLIP-L": [52.1, 59.9, 62.1, 64.2, 64.9, 68.9],
    }
    colors = {"DyHead-T(O365, 지도)": C_GRAY, "GLIP-T": C_ORANGE, "GLIP-L": C_BLUE}
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for name, vals in series.items():
        xs = [xi for xi, v in zip(x, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, marker="o", color=colors[name], label=name)
        for xi, v in zip(xs, ys):
            ax.text(xi, v + 0.55, f"{v}", ha="center", fontsize=7.6, color=colors[name])
    ax.annotate("zero-shot GLIP-L(52.1)\n> 10-shot 지도 DyHead(50.8)",
                xy=(0, 52.1), xytext=(2.2, 53.5), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.9))
    ax.scatter([10], [52.1], color=C_BLUE, marker="v", s=42, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(shots, fontsize=8)
    ax.set_ylabel("ODinW 13-데이터셋 평균 AP")
    ax.set_title("ODinW few-shot 스윕: GLIP zero-shot이 지도 few-shot 몇샷보다 강한가", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(28, 71)
    src_note(ax, "로: GLIP arXiv 2112.03857, Table 11 13-avg AP 열(±: 3회 편차 원문) · 'fully' 열 = 해당 아키텍처 전체 데이터 지도학습")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "02_glip_odinw_shots.png"))
    plt.close(fig)


# ── ③ SAM 데이터 엔진 3단계 (§4/Fig. 6) ───────────────────────────────────
def chart_data_engine():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    stages = ["1단계\n보조-수동\n(120k장)", "2단계\n반자동\n(+180k장)", "3단계\n자동(점그리드)"]
    cum = [4.3e6, 10.2e6, 1.1e9]
    per_img = [44, 72, 100]
    ax = axes[0]
    ax.bar([0, 1, 2], cum, color=[C_ORANGE, C_GREEN, C_BLUE])
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(stages, fontsize=8)
    for i, v in enumerate([4_300_000, 10_200_000, 1_100_000_000]):
        ax.text(i, v * 1.35, f"{v/1e6:.1f}M" if v < 1e9 else "1.1B",
                ha="center", fontsize=9)
    ax.set_ylabel("누적 마스크 수(로그)")
    ax.set_title("데이터 엔진 누적: 4.3M → 10.2M → 1.1B", fontsize=10.5)
    ax = axes[1]
    ax.plot([0, 1, 2], [20, 44, 72], marker="o", color=C_ORANGE, label="단계 1→2")
    ax.plot([1, 2], [72, per_img[2]], marker="o", color=C_BLUE)
    for xi, v in zip([0, 1, 2], [20, 44, 72, 100]):
        ax.text(xi, v + 2.5, f"{v}", ha="center", fontsize=8.5)
    ax.axhline(100, color=C_BLUE, lw=0.8, ls="--")
    ax.text(0.08, 101.6, "평균 ~100/이미지(3단계)", fontsize=8, color=C_BLUE)
    ax.axvline(1.5, color=C_GRAY, lw=0.8, ls=":")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel("마스크/이미지 개수")
    ax.set_title("마스크/이미지가 모델 재학습과 함께 증가(모델 개선→라벨 개선)", fontsize=10.5)
    ax.set_ylim(0, 118)
    ax.legend(fontsize=7.5, loc="upper left")
    src_note(fig.axes[0], "로: SAM arXiv 2304.02643 §4/Fig.6 · 3단계 99.1% 자동 생성 · 품질검증: 500장 인간 재측정 94% 쌍 IoU>90%")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "03_sam_data_engine.png"))
    plt.close(fig)


# ── ④ SAM 평가: 에지(열세) + LVIS 프로포절(rare 우위) ─────────────────────
def chart_sam_eval():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax = axes[0]
    names = ["SAM", "EDETR"]
    ods = [0.768, 0.840]
    ax.bar(names, ods, color=[C_ORANGE, C_GRAY], width=0.5)
    for i, v in enumerate(ods):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.70, 0.87)
    ax.set_title("에지 감지 BSDS500 — SAM 미달", fontsize=10.5)
    ax.set_ylabel("ODS")
    ax = axes[1]
    cats = ["전체", "rare", "small"]
    sam = [59.3, 65.8, 45.5]
    vitdet = [63.0, 58.3, 51.7]
    xx = [0, 1, 2]
    ax.bar([x - 0.18 for x in xx], sam, width=0.36, color=C_ORANGE, label="SAM")
    ax.bar([x + 0.18 for x in xx], vitdet, width=0.36, color=C_GRAY, label="ViTDet-H")
    for x, v in zip(xx, sam):
        ax.text(x - 0.18, v + 0.4, f"{v}", ha="center", fontsize=8.5)
    for x, v in zip(xx, vitdet):
        ax.text(x + 0.18, v + 0.4, f"{v}", ha="center", fontsize=8.5)
    ax.set_xticks(xx)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("LVIS v1 프로포절 AR@1000")
    ax.set_title("객체 프로포절 — rare 클래스에서 SAM 우위", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 74)
    src_note(fig.axes[0], "로: SAM arXiv 2304.02643 Table 3(에지, 16×16 점 그리드→768 후보, Sobel) · Table 4(64×64 그리드·NMS 0.9, 이미지당 ~900 마스크)")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "04_sam_eval_edges_proposals.png"))
    plt.close(fig)


def main():
    chart_coco_zeroshot()
    chart_odinw()
    chart_data_engine()
    chart_sam_eval()
    files = sorted(os.listdir(FIG_DIR))
    for f in files:
        p = os.path.join(FIG_DIR, f)
        print(f"  {f}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
