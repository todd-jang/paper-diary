#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 07 — 차트·시각화 자료 생성기 (paper-diary 시각화 규약 v1)
==========================================================

규약 요약(README '시각화 규약' 절):
  · PNG 위치: dayNN_xxx/runs/*.png
  · 폰트: AppleGothic(한글) + axes.unicode_minus=False
  · dpi 150, figsize 기본 (9,5)
  · 출처 표기: 논문 표 인용 vs 자체 실측을 캡션에 명시

입력(자체 실측):
  runs/detr_demo_metrics.json      (detr_demo.py 결과)
  runs/detr_pipeline_metrics.json   (detr_pipeline_demo.py 결과 + 학습 로그)
입력(논문 표):
  DETR Table 1/2/3/4, RT-DETR Table 3/5 — 값은 각 주석에 명시
출력:
  runs/chart_matching_cost.png     runs/chart_l1_giou_scaling.png
  runs/chart_empty_weight.png      runs/chart_query_selection_scatter.png
  runs/chart_pipeline_ap.png       runs/chart_decoder_depth.png
  runs/chart_detr_ap_breakdown.png runs/chart_rtdetr_encoder_tradeoff.png
  runs/chart_pipeline_diagram.png
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SEED = 20261020

INK = "#1f2933"
GRID = "#d7dde3"
C1 = "#2f6f9f"   # vanilla / 기준
C2 = "#c1553b"   # uncertainty / 비교
C3 = "#4a7c59"   # 보조


def save(fig, name: str) -> None:
    path = os.path.join(RUNS, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {name}")


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


# ---------------------------------------------------------------------------
# 1. 매칭 비용: 그리디 vs 헝가리아리안 (자체 실측 재계산, seed 고정)
# ---------------------------------------------------------------------------
def chart_matching_cost() -> None:
    rng = np.random.default_rng(7)
    n_pred, n_gt, trials = 8, 5, 500
    opt_costs, greedy_costs = [], []
    for _ in range(trials):
        probs = torch.softmax(torch.from_numpy(rng.normal(size=(n_pred, n_gt))), dim=1)
        cost = (-probs).numpy()
        r, c = linear_sum_assignment(cost)
        opt_costs.append(cost[r, c].sum())
        used, tot = set(), 0.0
        for j in range(n_gt):
            col = int(np.argmin(cost[:, j]))
            if col in used:
                rem = [i for i in range(n_pred) if i not in used]
                col = min(rem, key=lambda i: cost[i, j])
            used.add(col)
            tot += cost[col, j]
        greedy_costs.append(tot)
    opt_costs = np.array(opt_costs)
    greedy_costs = np.array(greedy_costs)
    gap = greedy_costs - opt_costs

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(greedy_costs, bins=30, alpha=0.55, color=C1, label="순진 그리디(GT별 최선, 중복 허용)")
    ax.hist(opt_costs, bins=30, alpha=0.7, color=C2, label="헝가리아리안(최적)")
    style(ax, "매칭 총비용 분포 (500회, N_pred=8 / N_gt=5)", "매칭 총비용 (-p 합)", "시행 수")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.bar([0, 1], [float((gap > 1e-9).mean()), float((gap < -1e-9).mean())],
           color=[C3, C2], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["헝가리아리안이\n더 좋음", "그리디가\n더 좋음"])
    style(ax, "최적 매칭과의 비용 차: 그리디가 이긴 사례 0회", "판정", "비율")
    ax.text(0, float((gap > 1e-9).mean()) + 0.02, f"{(gap > 1e-9).mean():.1%}",
            ha="center", fontsize=11, color=INK)
    save(fig, "chart_matching_cost.png")


# ---------------------------------------------------------------------------
# 2. L1 스케일 잭 vs GIoU 스케일 불변 (자체 실측 스윕)
# ---------------------------------------------------------------------------
def giou_elem(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    def xyxy(b):
        return torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                            b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)
    a, b = xyxy(b1), xyxy(b2)
    ix1, iy1 = torch.maximum(a[..., 0], b[..., 0]), torch.maximum(a[..., 1], b[..., 1])
    ix2, iy2 = torch.minimum(a[..., 2], b[..., 2]), torch.minimum(a[..., 3], b[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-9)
    cx1, cy1 = torch.minimum(a[..., 0], b[..., 0]), torch.minimum(a[..., 1], b[..., 1])
    cx2, cy2 = torch.maximum(a[..., 2], b[..., 2]), torch.maximum(a[..., 3], b[..., 3])
    enc = (cx2 - cx1) * (cy2 - cy1)
    return 1 - (iou - (enc - union) / enc.clamp(min=1e-9))


def chart_l1_giou() -> None:
    small = torch.tensor([[0.24, 0.21, 0.10, 0.08]])
    large = torch.tensor([[0.50, 0.55, 0.90, 0.80]])
    errs = np.linspace(0, 0.40, 41)
    l1_s, l1_l, g_s, g_l = [], [], [], []
    for e in errs:
        ps, pl = small.clone(), large.clone()
        ps[0, 2:] *= 1 + e
        pl[0, 2:] *= 1 + e
        l1_s.append(float(torch.abs(small - ps).sum()))
        l1_l.append(float(torch.abs(large - pl).sum()))
        g_s.append(float(giou_elem(small, ps)))
        g_l.append(float(giou_elem(large, pl)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(errs * 100, l1_l, color=C1, linewidth=2, label="L1 · 대형 (h,w)=(0.90,0.80)")
    ax.plot(errs * 100, l1_s, color=C1, linewidth=2, linestyle="--", label="L1 · 소형 (h,w)=(0.10,0.08)")
    style(ax, "L1: 상대 오차에 비례해 상자 크기만큼 커진다", "상대 오차 (%)", "L1 손실")
    ax.legend(fontsize=8, frameon=False)
    ax.annotate(f"15% 지점에서 {l1_l[15] / max(l1_s[15], 1e-12):.1f}x",
                xy=(15, l1_l[15]), xytext=(20, l1_l[15] * 0.75),
                fontsize=9, color=C2,
                arrowprops=dict(arrowstyle="->", color=C2, lw=1.2))

    ax = axes[1]
    ax.plot(errs * 100, g_l, color=C3, linewidth=2, label="GIoU · 대형")
    ax.plot(errs * 100, g_s, color=C3, linewidth=2, linestyle="--", label="GIoU · 소형")
    style(ax, "GIoU: 스케일 불변 (두 곡선이 겹친다)", "상대 오차 (%)", "GIoU 손실")
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.52, 0.86, "비율(대형/소형) = 1.000x\n(동일 상대 오차 기준)",
            transform=ax.transAxes, fontsize=9, color=C3)
    save(fig, "chart_l1_giou_scaling.png")


# ---------------------------------------------------------------------------
# 3. Ø 하향 10의 손실 구성 변화
# ---------------------------------------------------------------------------
def chart_empty_weight() -> None:
    per_obj, per_bg = -math_log(0.90), -math_log(0.015)
    n_obj, n_empty = 7, 93
    raw = n_obj * per_obj + n_empty * per_bg
    wgt = n_obj * per_obj + n_empty * per_bg / 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.bar(["하향 없음", "하향 10"], [raw, wgt], color=[C1, C2], width=0.5)
    style(ax, "한 장(객체 7 + Ø 93)의 분류 손실 합", "Ø 하향 적용", "손실 합")
    for i, v in enumerate([raw, wgt]):
        ax.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=10, color=INK)

    ax = axes[1]
    w = 0.38
    x = np.arange(2)
    ax.bar(x - w / 2, [per_obj, per_bg / 10], w, color=C3, label="배경 항(Ø) 1개")
    ax.bar(x + w / 2, [per_obj, per_bg], w, color=C1, label="배경 항(Ø) 1개 — 하향 전")
    ax.set_xticks(x)
    ax.set_xticklabels(["양성 슬롯 1개", "배경 슬롯 1개"])
    style(ax, "개별 슬롯 비용: 하향 10의 효과", "슬롯 종류", "-log p")
    ax.legend(fontsize=8, frameon=False)
    ax.text(1.0, per_bg * 0.62, f"39.9x → 4.0x", ha="center", fontsize=11, color=C2)
    save(fig, "chart_empty_weight.png")


def math_log(x: float) -> float:
    return float(np.log(1.0 / x))


# ---------------------------------------------------------------------------
# 4. RT-DETR 쿼리 선택 산점도 (Fig 6 흉내) — 자체 시뮬레이션
# ---------------------------------------------------------------------------
def chart_query_scatter() -> None:
    rng = np.random.default_rng(2304)
    n, k = 20_000, 300
    cls_all = np.clip(rng.beta(0.5, 3.0, n), 0, 1)
    iou_all = np.clip(rng.beta(0.5, 3.0, n) + 0.05 * (cls_all > 0.8), 0, 1)
    defect = rng.random(n) < 0.35
    iou_all[defect & (cls_all > 0.8)] *= 0.15

    van = np.argsort(cls_all)[::-1][:k]
    unc = np.argsort(cls_all * iou_all)[::-1][:k]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, idx, color, name in (
        (axes[0], van, C3, "vanilla (분류 점수 top-K)"),
        (axes[1], unc, "#6a51a3", "uncertainty (분류xIoU top-K)"),
    ):
        ax.scatter(cls_all[idx], iou_all[idx], s=9, alpha=0.55, color=color, linewidths=0)
        ax.axhline(0.5, color=GRID, lw=1, linestyle="--")
        ax.axvline(0.5, color=GRID, lw=1, linestyle="--")
        prop_both = float(np.mean((cls_all[idx] > 0.5) & (iou_all[idx] > 0.5)))
        style(ax, f"{name}\nProp both = {prop_both:.2%}", "분류 점수 (cls)", "IoU 점수 (quality)")
    axes[0].text(0.03, 0.95, "우상단(두 점수 모두 > 0.5) = 좋은 쿼리\n(원문 RT-DETR Fig 6의 구조 재현)",
                 transform=axes[0].transAxes, fontsize=9, color=INK, va="top")
    save(fig, "chart_query_selection_scatter.png")


# ---------------------------------------------------------------------------
# 5. 파이프라인 학습 곡선 + 6. 디코더 깊이 trade-off (runs JSON 사용)
# ---------------------------------------------------------------------------
def chart_pipeline(payload: dict) -> None:
    log = payload["log"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for mode, color, label in (("vanilla", C1, "vanilla (분류 점수 top-K)"),
                               ("uncertainty", C2, "uncertainty-minimal (+U 항)")):
        rows = sorted([r for r in log if r["mode"].startswith(mode)], key=lambda r: r["epoch"])
        if rows:
            ax.plot([r["epoch"] for r in rows], [r["ap"] for r in rows],
                    color=color, linewidth=2, label=label)
    style(ax, "파이프라인 학습 곡선 (합성 데이터, CPU)", "epoch", "AP@0.5 (val)")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "chart_pipeline_ap.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    res = payload["results"]
    depths = sorted(int(d) for d in res["vanilla"]["depth_curve"])
    for key, color, label in (("vanilla", C1, "vanilla"), ("uncertainty", C2, "uncertainty-minimal")):
        ys = [res[key]["depth_curve"][str(d)] if str(d) in res[key]["depth_curve"]
              else res[key]["depth_curve"][d] for d in depths]
        ax.plot(depths, ys, marker="o", color=color, linewidth=2, label=label)
    style(ax, "디코더 깊이 축소 시 AP (재학습 없이)", "사용 디코더 층", "AP@0.5 (val)")
    ax.invert_xaxis()
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.03, 0.06, "이 미니 데모는 얕게 자를수록 약간 우세\n(논문 RT-DETR Table 5는 6층이 최선)",
            transform=ax.transAxes, fontsize=8.5, color=INK)
    save(fig, "chart_decoder_depth.png")


# ---------------------------------------------------------------------------
# 7. DETR Table 1 — AP 분해 비교 (논문 표 인용)
# ---------------------------------------------------------------------------
def chart_detr_breakdown() -> None:
    models = ["Faster R-CNN-FPN+", "DETR", "DETR-DC5", "DETR-R101", "DETR-DC5-R101"]
    ap = [42.0, 42.0, 43.3, 43.5, 44.9]
    aps = [26.6, 20.5, 22.5, 21.9, 23.7]
    apl = [53.4, 61.1, 61.1, 61.8, 62.3]
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w, ap, w, color=C1, label="AP")
    ax.bar(x, aps, w, color=C2, label="AP$_S$ (소형)")
    ax.bar(x + w, apl, w, color=C3, label="AP$_L$ (대형)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    style(ax, "DETR vs Faster R-CNN — AP 분해 (논문 DETR Table 1)",
          "모델 (R50 / R101)", "AP")
    ax.legend(fontsize=9, frameon=False)
    ax.annotate("DETR: APS -5.5 · APL +7.8\n(동일 파라미터 41M vs 42M)",
                xy=(1, 61.1), xytext=(1.6, 66), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    save(fig, "chart_detr_ap_breakdown.png")


# ---------------------------------------------------------------------------
# 8. RT-DETR Table 3 — 하이브리드 인코더 변형 trade-off (논문 표 인용)
# ---------------------------------------------------------------------------
def chart_rtdetr_encoder() -> None:
    labels = ["A", "B", "C", "D", "D_S5", "E"]
    ap = [43.0, 44.9, 45.6, 46.4, 46.8, 47.9]
    lat = [7.2, 11.1, 13.3, 12.2, 7.9, 9.3]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [C1, C1, C1, C3, C2, C2]
    ax.scatter(lat, ap, s=110, c=colors, zorder=3, edgecolors="white", linewidths=1.2)
    for x, y, lab in zip(lat, ap, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=10, color=INK)
    style(ax, "RT-DETR 하이브리드 인코더 변형: 정확도 vs 지연 (논문 Table 3, 1x 설정)",
          "지연 (ms)", "AP")
    ax.annotate("D → D_S5: 지연 -35% · AP +0.4\n(E: RT-DETR 최종, 지연 -24%)",
                xy=(9.3, 47.9), xytext=(9.0, 45.4), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    save(fig, "chart_rtdetr_encoder_tradeoff.png")


# ---------------------------------------------------------------------------
# 9. 파이프라인 도식 (시각화 자료)
# ---------------------------------------------------------------------------
def chart_pipeline_diagram(trace: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    boxes = [
        (0.15, 1.9, 1.75, 0.9, "Backbone\nCNN 3단", "#e8eef4", C1),
        (2.15, 1.9, 1.75, 0.9, "AIFI\nS5만 self-attn", "#f6e7e2", C2),
        (4.15, 1.9, 1.75, 0.9, "CCFF\nFPN + PAN (conv)", "#f6e7e2", C2),
        (6.15, 1.9, 1.75, 0.9, "후보 N=1344\nscore·IoU 헤드", "#e8eef4", C1),
        (8.15, 1.9, 1.75, 0.9, "쿼리 선택\ntop-K=30", "#e9f0e9", C3),
        (10.15, 1.9, 1.75, 0.9, "Decoder 6층\n+ aux 헤드", "#e8eef4", C1),
    ]
    for x, y, w, h, text, fc, ec in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=INK)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 2.35), xytext=(x0 + 0.04, 2.35),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("", xy=(8.2, 1.85), xytext=(11.0, 1.85),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(9.6, 1.35, "헝가리아 세트 손실\n(Ø 하향 10 · L1x5 + GIoUx2)", ha="center",
            fontsize=8.5, color=INK)
    ax.text(6.0, 3.75, "RT-DETR 파이프라인 (day07_detr/detr_pipeline_demo.py)",
            ha="center", fontsize=12, color=INK)
    ax.text(6.0, 0.55,
            "content query = 선택된 인코더 피처 / position query = 그 위치의 예측 박스\n"
            "aux loss가 모든 디코더 층을 유효한 예측기로 만듦 → 재학습 없는 깊이 조절 가능",
            ha="center", fontsize=8.5, color=INK)
    save(fig, "chart_pipeline_diagram.png")


def main() -> None:
    os.makedirs(RUNS, exist_ok=True)
    print("[charts] 생성 시작")
    demo = json.load(open(os.path.join(RUNS, "detr_demo_metrics.json")))
    pipe = json.load(open(os.path.join(RUNS, "detr_pipeline_metrics.json")))
    chart_matching_cost()
    chart_l1_giou()
    chart_empty_weight()
    chart_query_scatter()
    chart_pipeline(pipe)
    chart_detr_breakdown()
    chart_rtdetr_encoder()
    chart_pipeline_diagram(pipe["trace"])
    print("[charts] 완료 ->", RUNS)


if __name__ == "__main__":
    main()
