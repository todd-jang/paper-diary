#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 07 — DETR (arXiv:2005.12872) + RT-DETR (arXiv:2304.08069) 심화 실습
=======================================================================

4 검증 (원문 수치 대조 목표 명시):

[A] 헝가리아리안 vs 그리디 매칭
    ① 최적성(헝가리아리안 총비용 ≤ 그리디 총비용)
    ② 유일성(순진 그리디의 중복 할당 발생률), ③ 순열 불변성
    근거: 원문 §3.1 식 (1)–(2), "guarantees that each target element
    has a unique match"

[B] L1 스케일 의존 vs GIoU 스케일 불변
    동일 '상대' 오차의 소형/대형 박스에서 L1 비율 vs GIoU 비율(≈1)
    근거: 원문 §3.1 "the most commonly-used l1 loss will have different
    scales for small and large boxes even if their relative errors are
    similar" + Table 4 (class+L1 35.8 AP / class+GIoU 39.9 AP / full 40.6 AP)

[C] ∅(no object) 하향 10
    배경 슬롯의 손실 기여 변화를 클래스 분해로 확인
    근거: 원문 §3.1 "down-weight the log-probability term when c_i = ∅
    by a factor 10 to account for class imbalance" + §4 COCO 평균 7 objects

[D] RT-DETR uncertainty-minimal 쿼리 선택 (시뮬레이션)
    cls 점수 기반 top-K vs score×IoU 기반 top-K의
    'cls·IoU 둘 다 > 0.5' 피처 비율(Prop both)
    근거: RT-DETR §4.3 식 (2)–(3), Table 4 (vanilla 0.30% → 0.67%,
    AP 47.9 → 48.7, K=300)
"""
import json
import math
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(OUT_DIR, exist_ok=True)
LINE = "-" * 72


def header(title: str) -> None:
    print(f"\n{LINE}\n[{title}]\n{LINE}")


def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, h, w = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou_loss(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """원문 식 (10): 1 - (IoU - |B(b1,b2) \\ (b1∪b2)| / |B(b1,b2)|), min/max 기반."""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box1).unbind(-1)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box2).unbind(-1)
    ix1, iy1 = torch.maximum(ax1, bx1), torch.maximum(ay1, by1)
    ix2, iy2 = torch.minimum(ax2, bx2), torch.minimum(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-12)
    u1, u2 = torch.minimum(ax1, bx1), torch.minimum(ay1, by1)
    u3, u4 = torch.maximum(ax2, bx2), torch.maximum(ay2, by2)
    enclose = (u3 - u1) * (u4 - u2)
    return 1 - (iou - (enclose - union) / enclose.clamp(min=1e-12))


def optimal_cost(cost: np.ndarray) -> float:
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def naive_greedy_cost(cost: np.ndarray, heads: list) -> float:
    """순진 그리디(중복 허용, 중복이면 남은 것 중 다음 최선)의 총비용."""
    used = [False] * cost.shape[0]
    rows, cols = [], []
    for j, h in enumerate(heads):
        if not used[h]:
            rows.append(h)
            cols.append(j)
            used[h] = True
        else:
            rem = [i for i in range(cost.shape[0]) if not used[i]]
            pick = min(rem, key=lambda i: cost[i, j])
            rows.append(pick)
            cols.append(j)
            used[pick] = True
    return float(cost[rows, cols].sum())


# ---------------------------------------------------------------------------
# [A] 헝가리아리안 vs 그리디
# ---------------------------------------------------------------------------
def run_a() -> dict:
    header("A. 헝가리아리안 vs 그리디 매칭 (식 1-2)")
    rng = np.random.default_rng(7)
    n_trials, n_pred, n_gt = 500, 8, 5  # 100 슬롯 + ∅ 패딩의 축약 모형
    opt_wins = 0
    dup_greedy = 0
    norm_cost_sum = 0.0
    for _ in range(n_trials):
        # 시드 고정: numpy RNG로만 표본을 뽑는다(전역 torch RNG는 비결정적이라 사용 금지)
        probs = torch.softmax(torch.from_numpy(rng.normal(size=(n_pred, n_gt))), dim=1)
        # 매칭 비용: -p(c_target) — 원문은 로그 대신 확률을 쓴다(상한 [0,1])
        cost = (-probs).numpy()
        cost_opt = optimal_cost(cost)

        # 순진 그리디: GT 순서대로 자기 최선 예측을 가져감(중복 허용)
        heads = [int(np.argmin(cost[:, j])) for j in range(n_gt)]
        if len(set(heads)) < n_gt:
            dup_greedy += 1
        if cost_opt < naive_greedy_cost(cost, heads) - 1e-9:
            opt_wins += 1
        norm_cost_sum += -cost_opt / n_gt  # 평균 매칭 확률 p̄ ∈ [0,1]

    # 순열 불변성
    torch.manual_seed(7)
    probs = torch.softmax(torch.randn(n_pred, n_gt), dim=1)
    perm = torch.randperm(n_pred)
    loss_full = optimal_cost((-probs).numpy())
    loss_perm = optimal_cost((-probs[perm]).numpy())

    print(f"시행 수                      : {n_trials} (N_pred={n_pred}, N_gt={n_gt})")
    print(f"optimal < naive_greedy 비율  : {opt_wins / n_trials:.3f}  (greedy가 이기는 시행은 없음)")
    print(f"순진 그리디 중복시행 비율      : {dup_greedy / n_trials:.3f}  (예측 1개에 GT 2개가 몰려 1:N 할당)")
    print(f"평균 매칭 확률 p̄ = {norm_cost_sum / n_trials:.3f}  (균등 랜덤 모델 1/{n_gt} = {1 / n_gt:.2f} 대비)")
    print(f"순열 불변성                  : |L(원안) - L(순열)| = {abs(loss_full - loss_perm):.2e}")
    print("결론(식 1-2): one-to-one 매칭은 중복 예측의 학습 신호를 원천 차단하고")
    print("예측 슬롯 간 책임 분리를 만든다. 정답은 절차가 아니라 최적화된 총비용이다.")
    return {
        "n_trials": n_trials,
        "opt_beats_greedy_ratio": opt_wins / n_trials,
        "naive_greedy_dup_ratio": dup_greedy / n_trials,
        "matched_avg_prob": norm_cost_sum / n_trials,
        "perm_invariance_gap": abs(loss_full - loss_perm),
    }


# ---------------------------------------------------------------------------
# [B] L1 스케일 의존 vs GIoU 스케일 불변
# ---------------------------------------------------------------------------
def run_b() -> dict:
    header("B. L1 스케일 의존 vs GIoU 스케일 불변 (Table 4 구조)")
    base_small = torch.tensor([[0.24, 0.21, 0.10, 0.08]])  # APS 대역
    base_large = torch.tensor([[0.50, 0.55, 0.90, 0.80]])  # APL 대역
    rel_err = 0.15
    pred_small = base_small.clone()
    pred_small[0, 2:] *= 1 + rel_err
    pred_large = base_large.clone()
    pred_large[0, 2:] *= 1 + rel_err

    l1_small = float(torch.abs(base_small - pred_small).sum(-1))
    l1_large = float(torch.abs(base_large - pred_large).sum(-1))
    g_small = float(giou_loss(base_small, pred_small))
    g_large = float(giou_loss(base_large, pred_large))

    print(f"소형 박스 (h,w)=({base_small[0, 2]:.2f},{base_small[0, 3]:.2f}) · 15% 상대 오차:")
    print(f"  L1 = {l1_small:.2e}   GIoU = {g_small:.4f}")
    print(f"대형 박스 (h,w)=({base_large[0, 2]:.2f},{base_large[0, 3]:.2f}) · 같은 15% 상대 오차:")
    print(f"  L1 = {l1_large:.2e}   GIoU = {g_large:.4f}")
    print(f"비율(대형/소형)        : L1 {l1_large / l1_small:.2f}x   GIoU {g_large / g_small:.3f}x")
    print(f"합성 손실(L1 x5 + GIoU x2) 소형 : {5 * l1_small + 2 * g_small:.4f}")
    print(f"합성 손실(L1 x5 + GIoU x2) 대형 : {5 * l1_large + 2 * g_large:.4f}")
    print("결론: L1은 리터럴 스케일 잭이라 학습 신호가 대형 박스에 쏠리며,")
    print("스케일 간 균형은 GIoU(스케일 불변)가 담당한다. lambda_L1=5 / lambda_iou=2 (0.A.4)")
    return {"l1_ratio": l1_large / l1_small, "giou_ratio": g_large / g_small}


# ---------------------------------------------------------------------------
# [C] ∅ 하향 10
# ---------------------------------------------------------------------------
def run_c() -> dict:
    header("C. ∅ 하향 10 - 배경 슬롯 손실 기여 (예제 계산)")
    per_obj = -math.log(0.90)
    per_bg = -math.log(0.015)
    n_empty, n_obj = 93, 7  # 원문 §4: COCO 평균 7 objects/image
    raw = n_empty * per_bg + n_obj * per_obj
    wgt = n_empty * per_bg / 10 + n_obj * per_obj
    print(f"양성 슬롯 1개 (p_cat = 0.90)   : -log p = {per_obj:.3f}")
    print(f"배경 슬롯 1개 (p_empty = 0.015) : -log p = {per_bg:.3f}")
    print(f"배경 항이 양성 항보다 무거움    : {per_bg / per_obj:.1f}x")
    print(f"하향 10 적용 후                : {per_bg / 10 / per_obj:.1f}x")
    print(f"한 장(7객체 + 93 empty) 손실 합 : 하향 없이 {raw:.1f} -> 하향 10 {wgt:.1f}  ({raw / wgt:.1f}x 감소)")
    print(f"배경 항의 전체 비중            : 하향 없이 {n_empty * per_bg / raw:.1%} -> 하향 10 {n_empty * per_bg / 10 / wgt:.1%}")
    print("결론: 하향 10은 class imbalance 조절 장치로서 Faster R-CNN의 양·음")
    print("proposal 서브샘플링에 대응한다 (매칭 비용에서 ∅ 항은 상수라는 것도 기록).")
    return {"per_obj": per_obj, "per_bg": per_bg,
            "ratio_raw": per_bg / per_obj, "ratio_after": per_bg / 10 / per_obj}


# ---------------------------------------------------------------------------
# [D] RT-DETR uncertainty-minimal 쿼리 선택 (시뮬레이션)
# ---------------------------------------------------------------------------
def run_d() -> dict:
    header("D. RT-DETR 쿼리 선택 - cls-only vs score x IoU (Table 4 흉내)")
    rng = np.random.default_rng(2304)
    n_feats, top_k = 20_000, 300  # K=300 (원문 §5.3)
    cls_all = np.clip(rng.beta(0.5, 3.0, n_feats), 0, 1)
    iou_all = np.clip(rng.beta(0.5, 3.0, n_feats) + 0.05 * (cls_all > 0.8), 0, 1)
    defect = rng.random(n_feats) < 0.35  # '분류 높으나 위치 불신' 클러스터
    iou_all[defect & (cls_all > 0.8)] *= 0.15

    def prop_both(key):
        top = np.argsort(key)[::-1][:top_k]
        return float(np.mean((cls_all[top] > 0.5) & (iou_all[top] > 0.5)))

    van = prop_both(cls_all)
    unc = prop_both(cls_all * iou_all)
    print(f"top-K = {top_k} / 후보 feature = {n_feats:,}")
    print(f"Prop both>0.5 - vanilla(cls top-K)        : {van:.3%}")
    print(f"Prop both>0.5 - uncertainty(score x IoU)  : {unc:.3%}")
    print(f"증가배수 : {unc / max(van, 1e-9):.2f}x  (원문 Table 4: 0.30% -> 0.67%, AP 47.9 -> 48.7)")
    print("결론: 쿼리 초기화 품질은 분류와 위치가 '둘 다' 결정한다. cls만 보면")
    print("'confident하지만 위치 불량' 피처가 top-K에 들어온다. 이것이 식 (2)의 U = ||P-C||.")
    return {"vanilla_prop": van, "uncertainty_prop": unc, "gain": unc / max(van, 1e-9)}


def main() -> None:
    results = {}
    results["A_hungarian"] = run_a()
    results["B_loss_scale"] = run_b()
    results["C_empty_downweight"] = run_c()
    results["D_query_selection"] = run_d()
    path = os.path.join(OUT_DIR, "detr_demo_metrics.json")
    json.dump(results, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"\n{LINE}\n요약 JSON -> {path}")


if __name__ == "__main__":
    main()
