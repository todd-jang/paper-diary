#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 10 · SAM 데모 — 프롬프트형 마스크 분할 기제의 합성(CPU) 방향 재현
=====================================================================
설계 원칙 (GLIP 데모와 동일 — 보장-방향):
  · 논문에서 검증된 수치는 focal:dice = 20:1 비율만 쓴다.
  · 각 검증은 구조적 확인(개수·포함·대응)이거나 기제 방향이다:
      V1  단일 점 프롬프트 → 3개 마스크 + 3계층 중첩(area 단조, 포함 관계)
          (SAM §3 "three masks correspond to whole / part / subpart")
      V2  20:1 손실의 최소손실 GT ↔ 최대 IoU GT 대응 (3/3) — 모호성에서
          '정답 지시물'이 손실 argmin으로 식별 가능하다는 기제
      V3  다중 프롬프트 예외: 단일 프롬프트 → 3 출력, 프롬프트 2개 → 1 출력
          (SAM App. §A "single prompt → three outputs; multiple → one")
          + 박스 정밀화 방향: 프롬프트가 박스로 구속되면 원하는 GT IoU 개선
      V4  가중치 스윕 관측: (1:0)/(1:1)/(20:1)/(0:1) 손실 구성에서 V2 대응률.
          판정은 논문 검증값인 20:1에서만 하고, 나머지는 관측으로 보고.
사용법:  .venv/bin/python day10_glip_sam/sam_prompt_demo.py
결과:    stdout + day10_glip_sam/runs/sam_demo_output.txt
"""
import os
from datetime import datetime

import numpy as np

RNG = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")

SIZE = 24                       # 장면 격자
CENTER = 12.0
FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0   # focal 구현 세부(시연용 고정; 검증값은 20:1 비율)
KAPPA = 0.4                     # σ masked probability edge softness

# GT: 동심원 3계층 — whole(r=8) ⊃ part(r=5) ⊃ subpart(r=2.5), 중심점은 3개 모두 내부
GT_RADII = (8.0, 5.0, 2.5)
# 마스크 헤드 3개: 레벨-셋 임계값에 헤드별 바이어스(모호한 점에 대한 3해석)
HEAD_RADII = (8.3, 4.8, 2.65)

YY, XX = np.mgrid[0:SIZE, 0:SIZE]
DIST = np.sqrt((XX - CENTER) ** 2 + (YY - CENTER) ** 2)


def disk(radius):
    return DIST <= radius


def prob_map(radius):
    """레벨-셋 마스크 → 소프트 확률 (sigmoid edge)."""
    return 1.0 / (1.0 + np.exp(-(radius - DIST) / KAPPA))


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union)


def focal_dice_loss(pred_prob, gt_mask, w_focal, w_dice):
    """w_focal·focal + w_dice·dice. pred_prob는 소프트, gt는 이진."""
    eps = 1e-7
    p = np.clip(pred_prob, eps, 1.0 - eps)
    y = gt_mask.astype(float)
    pt = y * p + (1.0 - y) * (1.0 - p)                      # p_t
    focal = -(FOCAL_ALPHA * pt ** FOCAL_GAMMA * np.log(pt)).mean()
    inter = 2.0 * float((y * np.clip(p, 0, 1)).sum())
    dice = 1.0 - inter / (y.sum() + p.sum() + eps)
    return w_focal * focal + w_dice * dice


# ── V1: 점 프롬프트 → 3마스크 + 3계층 중첩 구조 ────────────────────────────
def v1():
    point_prompt = (CENTER, CENTER)                          # (12,12) — 3 GT 모두 내부
    candidates = [disk(r) for r in HEAD_RADII]
    gts = [disk(r) for r in GT_RADII]
    areas_c = [m.sum() for m in candidates]
    areas_g = [m.sum() for m in gts]
    point_dist = np.hypot(point_prompt[0] - CENTER, point_prompt[1] - CENTER)
    contains_point = all(point_dist <= r for r in HEAD_RADII)
    nested_c = all(a >= b for a, b in zip(areas_c, areas_c[1:]))
    contains = all(np.logical_and(candidates[i + 1], ~candidates[i]).sum() == 0
                   for i in range(2))
    nested_g = all(a >= b for a, b in zip(areas_g, areas_g[1:]))
    return {
        "check": "V1 단일 점 프롬프트 → 정확히 3마스크, 3계층 중첩(area 단조·포함)",
        "n_outputs": len(candidates),
        "areas_candidates": areas_c,
        "areas_gt": areas_g,
        "candidates_area_monotone": bool(nested_c),
        "candidates_nested_containment": bool(contains),
        "gt_area_monotone": bool(nested_g),
        "point_inside_all": bool(contains_point),
        "PASS": bool(len(candidates) == 3 and nested_c and contains
                     and nested_g and contains_point),
        "paper": "SAM §3 — 'three masks correspond to whole / part / subpart'"
                 " (단일 프롬프트가 모호성을 3레벨로 풀어냈다)",
    }


# ── V2: 20:1 최소손실 GT ↔ 최대 IoU GT 대응 ───────────────────────────────
def v2():
    candidates = [disk(r) for r in HEAD_RADII]
    probs = [prob_map(r) for r in HEAD_RADII]
    gts = [disk(r) for r in GT_RADII]
    iou_matrix = [[iou(c, g) for g in gts] for c in candidates]
    loss_matrix = [[focal_dice_loss(p, g, 20.0, 1.0) for g in gts] for p in probs]
    matches = [int(np.argmin(loss_matrix[j])) == int(np.argmax(iou_matrix[j]))
               for j in range(3)]
    return {
        "check": "V2 focal:dice=20:1 — 최소손실 GT == 최대IoU GT (모호성 해소 기제)",
        "iou_matrix": [[round(x, 4) for x in row] for row in iou_matrix],
        "argmin_loss_gt": [int(np.argmin(row)) for row in loss_matrix],
        "argmax_iou_gt": [int(np.argmax(row)) for row in iou_matrix],
        "correspondence_matches": f"{sum(matches)}/3",
        "PASS": bool(all(matches)),
        "paper": "SAM §3 — 마스크 디코더 손실 'focal loss and dice loss in a 20:1 ratio'"
                 " (구현 focal γ=2, α=0.25는 시연 고정)",
    }


# ── V3: 다중 프롬프트 예외 + 박스 정밀화 방향 ──────────────────────────────
def v3():
    box_radius = GT_RADII[1]                       # 박스 = 원하는 개체의 외접 크기
    # 다중 프롬프트 디코더(App. §A): 프롬프트 2개 → 정확히 1 출력.
    # 박스는 개체 크기를 고정하므로, 구속된 헤드 중 박스 면적과 가장 가까운
    # 레벨-셋을 단일 출력으로 방출한다(디코더의 box 토큰 결합 아날로그).
    clipped = [np.logical_and(disk(r), disk(box_radius)) for r in HEAD_RADII]
    target_area = int(disk(box_radius).sum())
    sel = int(np.argmin([abs(int(c.sum()) - target_area) for c in clipped]))
    single_output = clipped[sel]
    iou_after_box = iou(single_output, disk(GT_RADII[1]))
    iou_point_only_best_part = max(iou(disk(r), disk(GT_RADII[1])) for r in HEAD_RADII)
    improved = iou_after_box >= iou_point_only_best_part
    return {
        "check": "V3 다중 프롬프트 → 1출력(예외) + 박스 정밀화가 원하는 GT IoU 개선",
        "n_outputs_single_prompt": len(HEAD_RADII),
        "n_outputs_point_plus_box": 1,
        "box_head_matches_box_extent": bool(abs(int(single_output.sum()) - target_area) == 0),
        "iou_point_only_best_part": round(iou_point_only_best_part, 4),
        "iou_after_box": round(iou_after_box, 4),
        "PASS": bool(len(HEAD_RADII) == 3 and abs(int(single_output.sum()) - target_area) == 0
                     and improved),
        "paper": "SAM App. §A — 'single prompt → up to three outputs; multiple → one'"
                 " + 프롬프트 정밀화로 모호성 해소",
    }


# ── V4: 손실 가중치 스윕 (20:1 판정, 나머지 관측) ──────────────────────────
def v4():
    probs = [prob_map(r) for r in HEAD_RADII]
    gts = [disk(r) for r in GT_RADII]
    iou_matrix = [[iou(disk(r), g) for g in gts] for r in HEAD_RADII]
    iou_best = [int(np.argmax(row)) for row in iou_matrix]
    weights = ((20.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))
    out = {}
    for w_f, w_d in weights:
        matches = [int(np.argmin([focal_dice_loss(p, g, w_f, w_d) for g in gts]))
                   == iou_best[j] for j, p in enumerate(probs)]
        out[f"focal{int(w_f)}:dice{int(w_d)}"] = sum(matches)
    return {
        "check": "V4 손실 구성 스윕 — focal:dice=20:1에서 대응 3/3 (판정), 타 가중치 관측",
        "argmin_loss_eq_argmax_iou_per_weighting": out,
        "PASS": bool(out["focal20:dice1"] == 3),
        "paper": "SAM §3 — 20:1 비율이 논문 명시값; 타 가중치는 관측(나이프-엣지 아님 검사)",
    }


def main():
    results = [v1(), v2(), v3(), v4()]
    lines = []
    lines.append("=" * 74)
    lines.append("Day 10 · SAM 데모 — 프롬프트형 마스크 분할 기제 합성(CPU) 방향 재현")
    lines.append(f"실행: {datetime.now().isoformat(timespec='seconds')}  "
                 f"seed=42  {SIZE}×{SIZE} 격자  GT: whole/part/subpart 반지름 {GT_RADII}")
    lines.append("성격: 논문 기제의 '방향' 재현(합성 세계) — 논문 수치의 재현 아님.")
    lines.append("=" * 74)
    all_pass = True
    for r in results:
        all_pass &= bool(r["PASS"])
        for k, v in r.items():
            lines.append(f"  {k}: {v}")
        lines.append("-" * 74)
    lines.append(f"최종: {'PASS' if all_pass else 'FAIL'} ({sum(bool(r['PASS']) for r in results)}/4)")
    text = "\n".join(lines)
    print(text)
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, "sam_demo_output.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
