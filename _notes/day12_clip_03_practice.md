---
layout: default
title: "Day 12 CLIP — 03 · 실습 — `clip_demo.py` 4검증 (CPU 실측)"
paper_id: clip
date: 2026-10-23
---

# 03 · 실습 — `clip_demo.py` 4검증 (CPU 실측)

> 스크립트: `day12_clip/clip_demo.py` · 로그: `runs/clip_demo_output.txt`
> **환경:** 합성 세계, numpy 전용, CPU, 시드 42, **학습 없음** — CLIP(2103.00020)의 핵심 기제를 **구조·방향** 측면에서 재현한다.
> **차트 4종**(`runs/figures/`, `make_charts.py`로 재생성; 표 수치는 `00_검증팩트시트.md`에서만 취함):
> `01_clip_training_scale.png` · `02_clip_table1_vngrams.png` · `03_clip_dataset_deltas.png` · `04_clip_robustness.png`

## 0. 실행 방법

```bash
cd /Users/robotjang/paper-diary
.venv/bin/python day12_clip/clip_demo.py          # 즉시 (합성 matmul만)
.venv/bin/python day12_clip/make_charts.py        # 차트 4종 재생성
```

- **이 데모가 하는 일**: 논문의 *수치* 재현이 아니라, 설계 판단의 **구조적 근거**(대칭 CE의 2축 분해, τ 클립의 항등식, 임베딩 앙상블의 상각 비용, 프롬프트 평균의 1/K 분산 감소)를 **자체 실측**으로 확인한다. 아래 수치는 논문 수치와 직접 비교 대상이 아니다. 비교 가능한 것은 **방향과 구조**다.
- **결정론성**: numpy `default_rng(42)` 전용 — 같은 numpy 버전이면 **비트 동일 재현**. 전역 상수(RNG 소비 순서)를 바꾸면 값이 바뀐다.
- **판정 설계 원칙**: 임계값 넘기용 파라미터 탐색 금지 — 판정 근거는 (i) 항등식/정리(V1 분해·V2 클립·V3 σ=0 동치), (ii) 구조적 단조(V1 τ 비증가·V2 max prob·V3 agreement), (iii) 통계 보장 방향(V4 1/K)이다. 측정값은 추세 자료로만 기록.

---

## 1. V1 — 대칭 교차엔트로피: 분해 항등식 + 정렬 방향 + τ 비증가 (§2.2 + pseudocode)

**논문 명제**: 배치 N개 정렬 쌍에 대해 `logits = I_e·T_eᵀ·exp(t)`, `labels = arange(n)`, 손실은 두 축 CE의 평균 — `loss = (loss_i + loss_t)/2`.

**방법**: 세계를 "정렬 쌍 = 같은 클래스 프로토타입 + **같은 노이즈 방향 공유**"(두 인코더가 같은 잠재 방향으로 수렴한 상태의 합성)으로 고정 → 대각 코사인 = 1(정확히), 비대각 < 1 → 대각이 행·열 최대라는 구조적 전제가 성립. 이 전제에서 τ↑는 softmax를 대각으로 몰게 하므로 정렬 손실은 **τ에 비증가**(구조적 보장). 랜덤 배치(독립 임베딩)를 대조군으로, 같은 τ 그리드에서 측정.

**실측**
```
taus         = [0.5, 1.0, 2.0, 4.0, 7.0]
aligned_loss = [1.994971, 1.20369, 0.037012, 0.0, 0.0]
random_loss  = [3.481177, 3.518944, 3.92432, 15.560932, 307.334816]
identity_exact: True          # 동일 logits를 2코드 경로로 재계산 == 완전 일치
diagonal_is_row_max: True     # aligned 세계의 전제 확인
aligned_leq_random_all_taus: True
aligned_loss_nonincreasing_in_tau: True
```

**읽는 법** — 분해 항등식은 '정리'로 통과(측정 아님). 정렬 배치는 τ↑에서 대각(정답 쌍)으로 확률질량이 몰리며 0으로 수렴; 랜덤 배치는 τ↑에서 **지배 로짓(엉뚱 쌍)**으로 몰리며 폭발(307.3). **temperature는 구조를 증폭한다** — 대조 학습의 배치 설계(n 개 negative)가 "정답 쌍이 지배하게" 만들었을 때만 τ↑가 이득.

---

## 2. V2 — temperature 클립 ×100: 클립 항등식 + 랭킹 불변 + 신뢰도 추세 (§4)

**논문 명제**: τ는 log-parameterized 학습 파라미터, logit 스케일 `exp(τ)`는 **×100에서 클립**(§4 문장 + 부록 하이퍼파라 표 `Maximum temperature 100.0`).

**방법**: 하나의 세계(고정 임베딩)에서 (i) `τ=5.5`(exp≈244.7, 클립 구간)와 `τ=log(100)`의 확률이 정확히 같은지(항등식), (ii) 공통 양수 스케일 0.5→100에서 argmax 불변(랭킹 불변 — 양의 공통 배율은 순위를 바꿀 수 없음), (iii) 최대 확률의 스케일 비감소(측정 추세).

**실측**
```
tau_log_params = [5.5, 4.6052]
clip_identity_exact: True
ranking_invariant_to_common_scale: True
maxprob_by_scale = [0.048856, 0.075337, 0.544253, 0.999996, 1.0]   # 스케일 0.5,1,4,20,100
maxprob_monotone_nondecreasing: True
```

**읽는 법** — 클립 구간(`exp(τ)>100`)은 **정보가 없는 이동**: 어떤 τ든 스케일 100과 정확히 동치(항등식). 스케일이 커질수록 확률은 날나카로워지지만(0.0489→1.0) **순위는 불변** — τ는 분류 경계가 아니라 확률의 날카로움(학습 신호 강도)을 다룬다.

---

## 3. V3 — 임베딩 앙상블의 상각: σ=0 동치 + 비용 1:K + agreement 비증가 (§3.1.1)

**논문 명제**: "*cache a single set of averaged text embeddings so that the compute cost of the ensemble is the same as using a single classifier when amortized*".

**방법**: 클래스별 K=8 프롬프트 임베딩 = `a·W_c + σ·u_ck`. (i) σ=0에서 임베딩 평균 분류기가 단일(베이스) 분류기와 argmax 동치인지(항등식), (ii) 인코딩 횟수 구조(확률 앙상블 K회 vs 임베딩 앙상블 1회 캐시), (iii) σ>0에서 평균 임베딩 분류기와 베이스의 agreement가 σ에 비증가하는지(공통난수 구조).

**실측**
```
identity_at_sigma_zero: True
encoding_cost_structure: {'prob_ensemble_encodes_per_inference': 8,
                          'embedding_ensemble_encodes_per_inference': 1}
agreement_with_base(σ=0,0.2,0.5,1.0) = [1.0, 0.7812, 0.5, 0.2812]   # 비증가
```

**읽는 법** — σ=0 동치는 '정리'로 통과. **상각**의 실체: 앙상블 이득이 있어도 매 추론 비용은 1회 인코딩(K개 프롬프트는 선불 후 캐시). 구조적 단조는 "프롬프트를 어느 정도 다르게 섞어도 방향이 유지된다"는 논문 문장의 합성 재현.

---

## 4. V4 — 프롬프트 평균의 1/K 분산 감소 + 검색 우위 (§3.1.1 방향)

**논문 명제**: 80개 컨텍스트 프롬프트 앙상블 = ImageNet +3.5%(단일 기본 템플릿 대비). K개 프롬프트 임베딩의 평균은 노이즈 방향을 **K배 줄인다**(통계 보장). 본 데모는 K=8(계산량 축소, 방향 확인용).

**방법**: `P_ck = a·W_c + σ·u_ck`(동일 노이즈 분산)에서 잔여 노이즈 비율 `‖mean_k u‖² / ‖u‖²`의 기대값 1/K=0.125 대조 + 정답 클래스 검색 정확도(mean vs 단일 프롬프트).

**실측**
```
resid_single 0.7098   resid_mean 0.085   ratio 0.1198   expected 1/K = 0.125
retrieval_acc_mean_prompt 0.7656  >  retrieval_acc_single_prompt 0.7031
```

**읽는 법** — 실측 비율 0.1198은 기대값 0.125와 통계 관측오차 내 일치(평균의 에너지 = 단일 항의 1/K). 검색 정확도도 평균 프롬프트가 우위 — 논문의 프롬프트 앙상블 이득(+1.3% → +3.5%)과 **동일 부호·동일 기제**(임베딩 공간 노이즈 감소).

---

## 5. 차트 4종 — 팩트시트 수치 시각화 (`runs/figures/`)

| 파일 | 내용 | 출처(팩트시트 셀) |
|---|---|---|
| `01_clip_training_scale.png` | WIT 400M pairs · 500k queries · ≤20k/query · 배치 32,768 · 32 epochs + 사전학습 비용(RN50x64 592×V100·18일 / 최대 ViT 256·12일 — GPU-days는 파생 계산 명시) | (a)(b)(d) |
| `02_clip_table1_vngrams.png` | 논문 Table 1 — aYahoo/ImageNet/SUN, Visual N-Grams 72.4/11.5/23.0 vs CLIP 98.4/76.2/58.5 | (e) `zeroshot_table` |
| `03_clip_dataset_deltas.png` | 단일 데이터셋 zero-shot 승패 — STL10 99.3(절대값) · Cars/Food101 +20p · K700 +14.5 · UCF101 +7.7 · Flowers102/FGVCAircraft −10p (RN50 피처 기준) | (e) 본문 |
| `04_clip_robustness.png` | 자연 분포 이동 — Zero-Shot CLIP vs NS EfficientNet-L2 vs FixResNeXt101-32x48d V2 (IN/V2/A/R/ObjectNet/Sketch) | (k) `table:robustness` |

---

## 6. 최종 판정

```
최종: PASS (4/4)  —  2026-10-23T09:53:14  seed=42
```

| 검증 | PASS 근거 유형 |
|---|---|
| V1 | 항등식(분해) + 구조적 전제(대각 최대) + 측정 추세(τ 비증가) |
| V2 | 항등식(클립) + 정리(랭킹 불변) + 측정 추세(maxprob) |
| V3 | 항등식(σ=0 동치) + 구조(비용 1:K) + 측정 추세(agreement 비증가) |
| V4 | 통계 보장(1/K) + 방향(검색 정확도) |
