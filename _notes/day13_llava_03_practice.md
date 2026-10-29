---
layout: default
title: "Day 13 LLaVA — 03 · 실습 — `llava_demo.py` 4검증 (CPU 실측)"
paper_id: llava
date: 2026-10-26
---

# 03 · 실습 — `llava_demo.py` 4검증 (CPU 실측)

> 스크립트: `day13_llava/llava_demo.py` · 로그: `runs/llava_demo_output.txt`
> **환경:** 합성 세계, numpy 전용, CPU, 시드 42, **학습 있음(해당 검증만 — masked GD / 최소제곱)** — LLaVA(2304.08485)의 핵심 기제를 **구조·방향** 측면에서 재현한다.
> **차트 4종**(`runs/figures/`, `make_charts.py`로 재생성; 표 수치는 `00_검증팩트시트.md`에서만 취함):
> `01_llava_training_stages.png` · `02_llava_table4_data_mix.png` · `03_llava_table5_in_the_wild.png` · `04_llava_scienceqa.png`

## 0. 실행 방법

```bash
cd /Users/robotjang/paper-diary
.venv/bin/python day13_llava/llava_demo.py          # 즉시 (합성 matmul + 마스크 GD)
.venv/bin/python day13_llava/make_charts.py         # 차트 4종 재생성
```

- **이 데모가 하는 일**: 논문의 *수치* 재현이 아니라, 설계 판단의 **구조적 근거**(선형 projection의 정합, 3종 지시 데이터의 혼입 방향, 답변-토큰 loss 마스크의 관문성, judge 앙상블의 구조)를 **자체 실측**으로 확인한다. 아래 수치는 논문 수치와 직접 비교 대상이 아니다. 비교 가능한 것은 **방향과 구조**다.
- **결정론성**: numpy `default_rng(42)` 전용 — 같은 numpy 버전이면 **비트 동일 재현**. 전역 상수(RNG 소비 순서)를 바꾸면 값이 바뀐다.
- **판정 설계 원칙**: 임계값 넘기용 파라미터 탐색 금지 — 판정 근거는 (i) 항등식/정리(V1 복원·V2 적산·V3 마스크·V4 분해), (ii) 구조적 방향(V1 heldout·V2 순서·V3 grounding·V4 앙상블>단독), (iii) 통계 보장(V4 MC 근사)이다. 측정값은 추세 자료로만 기록.

---

## 1. V1 — 식(1) 선형 projection의 정합: σ=0 완전 정합 + 잔차 직교성 + 모듈 동결 (§4.1·§4.2)

**논문 명제**: `H_v = W·Z_v` — 단일 선형 W가 CLIP 그리드 피처를 Vicuna 워드 임베딩 차원으로 투영한다. Stage 1은 비전·LLM **전부 고정**, 학습 파라미터는 **W만**이고, 논문 표현은 "training a compatible visual tokenizer for the frozen LLM"이다.

**방법**: 세계 = 개념 128개(언어 공간 단위 방향 E) · CLIP 공간은 가역 정사각 선형상 A를 거친 피처(Z = E·A + σ·N, σ=0.05) · 학습 112(heldout 16) — **과소결정(암기)을 방지하는 위계**(학습 수 > 피처 차원). 최소제곱으로 W만 fit한다(비전·LLM 미접촉). 확인: (a) σ=0 세계에서 fit한 W가 **전 개념(학습+heldout)** 을 잃는 것 없이 복원하는지(E·A가 가역 → W₀ = A⁻¹가 정확해 → 잔차 0 항등식), (b) 노이즈 세계의 최소제곱 잔차가 설계 공간에 직교(정규방정식 — 열 단위 스케일 정규화 최대값 < 1e-9), (c) fitting 전후 CLIP 피처·개념 방향 배열 비트 동일(θ={W} 구현 감사), (d) heldout 정렬 코사인 > 랜덤 probe(방향 측정).

**실측**
```
noiseless_full_reconstruction_max_abs: 0.0      # σ=0 세계: W₀=A⁻¹ → 전 개념 완전 복원(항등식)
residual_orthogonality_scaled_max: 0.0          # 노이즈 세계 잔차의 열 직교성(정리 수준 0)
modules_bit_identical_after_fit: True           # encoder/vocab 동결, θ={W}만 갱신
heldout_mean_cos(fit/probe): [0.668, 0.0476]    # heldout 16개: 정합 > 랜덤 probe(방향)
```

**읽는 법** — (a)은 '정리'로 통과: CLIP 피처 공간이 언어 공간의 **가역 선형상**이면 단일 선형 W가 잃는 것이 없다 — Q-former·gated cross-attention이 없어도 결합 자체는 로스가 없다는 구조적 근거. (b)는 Stage 1 = 최소제곱 정합의 수학적 동의어. (d)의 heldout 코사인 0.668 vs probe 0.048 — 노이즈 세계에서도 **학습한 개념이 아닌 개념**까지 방향이 전이된다(정합이 '암기'가 아님). 한 걸음 주의: v1 논문의 실 W는 576토큰×4096차원 등 스케일이 다르고, heldout 정합을 **보장하는 게 아니라 가능하게 하는** 구조다.

---

## 2. V2 — Table 4 데이터 혼입: 적산 항등식 + All 순서 방향 + 공동 확장 (§3·Table 4)

**논문 명제**: 3종 지시 데이터(conversation 58K / detailed description 23K / complex reasoning 77K)의 혼입이 능력을 만든다 — Table 4 All 상대점수: Full **85.1** > Detail+Complex **81.9** > Conv+5%D+10%C **80.5** > Conv-only **73.8** ≫ No-tuning **21.5**.

**방법**: 능력 축 3개 × 데이터 유형 3종의 합성 학습 모형. 각 예시는 자기 축에 자기 기여(감쇠 포화 `sat(c)=1−e^{−c/τ}`, τ=30K)를 주고 타 축에 작은 전이 κ=0.08을 준다(비대칭 전이 행렬 TRAN). 포화는 "같은 종류의 데이터를 계속 넣으면 제자리"를 합성하는 장치 — 혼입의 이득은 **전이 항**에서 나와야 Table 4의 순서가 재현된다. 확인: (a) 루프 적산 경로와 폐형 행렬 경로의 적산 항등식, (b) All 평균의 행 순서 == Table 4 순서, (c) 소량 혼입(5% detail + 10% complex)이 Conv 축을 대화 전용보다 끌어올리는 공동 확장, (d) 모든 튜닝 행 > 무튜닝.

**실측**
```
scores(conv/detail/complex):
  Full (58/23/77)  [25.972,  25.6777, 26.0345]    # 3종 전부
  Detail+Complex   [25.1167, 25.6093, 25.966 ]    # 대화 제거
  Conv+5%D+10%C    [25.8765, 25.1241, 25.2978]
  Conv only        [25.8553, 25.0684, 25.0684]
  No tuning        [25.0,    25.0,    25.0   ]
all_col_relative_mean: Full 25.8947 > DC 25.564 > Mixed 25.4328 > Conv 25.3307 > NoT 25.0
all_col_match_paper_order: True     # Table 4 All 순서 정확히 재현
small_mix_co_boosts_conv: True      # 혼입 25.8765 > 대화 전용 25.8553 (Conv 축)
full_best_on_every_axis: True       # 전 축에서 Full 최고
loop_vs_closed_max_dev: 0.0         # 적산 항등식 — 0.0 정확 일치
```

**읽는 법** — 입력 개수(58/23/77K)만 주면 포화+전이 두 항으로 Table 4의 **행 순서 전체**가 재현된다. 특히 두 은밀한 관측의 구조적 재현: ① 대화 축에서 **Conv+5%D+10%C(25.8765) > Conv only(25.8553)** — reasoning·상세 데이터의 부가 충전이 자기 축이 아니라 '대화 능력'을 끌어올린다(논문 §5 명시적 관측 ②), ② **Detail+Complex(25.564)가 Conv+혼입(25.4328)보다 All 높음** — 대화 데이터를 통째로 걷어내도(58K 전멸) 상세·추론 혼입량을 다 채우면 All이 오히려 더 크다(Table 4에서도 81.9 > 80.5).

---

## 3. V3 — 식(2)~(3) loss 마스크: 항등식 + 기울기 관문 + grounding (Table 2·Eq.3)

**논문 명제**: 시퀀스 `system Human: instruct Assistant: answer` 중 **assistant 답변 토큰(녹색)만** loss에 들어가고, 식(3)은 모든 조건부에 `X_v`를 명시 — pθ(X_a | X_v, X_instruct) — 이미지가 grounding이다.

**방법**: 미니 언어 모델(mean-pooled prefix + 선형 head) 합성. 시퀀스 `[VIS·c, 지시×8, 답변×6]`(길이 15) — **동일 지시 템플릿 반복**으로 두어 개념 신호가 시각 토큰에만 있도록 고정(사람 seed in-context와 유사). 마스크 GD(600 스텝·lr 0.35)로 **답변 위치 6토큰만** 학습. 확인: (a) 마스크 항등식 — 전 위치 CE를 계산 뒤 마스크 가중합 == 답변 rows 직접 평균(두 경로), (b) 기울기 관문 — dW를 위치별로 분해하면 instruction 위치 행이 **정확히 0**이고 전체 dW == 답변 rows만의 dW(다른 코드 경로와 일치), (c) grounding 방향 — 시각 토큰을 예시·개념 간 동일 '중립 토큰'으로 바꾼 대조군과 비교해 **답변 첫 토큰(위치 9) CE**가 낮은지, (d) 답변 CE 세 구간 추세.

**실측**
```
mask_identity_exact: True                      # 두 경로 정확 일치
grad_instr_rows_exactly_zero: True             # instruction 위치 기울기 행 == 0 (정확히)
grad_equals_answer_rows_only: True             # 전체 dW == 답변 rows만의 dW (항등식)
token_budget: {loss_tokens_per_sample: 6, sequence_length: 15, answer_tokens: 6}
turn_order_invariance_exact(observed): True    # 식(2) [X_q,X_v]/[X_v,X_q] — mean-pool 조건부 CE 불변
answer_ce_windows(first/mid/last): [2.1605, 0.9499, 0.7111]   # 세 구간 비증가
answer_first_token_ce(visual/neutral/uniform): [0.7753, 0.9428, 2.7726]
final_ce_with_visual 0.6922  <  final_ce_neutral_visual 0.7006
```

**읽는 법** — (a)(b)는 마스크가 **연산 목록**을 재배치할 뿐 학습 신호의 총량을 바꾸지 않는다는 구조 증명(답변 토큰 6/15 = 40%만 학습 예산). grounding 게이트(위치 9 = 답변 첫 토큰)가 이 데모의 굳은 살: 시각 토큰이 개념을 실어 주는 세계(0.7753) vs 중립으로 통일된 세계(0.9428) vs 균등추론(2.7726 = ln 16) — **중립 세계는 위치 9의 근거가 균등 추론 밖에 없다**는 감쇠가 그대로 CE에 등장. `turn_order_invariance`는 식(2)의 첫 턴 무작위 배치가 mean-pool 조건부 하에서 loss 불변이라는 관측(합성 언어 모델의 특성 표기 — 실 transformer는 위치 인코딩으로 구분됨).

---

## 4. V4 — Table 7 judge 앙상블: 분해 항등식 + 호출률 축약 + 방향 (§5.2·Table 7)

**논문 명제**: ScienceQA에서 LLaVA 단독 **90.92**, LLaVA + GPT-4(judge) **92.53** new SoTA (GPT-4† text-only 82.69[저자 eval]) — "*first time that GPT-4 is used for model ensembling*". 텍스트 전용 심판이 이미지 문항까지 개선 — **이미지 없이 답되는 문제를 식별·교정**하기 때문.

**방법**: ScienceQA test 4241 규모 가정의 합성. 문항 클래스 2종(이미지 불필요 40% / 입력 요구 60%), 클래스별 정답률 A(LLaVA 흐름: 0.92/0.90 → 단독 ≈ 0.908) · B(GPT-4 text-only 흐름: 0.95/0.745 → 단독 ≈ 0.827) — **이미지 불필요 클래스에서 심판(B 계열)이 강한 비대칭 구조**. 심판 규칙: 두 모델 답이 다르면(정확히 한쪽만 오류) 심판 strength p_j=0.75로 고친다, 둘 다 틀리면 고칠 수 없음. [합성 규모] 파라미터라벨 — 방향 구성용. 확인: (a) 앙상블 정확도를 겹친 기반항 경로와 문항 분기 경로로 각각 계산 → 정확 일치, (b) 심판 호출률 P(disagree) = 1 − pA·pB의 열합 축약 항등식(가지 합 qA·pB + pA·qB + qA·qB와 동치) + Monte-Carlo 관측 일치, (c) 앙상블 > 단독 최고.

**실측**
```
ensemble_accuracy_formula / branch_path: 0.9253 / 0.9253 (항등식 exact)
single_A(llava-like) 0.908  ·  single_B(gpt4-text-like) 0.827
ensemble_beats_best_single: True       # 앙상블 − 단독 최고 = +0.0173
judge_call_rate(closed_1-pApB) 0.2481  ≈  judge_call_rate(mc) 0.2475
mc_vs_formula: mc 0.9254 vs formula 0.9253 (abs_dev 0.0001 < 5e-3)
```

**읽는 법** — 세계를 임의로 놓아도 **통일된 조건**이 재현된다: ① 전체 클래스에서 단독 최고(0.908)보다 앙상블(0.9253)이 위 — 심판의 추가 한 벌이 **단독 최고 위에 겹쳐 쌓이는** 구조, ② 심판 호출은 강제가 아니라 분기 비용 — 호출률 ≈ 24.8%로 명시보다 훨씬 적다, ③ 이미지 불필요 클래스(40%)가 심판 이득의 거의 전부를 운반 — '텍스트 전용 심판이 어디서 고치는가'(논문 관측)의 방향 재현. 경계: 실측값 0.9253은 심판 strength·클래스 배합의 합성 규모 결과이지 논문 수치 재현이 아니다 — 논문 수치와 직접 비교 금지(00_검증팩트시트.md의 셀로만 인용).

---

## 5. 차트 4종 — 팩트시트 수치 시각화 (`runs/figures/`)

| 파일 | 내용 | 출처(팩트시트 셀) |
|---|---|---|
| `01_llava_training_stages.png` | Stage 1(CC-595K · 1ep · lr 2e-3 · bs 128 · ≤4h) vs Stage 2(158K · 3ep · lr 2e-5 · bs 32 · ≤10h) — 데이터·lr·에폭·시간 | (d) |
| `02_llava_table4_data_mix.png` | Table 4 — 5행 × Conv/Detail/Complex/All (Full 85.1 강조 · No-tuning 21.5) | (e) |
| `03_llava_table5_in_the_wild.png` | Table 5 — OpenFlamingo 19.1 · BLIP-2 38.1 · LLaVA 67.3 (±std 포함) | (f) |
| `04_llava_scienceqa.png` | Table 7 — Human 88.40 · GPT-3.5 73.97 · GPT-4† 82.69 · LLaMA-Adapter 85.19 · MM-CoT 91.68 · LLaVA 90.92 · **+judge 92.53**(강조) | (g) |

---

## 6. 최종 판정

```
최종: PASS (4/4)  —  2026-10-28T16:26  seed=42
```

| 검증 | PASS 근거 유형 |
|---|---|
| V1 | 항등식(σ=0 완전 복원·가역 선형상 → W*=A⁻¹) + 정리(정규방정식 직교성) + 구현 감사(모듈 동결) + 방향(heldout >> probe) |
| V2 | 항등식(적산 2경로) + 방향(Table 4 All 순서 재현) + 구조(공동 확장·전 축 Full 우위) |
| V3 | 항등식(마스크 2경로·dW 분해) + 구조(기울기 관문·토큰 예산 6/15) + 방향(grounding 게이트·CE 추세) |
| V4 | 항등식(분해 2경로·호출률 축약) + 통계(MC |dev| 1e-4) + 방향(앙상블 > 단독 최고) |
