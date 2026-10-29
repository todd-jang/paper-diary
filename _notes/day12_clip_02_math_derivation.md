---
layout: default
title: "Day 12 CLIP — Day 12 · 02 — 수식 유도: 대칭 교차엔트로피와 하이퍼네트워크 classifier"
paper_id: clip
date: 2026-10-23
---

# Day 12 · 02 — 수식 유도: 대칭 교차엔트로피와 하이퍼네트워크 classifier

> 모든 등호는 논문 표기 기준(figure pseudocode / §2.2 / §3.1.1 / §4), 검증된 표·캡션 셀만 인용. 데모 실측값은 `runs/clip_demo_output.txt`(seed 42)에서 그대로 옮기며, **논문 수치와 직접 비교 대상이 아니다** — 비교 가능한 것은 방향과 구조다.

## 1. pseudocode의 행렬 형태 — 배치 N개에서 N²개 쌍까지

배치에 정렬된 (image, text) 쌍 N=n개가 들어간다(pseudocode 주석 그대로):

```
I[n, h, w, c] - minibatch of aligned images
T[n, l]       - minibatch of aligned texts
```

두 인코더 뒤 **선형 projection + L2 정규화**만 붙는다:

```
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)   # [n, d_e]
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)   # [n, d_e]
```

- `W_i[d_i, d_e]`, `W_t[d_t, d_e]`만 배우는 projection — 본문 §2.2: 비선형 projection 제거.
- 정규화된 뒤 내적 = **코사인 유사도** 행렬 `A ∈ [-1,1]^{n×n}`:

```
logits = np.dot(I_e, T_e.T) * np.exp(t)        # [n, n]
labels = np.arange(n)
```

`t`는 **학습되는 temperature**(log-parameterized); `exp(t)`는 logit 스케일.

## 2. 대칭 교차엔트로피 — 축 두 개, 평균 하나

```
loss_i = cross_entropy_loss(logits, labels, axis=0)   # 각 텍스트 열 j에 대해
                                                       # 이미지 축 softmax → 정답 이미지 j
loss_t = cross_entropy_loss(logits, labels, axis=1)   # 각 이미지 행 i에 대해
                                                       # 텍스트 축 softmax → 정답 텍스트 i
loss   = (loss_i + loss_t)/2
```

- **axis=0**: softmax가 행(이미지) 축 위로 흐른다 → 텍스트를 기준으로 맞는 이미지를 고르는 CE.
- **axis=1**: softmax가 열(텍스트) 축 위로 흐른다 → 이미지를 기준으로 맞는 텍스트를 고르는 CE.
- 같은 행렬 `logits`를 두 방향에서 다 쓰므로 *"*텍스트가 이미지를 맞추는 검색"*과 *"*이미지가 텍스트를 맞추는 검색"*이 **동일 파라미터에서 동시에 최적화**된다. 두 인코더 모두 업데이트되는 이유.
- 위상: multi-class N-pair loss (Sohn 2016) → InfoNCE (van den Oord 2018) → ConVIRT (Zhang 2020) 계승, **대칭화**가 CLIP의 추가물(§2.2).
- softmax의 분모 = 배치 속 나머지 n−1개 한국쌍 = **배치가 곧 negative 수** — 배치 32,768이 힘인 이유(§4).

### 2.1 V1 실측 — τ에 따른 정렬/랜덤 배치 (합성 세계, D=64, n=32, seed 42)

```
taus         = [0.5, 1.0, 2.0, 4.0, 7.0]
aligned_loss = [1.994971, 1.20369, 0.037012, 0.0, 0.0]      # 정렬 배치: τ↑ → 단조 감소 → 0
random_loss  = [3.481177, 3.518944, 3.92432, 15.560932, 307.334816]
identity_exact: True  (같은 logits를 두 코드 경로로 재계산 == 완전 일치)
diagonal_is_row_max: True  (aligned 세계의 구조적 전제)
```

- **정렬 배치**: 대각(정답 쌍)이 행·열 최대 → τ↑가 정답 쪽으로 날카로워짐 → loss 단조 감소 → 0.
- **랜덤 배치**: τ↑가 확률질량을 *지배적인* 로짓(와장닫은 엉뚱 쌍)으로 밀어송함 → loss 증가(307.3).
- 읽는 법: **temperature는 구조를 증폭한다** — 정렬이 되면 신호, 안 되면 오류. 대조 학습은 "정렬되어 있을 때 τ↑가 뭘 하는지"만 보증한다(데모가 전제를 명시한 이유).

## 3. temperature — 학습 파라미터, ×100 클립 (§4 + 부록 하이퍼파라 표)

- `t`는 로그 스케일 파라미터(`logits = cos · exp(t)`), 초기값 Wu 2018의 **0.07 환산치**.
- 학습 불안정 방지 위해 **logit 스케일을 ×100으로 클립** — 부록 하이퍼파라미터 표 `Maximum temperature 100.0`.
- 논리적 결과 1 (클립 항등식): `exp(t) > 100`인 어떤 t에서도 스코어는 `cos · 100`과 **정확히 동일**.
- 논리적 결과 2 (랭킹 불변): 모든 로짓에 **같은 양수 배율**을 곱하면 argmax와 랭킹 불변 → τ가 클립 안에서 움직여도 **분류 가능성(순위)은 유지**, 확률의 날카로움만 바뀜.

### 3.1 V2 실측 — 클립 항등식 + 신뢰도 비감소

```
tau_log_params      = [5.5, 4.6052]        # exp(5.5)=244.7 (클립 구간) vs log(100)
clip_identity_exact        = True           # 244.7도 클립되면 100도 100도 → 정확히 동일
ranking_invariant...       = True           # 스케일 0.5→100에서 argmax 불변
maxprob_by_scale   = [0.048856, 0.075337, 0.544253, 0.999996, 1.0]   # 스케일 0.5,1,4,20,100
maxprob_monotone_nondecreasing: True
```

- 확률은 스케일에 날카로워지지만(0.0489 → 1.0) **순위는 불변** — "softmax 온도"의 정확한 역할.
- 클립 = 학습의 안전장치이지 추론의 제약이 아니다: `exp(t)>100` 구간은 **정 보가 없는 이동**만 발생.

## 4. 재사용: zero-shot = 하이퍼네트워크가 만든 선형 분류기

- 분류기 가중치 `W_c = l2_normalize(T_e(프롬프트_c))` — **텍스트 인코더가 클래스 가중치를 '생성'**.
- 점수 = `cos(이미지, W_c) · exp(τ)` → softmax → argmax.
- 논문 동등 문구(§3.1.1): "*multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, temperature scaling*".
- 배치 교체(새 데이터셋)는 **텍스트 프롬프트 문자열만** 바꾸는 일 — 가중치 학습 없음. "하이퍼네트워크적 관점" (§3.1.1 본문).

## 5. 프롬프트 앙상블 — 임베딩 공간 평균 + 상각 (§3.1.1)

들어가는 텍스트마다 임베딩이 흔들린다(같은 클래스를 다른 문장으로 표현):

```
P_ck = a·W_c + σ·u_ck      (u_ck: 서로 독립 노이즈 방향, k = 1..K 개 프롬프트)
```

논문의 선택:

- 확률 앙상블: 매 추론마다 K회 텍스트 인코더 → softmax 평균.
- **임베딩 앙상블(논문)**: `P̄_c = mean_k P_ck`를 **1회 계산해 캐시** → 매 추론 1회 점검.

### 5.1 비용 구조 — 상각(amortize)

```
encoding cost per inference:  확률 앙상블 = K회   ·   임베딩 앙상블 = 1회   (데모 8:1)
```

- §3.1.1 원어: "*cache a single set of averaged text embeddings so that the compute cost of the ensemble is the same as using a single classifier when amortized*".
- **클래스 수가 많아도** (예: ImageNet 1,000클래스 × 80프롬프트) 캐시 1세트로 끝 — "무료가 아니라 선지불".

### 5.2 분산 감소 — 잔여 노이즈 비율 = 1/K

푸리에 항이 K개 독립이면:

```
E‖mean_k u_ck‖² = σ² · D/K          # 표본평균의 에너지 = 단일 항의 1/K
resid_single = σ² · D                 = 0.7098   (데모 실측, D=64, σ=0.1052)
resid_mean   = resid_single / K       = 0.085
ratio = 0.1198   vs   expected 1/K = 0.125   ✓ (K=8)
```

검색 방향으로도 이익: 정답 클래스 검색 정확도 **mean-프롬프트 0.7656 > 단일 프롬프트 0.7031** (+6.3pp). 논문이 보고한 방향(ImageNet 템플릿 단일 +1.3% → 80프롬프트 앙상블 +3.5%)과 동일 부호.
- 단, 이는 **합성 세계의 통계 성질** 실측 — 논문 수치와 직접 비교 금지(방향 재현).

### 5.3 주의 — 임베딩 평균이 항상 확률 평균과 같지는 않다

```
agreement_with_base(σ) = [1.0, 0.7812, 0.5, 0.2812]   # σ = 0, 0.2, 0.5, 1.0
identity_at_sigma_zero = True
```

- σ=0(모든 프롬프트가 동일 임베딩)에서는 확률 앙상블과 임베딩 앙상블이 **argmax 동치**.
- σ>0에서는 둘이 갈라지고, agreement는 σ에 비증가(공통난수 단조 구조) — "다른 문장을 어느 정도 섞어도 원래 방향을 유지한다"의 실측.

## 6. 선형 probe와의 접속 (§3.2)

- zero-shot 분류기 정의(§4)가 곧 선형 분류기 정의와 같다: `L2-normalized inputs + L2-normalized weights + no bias + temperature`.
- 따라서 "선형 probe"에서의 우위(66 모델 × 27 데이터셋; ViT-L/14@336px 12데이터셋 평균 +2.6%)는 **같은 형식의 실험 안에서 피처 품질 격차**만을 측정 — "zero-shot이 = 선형 probe의 특수(타깃 클래스를 말로 지정한) 경우"라는 구조적 연결.

## 7. 요약표 — 수식 ↔ 데모 ↔ 논문

| 구분 | 수식(논문 표기) | 데모 필드 | 논문 위치 |
|---|---|---|---|
| 로짓 | `A = l2_norm(I_f·W_i) · l2_norm(T_f·W_t)ᵀ · exp(t)` | `logits` | §2.2 + pseudocode |
| 손실 | `(CE(axis=0) + CE(axis=1)) / 2` | `loss_sym` | pseudocode |
| 온도 | `t` log-param, `exp(t) ≤ 100` 클립 | `tau_log_params` | §4 + 하이퍼파라 표 |
| zero-shot | `argmax_c cos(img, T(프롬프트_c))·exp(τ)` | V2 `ranking_invariant` | §3.1.1 |
| 앙상블 | `P̄_c = mean_k T(프롬프트_c)` 캐시 | V3 `encoding_cost_structure` | §3.1.1 |
| 분산 감소 | `resid_mean / resid_single ≈ 1/K` | V4 `ratio = 0.1198` | §3.1.1 프롬프트 개념 |

> 잔여 경계: (i) 데모는 **합성 세계 수학 구조**의 재현 — 논문 수치와 같은 숫자가 아님, (ii) 80 프롬프트는 실논문 값이고 데모는 K=8(계산량 축소, 방향 확인용), (iii) 아키텍처 계층(ViT 어텐션 풀링 등)은 데모 범위 밖.
