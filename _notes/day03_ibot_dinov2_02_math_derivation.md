---
layout: default
title: "Day 03 iBOT & DINOv2 — 02. 수식 유도 — 온라인 토크나이저 자기증류(iBOT)와 DINOv2 목적함수"
paper_id: ibot-dinov2
date: 2026-10-18
---

# 02. 수식 유도 — 온라인 토크나이저 자기증류(iBOT)와 DINOv2 목적함수

> iBOT(2111.07832)와 DINOv2(2304.07193)의 목적함수를 단계별로 분해한다.
> 관통 목표: **"BEiT의 고정 dVAE 토크나이저를, EMA teacher 로 대체·학습"** 하는 수학을 따라가기.

---

## 0. 표기(Notation)

| 기호 | 의미 |
|---|---|
| `x` | 입력 이미지 |
| `u, v` | `x` 의 두 증강 뷰(augmented views) |
| `û, v̂` | `u, v` 에 blockwise 마스킹을 적용한 뷰(student 입력) |
| `f_s, h_s` | student 백본·projection head (파라미터 `θ`) |
| `f_t, h_t` | teacher 백본·projection head (파라미터 `θ'`, EMA) |
| `P^patch(·)` | 패치 헤드의 softmax 분포 |
| `P^cls(·)` | [CLS] 헤드의 softmax 분포 |
| `m_i ∈ {0,1}` | i번째 패치가 **가려졌는지** 지시(`M` = 가린 집합) |
| `τ_t, τ_s` | teacher / student 온도 |
| `c` | centering 벡터(teacher logits 평균) |

---

## PART A. iBOT — MIM 을 지식증류(KD)로 보기

### A.1 출발점: MIM = KD

BEiT 는 `z_i`(고정 dVAE 하드 토큰)를 정답으로 쓴다. iBOT 는 이를 **교사분포에서 증류**로 재해석한다:

```
L_KD(φ→θ) = − Σ_x Σ_i  log P_θ^patch(x_i)   ·   P_φ^patch(x_i)^T
             └ student 예측 ─┘                  └ (고정) teacher=토크나이저 ┘
```

- `φ` = **고정된** 토크나이저(BEiT 의 dVAE). 여기서 `φ` 를 **온라인(θ')** 으로 바꾸면:

```
L_MIM(θ'→θ) = − Σ_i  m_i · P_{θ'}^patch(u_i)^T · log P_{θ}^patch(û_i)      ... (1)
```

- `û` : student 가 보는 **가린** 뷰, `u` : teacher 가 보는 **원본** 뷰.
- `m_i` : 가려진 패치만 손실에 반영.
- **soft 타깃** `P_{θ'}^patch(u_i)`: hard argmax 가 아니라 분포 → 패치 의미 모호성을 반영.

### A.2 대칭화

두 뷰 `u↔v` 를 평균해 방향 편향을 없앤다:

```
L_MIM^sym = ½ [ L_MIM(u→û) + L_MIM(v→v̂) ]
```

### A.3 L[CLS] — 전역 자기증류

토크나이저가 **의미 있는** 분포를 내려면, teacher 가 전역 의미를 가져야 한다. 그래서 `[CLS]` 에서 DINO식 자기증류:

```
L_cls = − P_{θ'}^cls(v)^T log P_θ^cls(u)  −  P_{θ'}^cls(u)^T log P_θ^cls(v)   ... (2)
        (centering + sharpening 적용)
```

- **centering**: teacher logits 에서 배치평균 `c` 를 뺀다(붕괴 방지).
- **sharpening**: 낮은 온도 `τ_t` 로 분포를 뾰족하게.
- `L_cls` 가 **온라인 토크나이저의 품질을 떠받친다**(전역 의미 부트스트랩).

### A.4 EMA teacher (온라인 토크나이저의 정의)

teacher 는 student 의 지수이동평균, student 그래프에서 **stop-grad**:

```
θ'  ←  λ · θ'  +  (1 − λ) · θ          (λ ≈ 0.996)   ... (3)
```

- teacher 는 **천천히 진화하는 정답지** → 붕괴 방지·안정 학습.
- `f_t ∘ h_t^patch` = **online tokenizer**. 별도 사전학습(2단계)이 **불필요**.

### A.5 총 목적함수

```
L_iBOT = L_cls  +  L_MIM
```

- `L_cls` : 전역 의미(토크나이저 품질↑)
- `L_MIM` : 국소 패치 의미(표현의 지역 구조↑)
- 결과: **82.3% linear probing / 87.8% fine-tuning** (ImageNet-1K, headline).

---

## PART B. DINOv2 — DINO + iBOT + KoLeo

### B.1 DINO 손실 (전역)

teacher prototype 분포 `p_t`, student 분포 `p_s`:

```
L_DINO = − Σ  p_t · log p_s       ... (4)
```

- `p_t`: teacher `[CLS]` → softmax → **centering(또는 Sinkhorn)**.
- `p_s`: student `[CLS]` → softmax.

### B.2 Sinkhorn-Knopp centering (SwAV; 3 iterations)

DINO 의 "평균 빼기" 대신 **doubly-stochastic 정규화**로 배치를 균형화한다:

```
반복 3회:
  Q ← Q / Q.sum(dim=0, keepdim=True)     # 열(prototype 축) 정규화
  Q ← Q / Q.sum(dim=1, keepdim=True)     # 행(sample 축) 정규화
```

- 효과: 한 prototype 으로 쏠리는 **차원 붕괴**를 강하게 억제.
- student 는 softmax, teacher 는 Sinkhorn 결과를 타깃으로.

### B.3 KoLeo regularizer (특징 퍼뜨리기)

배치 내 점 `x_1..x_n` (L2 정규화)에 대해:

```
L_koleo = − (1/n) Σ_{i=1}^{n} log( d_{n,i} ) ,
          d_{n,i} = min_{j≠i} ‖ x_i − x_j ‖      ... (5)
```

- **Kozachenko–Leonenko 미분 엔트로피 추정량**에서 유래 → 특징이 공간을 **고르게** 채우도록.
- 효과: **검색(retrieval)** 개선(Oxford-M 55.6 → 63.9).

### B.4 목적함수 조립

```
L_DINOv2  ≈  L_DINO  +  α · L_iBOT(MIM)  +  β · L_koleo
             └전역┘      └패치 국소┘          └분산 정규화┘
```

- 역할 분담(절제실험):
  - **MIM(iBOT)** 제거 → 분할(ADE-20k) 47.1 → 44.2 (↓)
  - **KoLeo** 제거 → 검색(Oxford-M) 63.9 → 55.6 (↓)
- 즉 **전역·국소·분산** 세 항이 각각 다른 벤치를 담당.

### B.5 증류 (대형 → 소형)

1B 파라미터 `ViT-g/14` 교사를 **동결**하고 student 로 증류:

```
θ_s ← argmin  D( student(x) , teacher_frozen(x) )     (마스킹·stochastic depth 제거)
```

- 증류가 **from-scratch 보다 우수**(ViT-L 에서도). frozen 특징이 이미 강력 → **finetuning 선택사항**.

---

## PART C. 절제실험 (DINOv2, 검증된 수치)

| 항 | 설정 | INet-1k | INet-A | ADE-20k(mIoU) | Oxford-M(mAP) |
|---|---|---|---|---|---|
| 데이터 | LVD-142M | 85.8 | **73.9** | 47.7 | 64.6 |
| 데이터 | uncurated 142M | 83.3 | **59.4** | 48.5 | 54.3 |
| KoLeo | ✕ → ✓ | 85.3→**85.8** | 70.6→72.8 | 47.2→47.1 | **55.6→63.9** |
| MIM(iBOT) | ✕ → ✓ | 85.3→**85.8** | 72.0→72.8 | **44.2→47.1** | 64.3→63.9 |

> **읽는 법:** 데이터 큐레이션(LVD-142M)이 ImageNet-A 를 크게 올리고(59.4→73.9), **KoLeo 는 검색**, **MIM 은 분할**을 각각 끌어올린다. 세 요소가 **서로 다른 축**을 담당한다.

---

## PART D. 요약표 (목적함수 대조)

| 항목 | **BEiT** | **iBOT** | **DINOv2** |
|---|---|---|---|
| 타깃 | dVAE 하드 토큰 `z_i` | teacher **soft** 패치분포 | DINO+iBOT soft |
| 손실식 | `CE(z_i, p_MIM)` | 식(1) `L_MIM` + 식(2) `L_cls` | 식(4) `L_DINO` + `L_iBOT` + 식(5) `L_koleo` |
| 토크나이저 | 고정 dVAE(2단계) | **EMA teacher(식 3)** | EMA + 증류 |
| centering | — | teacher centering | **Sinkhorn-Knopp(3)** |
| 특기 | 의미 토큰 | dense/robust | 범용 frozen 특징 |

**요약:** 세 방법 모두 "가린 것을 조건부로 예측"한다. 차이는 **정답의 출처**다 — BEiT 는 **미리 학습한 고정 토크나이저**, iBOT 는 **온라인 EMA teacher(soft)**, DINOv2 는 **그 둘을 합치고 분산정규화(KoLeo)까지 더한 것**.
