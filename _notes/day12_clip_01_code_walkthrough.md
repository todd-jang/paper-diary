---
layout: default
title: "Day 12 CLIP — Day 12 · 01 — 코드워크스루: CLIP이 \"말\"을 classifier로 바꾸는 법"
paper_id: clip
date: 2026-10-23
---

# Day 12 · 01 — 코드워크스루: CLIP이 "말"을 classifier로 바꾸는 법

> **원칙**: 모든 수치는 `00_검증팩트시트.md`의 검증 셀만 사용. 본 문서는 코드(의사코드) 순서로 구조를 뜯어본다:
> CLIP = **텍스트 인코더가 클래스 분류기의 가중치를 생성하는 하이퍼네트워크** + 그 가중치를 만드는 신호가 배치 속의 N×N 대조 하나뿐인 사전학습.

## 1. 출발점 — 예측 기반과 대조 기반 사이의 선택

> §1.1: "*Two main lines of work"* — (1) 자연어에서 독립적으로 개념(단어)을 학습한 뒤 시각 개념证明과 연결하는 **predictive** 방법, (2) 이미지→캡션 예측 방법. 두 방향 모두 "풍부한 감독"의 비용은 큰 규모를 요구했다.

CLIP의 선택은 이 두 개가 아닌 **자연어로 유도된 mystery 문제(개념)**에서의 **대조 학습**이다:

- 데이터: **WIT** = 400M (image, text) pairs, 500k 쿼리, 쿼리당 최대 20,000 페어 (§2.1)
- 과정: 배치 32,768에서 **N²−N** 잘못된 쌍을 동시에 봄(나任何人: 배치 크기가 곧 negative 수. §2.2의 32,768 배치가 곧 신호 강도).
- 점수: 이미지·텍스트 임베딩의 **코사인 유사도 × exp(τ)** (τ는 학습 파라미터).

이 설계의 최소형은 figure pseudocode 고명(`src/pseudocode.pdf`) 11줄이다(pypdf로 텍스트 원본 확인):

```
I_f = image_encoder(I)                 # [n, d_i]
T_f = text_encoder(T)                  # [n, d_t]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)   # [n, d_e]
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)   # [n, d_e]
logits = np.dot(I_e, T_e.T) * np.exp(t)        # [n, n]
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t)/2
```

- **`loss_i` (axis=0)**: 열 → 텍스트가 기준. 줄 i가 올바른 이미지를 고르는 CE (텍스트→이미지 방향)
- **`loss_t` (axis=1)**: 행 → 이미지가 기준. 줄 i가 올바른 텍스트를 고르는 CE (이미지→텍스트 방향)
- 두 방향의 평균 = **대칭 교차엔트로피** (§2.2: "*loss = (loss_i + loss_t)/2*") — 양쪽 인코더 모두가 업데이트되도록.

### 1.1 그다음 추가된 것 하나 — 비선형 projection 제거 + temperature

- 멀티모달 공간 성자회 media projection 비선형 제거: "*We also simplify the image transformation function*" + "*a single linear projection ... We did not notice a difference in training efficiency*" (§2.2 요약)
- τ는 **log-parameterized**로 학습되고, logit 스케일은 **×100으로 클립**(§4.1 + 부록 하이퍼파라미터 표 Maximum temperature 100.0). §3.1.1에서 언급하듯 검증체계 설계를 하지 않은 이유는 τ=0.07 환산치에서 시작하는 것은 별개.

### 1.2 팀은 제작 공장 — WIT를 만드는 방법 (§2.1)

- 자연어 위키 `>= 100`회 이상 등장한 단어 500k 쿼리 + 고PMI bi-gram + 위키 문서 타이틀 + WordNet synset 미포함 추가 → "어휘 커버리지와 개념 균형"의 발상
- 쿼리당 **≤ 20,000 페어** 상한 = "한 단어에 수십만 동영상 이미지가 붙는 데이터 치우침 방지"(균형 용어)
- 이미지 EN 필터(불량 이미지 태그 제거) — YFCC100M이 이 필터만으로 100M→**15M**으로 축소한 사례(§2.1)가 같은 필터의 세기를 보여준다.

## 2. 아키텍처 — 말에서 [EOS]까지 (§2.4)

| 구성 | 실제 제원 | 설득 문맥 |
|---|---|---|
| 텍스트 인코더 | **63M / 12층 / 폭 512 / 8 heads** transformer | "512의 중간 크기에 ViT-B/32 등의 작은 이미지 인코더와 쌍"
| 토큰화 | 소문자 **BPE**, 빈/max 시퀀스 = **76** + `[SOS]`/`[EOS]` → 총 77 | README API 토크나이저와 통일
| 어휘 | 부록 하이퍼파라 표 = **49,408** (본문 "49,152"는 오기 가능성 — 팩트시트 §1.4-1) | 실험 보고 시 부록 표 기준
| 텍스트 표현 | **`[EOS]` 위치의 최상위 활성화** → layer norm → linear projection | "분류에 유용한 캡션 모음을 채우는 목적"
| 이미지 인코더 | RN50 계열: **ResNet-D 개선 + antialiased rect-2 blur pooling + attention pooling** / ViT: patch+position 임베딩 뒤 **layer norm 1층 추가** | 성능 선형 제약 내에서 sу울된 구조
| 스케일링 | ResNet : EfficientNet 식 폭·깊이·해상도 **균등 배분**; ViT: ViT-B/32·16 → ViT-L/14 etc; 텍스트는 **폭만** 스케일 | 고명 표에서 ViT 계열 비효율을 명시한 이유

> [ResNet-D + blur + attention pooling]은 *folk* 별도의 인코더라고 주장하는 것이 아님. §2.4의 "forture-purpose vision transformer" 문장과 접속되는 실현 경로.

## 3. 학습 제한 (§4.1 + 부록 하이퍼파라미터 표)

| 항목 | 값 |
|---|---|
| 모델 | RN50 · RN101 · RN50x4 · RN50x16 · RN50x64 · ViT-B/32 · ViT-B/16 · ViT-L/14(+**@336px 1 에폭 추가**) — **8종** |
| 기본 "CLIP" | **ViT-L/14@336px**"*Unless otherwise specified*" |
| 에폭 / 배치 | **32 / 32,768** |
| 증강 | **random square crop**만 (즉, 표준 증강 빈약) |
| 최적화 | decoupled weight decay Adam + cosine LR — *"*각 스케일 하이퍼파라 검토 후 단일 최적 조건을 유지" |
| Tài원 | **mixed-precision + gradient checkpointing + half-precision Adam 통계 + GPU별 부분 유사도** (필요 조건: 배치 32,768) |
| 비용 | **RN50x64 = 592 V100 × 18일**, 최대 ViT = **256 V100 × 12일**; 단일표 12.8B × 32 에폭 = 1초/장이면 **405년** |

## 4. Zero-shot = 하이퍼네트워크 classifier (§3.1)

CLIP의 zero-shot 파이프라인은 학습된 **텍스트 인코더**가 분류기 가중치를 마당 배부하는 구조다:

1. `[클래스 이름]`를 "라벨의 텍스트 인코더 = 선형 분류기 가중치"로：
   - 사전학습 중 배치 속 텍스트에서 학습된 임베딩 → 텍스트 인코더는 *컨텍스트 스케일 표현*을 **생성**하는 함수.
2. 이미지 임베딩 I 를 *(L2-정규화된 클래스 임베딩)*에 점곱 → τ exp 스케일 곱 → softmax:
   - 논문 해석: "*multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, temperature scaling*" (§3.1.1) — 로지스틱 회귀 관점으로 정확히 동등.
3. zero-shot "데이터셋 적응" = **텍스트 프롬프트만 교체** — 그 어떤 모델 파라미터도 학습되지 않는다.

### 4.1 프롬프트 엔지니어링·앙상블 (§3.1.1)

- 다의성(예: crane 기계/새, boxer 견종/선수) 해결 위해 **"A photo of a {label}."** 템플릿 작품 → **ImageNet +1.3%**
- **카테고리 지시형** (Pets "a type of pet", Food101 "a type of food", …) → 임베딩 공간의 "노이즈 감소"
- **앙상블은 확률이 아니라 임베딩 공간**에서: "*cache a single set of averaged text embeddings*" → **추론 시 사전 계산 1회** (§3.1.1) — "임베딩 앙상블"이 "아마 80 프롬프트 앙상블"이면 천수 배수가 아니라 **라운드 2회** 추론 비용.
- **80개 컨텍스트 프롬프트 앙상블 = ImageNet +3.5%**(단일 기본 템플릿 대비); 36 데이터셋 평균 앙상블 이득 **~5 pt** — "*similar to the gain from using 4 times more compute ... but is 'free'*".

### 4.2 분류가 아니라 "개념 정렬" 성질 (§3.1.2)

- 27 데이터셋: "supervised ResNet-50 features에 fully supervised 정규화 logistic regression을 붙인 것"(6 ML 방식, 강공정 대조)보다 **16/27 승**(ImageNet 포함).
- **STL10 99.3%**(zero-shot, "new state of the art") — 예제 0개로 학습되지 않은 데이터셋의 성취.
- UN때 성공/실패 양 범위: Stanford Cars·Food101 +20%p↑ / Flowers102·FGVCAircraft −10%p↓ — "fine-grained 개념 간만 컴팩트하게 구분하는 개념이 말로 잘 전달되지 않는다".
- **약점 데이터셋**: EuroSAT · RESISC45(위성) · PatchCamelyon(종양) · CLEVRCounts(개수) · GTSRB(표지) · KITTI Distance — "quite weak on several specialized, complex, or abstract tasks" (§3.1.2).

## 5. 표현 학습 — 선형 분류기로 다시 측정 (§3.2)

- 자연어 참징 "CLIP의 성능은 사전학습만의 부산물인가"를 수치에 붙임:
 - 방법: 인코더 뒤 **선형 분류기만** 붙여 평가 (66 모델 × 27 데이터셋) — fine-tuning은 중간 분석이 안 되는 흑상을 만든다.
 - 신규 데이터셋: **Country211**(211개국 조사), **Rendered SST2**(OCR용 448×448) 등 "구조적 조건이 약한" 데이터셋.
- 결과: **CLIP ViT-L/14@336px**가 기존 최선(ImageNet 학습 모델들)보다 **12 데이터셋 평균 +2.6%**, 27 확장 스위트에서 **+5%**; **21/27 데이터셋**에서 Noisy Student EfficientNet-L2 피처를 이김 (Fig. linear-probe-clip-vs-enet).
- 계산 효율: **ViT가 CLIP ResNet보다 약 3× 연산 효율** (§3.2 본문).

## 6. Robustness — Shift에서 얻는 것 (§3.3)

- zero-shot CLIP은 자연 분포 이동에서 **effective robustness**가 기존 분류 모델보다 높은 위치에 놓임 (이 "gap을 최대 75% 축소" — Fig. robustness plot 캡션).
- ImageNet으로 통한 인코더 적응(fine-tune, linear probe 등)은 **IN 정확도를 키워도 robustness gain이 지속되지 않음** — ImageNet-R −4.7 / ObjectNet −3.8 / Sketch −2.8 / ImageNet-A −1.9의 trade-off (Table 14/15 방향).
- **few-shot**으로 갈 수록 in-distribution 분류기에 가까워진다는 해석: "*high effective robustness seems to result from minimizing the amount of distribution specific training data*" (§3.3 본문).
- 부록 표: Zero-Shot CLIP **IN 76.2 · V2 70.1 · A 77.2 · R 88.9 · ObjectNet 72.3 · Sketch 60.2** (+Vid PM0/PM10 95.3/89.2, YTBB 95.2/88.5) — IN 정확도는 NS EfficientNet-L2보다 낮지만 그 외 5 곳에서 우위 갱신.

## 7. Limitations — "1000× compute" 코너 (§7)

- "*we estimate around a **1000x increase in compute** is required for zero-shot CLIP to reach overall SOTA performance*" — "평균적으로"의 전제 명시.
- 잔존 약점: **fine-grained**(차 형태, 꽃 분류, 항공기 종류), **계수**(CLEVRCounts), **MNIST**(88% — "*An embarrassingly simple baseline of logistic regression on raw pixels outperforms zero-shot CLIP*").
- 개념화 한계: zero-shot 분류는 "**choosing from only those concepts in a given zero-shot classifier**" — capion generation 같은 생성형 함수가 아님.
- 개발 집합 27-suite는 "*undeniably co-adapted with the development and capabilities of CLIP*" — 개발 중 validation 반복 조회에 따른 관측 편향.
- 데이터와 편향: "**CLIP models learn many social biases**" — training data was unfiltered / uncurated (§7·§8 연결).

## 8. 데모 매핑 — `clip_demo.py` V1~V4 (말 ↔ 코드)

| 검증 | 논문 매칭 | 데모에서 보여주는 것 |
|---|---|---|
| **V1 대칭 CE** | pseudocode `loss=(loss_i+loss_t)/2` + §2.2 대칭 문장 | 분해 항등식(동일 로짓 2가지 축 CE 재계산 == 정확 일치) · 정렬 배치 < 랜덤 배치 · 대각 행 최대 전제의 τ 비증가(softmax 날카로움 구조) |
| **V2 τ 클립 ×100** | §4 log-parameterized + "clipped ... by more than 100" | exp(τ)>100이 클립되어 스케일 100과 정확히 동일(항등식) + 공통 양 스케일 랭킹 불변 + 신뢰도 비감소 추세 |
| **V3 임베딩 앙상블 상각** | §3.1.1 "cache a single set of averaged text embeddings" | σ=0에서 단일 분류기와 argmax 동치 + 인코딩 비용 1:K 구조 + agreement(σ) 비증가 |
| **V4 프롬프트 평균 1/K** | §3.1.1 80 프롬프트 앙상블(본 데모 K=8) | K개 프롬프트 임베딩의 잔여 노이즈 비율 ≈ 1/K + mean-프롬프트 검색 정확도 > 단일 프롬프트 |

> 세부 실측 로그는 `03_실습.md`, 수치 유도는 `02_수식유도.md`로.
