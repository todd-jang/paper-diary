---
layout: default
title: "Day 10 GLIP & SAM — Day 10 · 02 — 수식 유도: GLIP의 로짓 스왑과 SAM의 애매성 손실"
paper_id: glip-sam
date: 2026-10-23
---

# Day 10 · 02 — 수식 유도: GLIP의 로짓 스왑과 SAM의 애매성 손실

> 모든 등호는 논문 표기 기준(GLIP Eqs. 2–3·4–6 / SAM App. §A), 검증된 표 셀만 인용.

## 1. GLIP — 분류 로짓을 "말"로 교체하는 정확한 형태

### 1.1 기준선: 통상 탐지 모델 (Eq. 2)
- 이미지 인코더 `O = Enc_I(Img) ∈ R^{N×D}`, 클래스 임베딩 `W ∈ R^{C×D}`
- 분류 로짓 `S_cls = O Wᵀ ∈ R^{N×C}`, 그리고 회귀 로짓(박스) `S_box`
- 손실 `L = L_cls(S_cls) + λ_loc L_loc(S_box)` — **cls 쪽이 "닫힌 C개"** 라는 것이 문제.

### 1.2 스왑: W 대신 프롬프트 토큰 (Eq. 3)
- 프롬프트 문장 `Prompt`를 언어 인코더에 통과: `P = Enc_L(Prompt) ∈ R^{M×D}` (M = 토큰 수, BERT의 sub-word 256 토큰 상한)
- 로짓 교체: `S_ground = O Pᵀ ∈ R^{N×M}`
- 정답 행렬 변환: `T ∈ {0,1}^{N×c}`(phrase 단위) → **`T' ∈ {0,1}^{N×M}`**(sub-word 확장) — 긍정 phrase의 모든 sub-word를 긍정화, 추가 토큰·`[NoObj]`는 부정
- 손실 함수 자체는 유지(focal/binary): "*the target we use is the same as our previously described detection loss except that classes are replaced by words appears in the prompt" (§3.1)*
- 추론 후처리: 토큰 확률 **평균**으로 phrase 점수 복원 (`SCORE_AGG: "MEAN"`), 후보 선택 `TOPK=9`.

**형태 비교 표**

| 항목 | 통상 탐지 | GLIP |
|---|---|---|
| 분류 점수 | `S_cls = O Wᵀ` | `S_ground = O Pᵀ` |
| 정답 | 클래스 index (C 고정) | sub-word 0/1 행렬 T′ |
| 후보 세계 | **닫힌** C개 | **열린** M(문장에 따라) |
| 추가 바뀐 것 | 없음 | 회귀머리 유지, NMS 구조 유지(레포 NMS_TH 존재) |

### 1.3 딥 퓨전의 3식 (Eq. 4–6)
언어 경로는 "**사전학습 BERT 위에 새로 추가한 BERTLayer**"만큼이 유일한 학습 추가물:

```
(4) O^{t2i}_i, P^{i2t}_i = X-MHA(O_i, P_i)          # 교차 모달 헤드 어텐션
(5) O_{i+1} = DyHeadModule(O_i + O^{t2i}_i)         # 이미지 쪽 융합
(6) P_{i+1} = BERTLayer(P_i + P^{i2t}_i)            # 언어 쪽 융합
```

- `O^{t2i}`는 **"추가 문맥 백터"** — 없으면 late-fusion으로 환원(§3.2).
- 실효 증거(레포 검증 소거, Table 8): O365-only에서 ✗→✓로 바꾸면 COCO 42.9→44.9, ODinW 0-shot 28.7→**33.2**.
- 퓨전 비용(Table 7, **P100 단일 GPU·batch 1** — 해상도 미기재, 첨부 금지): GLIP-T 4.84→**2.52 FPS**/1.0→2.4GB, GLIP-L 0.54→**0.32 FPS**/4.8→7.7GB — "말을 스며들게 하는" 비용이 FPS 2배 가까이.

### 1.4 GLIP의 "동일 성능" 주장의 정체 (App. §B)
- DyHead/Swin-Tiny, COCO val2017 변형 전후 **둘 다 49.4 AP** — "캡션 질을 닫힌 분류로 바꿔도 같은 성능"을 숫자로 보증하는 대조.
- 의미: 스왑은 **정보 손실이 아니라 후보 세계의 사이즈 문제**일 뿐.

## 2. SAM — 애매성을 손실 구조로 푸는 방법

### 2.1 "valid mask"의 형식적 정의 (§2)
- 과제: 프롬프트 `p` → 마스크 `M(p)` (이미지 1024², 임베딩 64²)
- 애매성: `p`가 여러 대상(옷 vs 사람)을 가리킬 수 있음 → **둘 중 "어느 하나라도 그럴듯하면" 정답**으로 정의. 즉 손실이 마스크 집합 {M₁, M₂, M₃} 위의 **min**으로 정의되는 구조.

### 2.2 손실 3식 (App. §A)
```
L_mask = 20 · L_focal + 1 · L_dice          # focal : dice = 20:1
L_rank = MSE(estIoU, IoU(M̂, GT))            # 스케일링 상수 1.0
L = min_{k∈{1,2,3}} (L_mask^k + L_rank^k)   # "최손값만 백프로프"
```
- "**we backprop only the minimum loss over masks**"(§3) — 애매성을 3중 최소화(multiple outputs + min)로 처리.
- IoU 랭킹 토큰: 추론 시 3개 후보 중 어느 것을 낼지 재판정 — 인간 연구에서 "SAM(3개 선택) 8.1 vs 표기(single) 7.7" 차이(LVIS 인간 평가, Fig. 18)가 이 랭킹 토큰의 효용.

### 2.3 다중 프롬프트 예외의 수식적 정체 (App. §A)
- 프롬프트 수 > 1 → 출력 후보 1개: `M(p₁,...,pₖ) = M₄` ("*This fourth mask is never returned for a single prompt and is the only mask returned for multiple prompts*").
- 의미: 후보 개수는 **프롬프트 결합도의 함수** — 점 하나는 애매성 후보 3개, 점+박스 같은 결합 프롬프트는 1개.
- 이 때문에 "SAM은 항상 3개 낸다"는 서술은 위변조 위험. 인용 시 "**단일 프롬프트 기준 3개**"로 기술.

### 2.4 데이터 엔진 3단계의 누적식 (§4)
```
masked_so_far: 4.3M → 10.2M → 1.1B
annotator cost: 34s/mask → 14s/mask → (자동)
model/trainings: 6회 → 5회 → (그리드 샘플로 확장)
```
- "SAM이 마스크를 그리는 도구"라는 **재귀 구조** — 데이터가 모델의 다음 판본을 만들고, 모델이 다음 데이터를 만든다.
- 품질 필터(§B): 예측 IoU 88.0 · 안정성 95.0 · box NMS 0.7 → 낮은 신뢰를 먼저 걸러내고 그 1.1B를 훈련에 씀.

### 2.5 손도장 평가를 숫자로 서는 법 (Fig. 18 인간 평가, 조건 표기)
- 10점 척도 평균(단일 점, LVIS v0.5): RITM **6.5** ← SAM-single **7.7** ← SAM **8.1** ← GT **8.5** (95% CI와 함께)
- 훈련 조건(모델카드 §F): **256×A100 / 68시간 / 90k iters / batch 256** / drop-path 0.4 / layer-decay 0.8.

## 3. 형식 대조 — GLIP vs SAM의 "열림"
| | GLIP | SAM |
|---|---|---|
| 열리는 것 | 후보(class · phrase) | 과제(마스크 생성 규칙) |
| 수식 대응 | `W` → `P` (분류기 로짓의 "말" 교체) | 출력 후보 개수를 프롬프트 결합도에 연동 |
| 손실 변화 | cls 로스는 유지, 타겟이 T′로 확장 | min over 3 · focal:dice 20:1 |
| 스케일 증거 | grounding 데이터가 검출 데이터를 넘어섬(LVIS APr 15.0→20.8) | 데이터가 4.3M→1.1B로 250배 |
| 실전 발톱 | ODinW 0-shot 52.1 ≈ 10-shot 50.8 | rare AR 65.8 vs ViTDet 58.3 |

**결론**: 둘 다 "특정 함수 하나(classifier / mask decoder)"를 **일반화 조이개를 열어주는 문장**으로 바꿨다. 다음 세션(D12 CLIP)은 이 문장을 **이미지 인코더 자체의 사전학습 목적**으로 빼는 단계다.
