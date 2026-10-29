---
layout: default
title: "Day 13 LLaVA — 04 · 비교표 — LLaVA 검증 수치 총정리"
paper_id: llava
date: 2026-10-26
---

# 04 · 비교표 — LLaVA 검증 수치 총정리

> 모든 수치는 `00_검증팩트시트.md`의 검증 셀만 사용. 원문 = arXiv HTML v2 전문(`arxiv.org/html/2304.08485`, 2023-12-11) 직접 조회.
> **범위 주의**: 이 표는 LLaVA v1(2304.08485)만 다룬다. **LLaVA-1.5(2310.03744)는 별도 논문 — D14 범위.** (D14 세션에서 본문 대조 예정은 1.5 abs 문구에만 한한다)

## 1. 지시 데이터 합성 엔진 (§3 · Table 1)

| 항목 | 값 |
|---|---|
| 총량 | **158K unique language-image instruction-following samples** |
| 3종 분해 | conversation **58K** · detailed description **23K** · complex reasoning **77K** (합산 검산 58+23+77=158 ✅) |
| teacher | **language-only GPT-4/ChatGPT** — 이미지 직접 입력 불가 → 기호적 표현만: (i) captions (ii) bounding boxes |
| 표기 주의 | visual image는 GPT 프롬프트에 **사용되지 않음(original word: "visual image is not used to prompt GPT")** |
| 사람 주석 | 유일한 사람 주석 = **in-context seed 소수 예시** |
| 품질 관측 | GPT-4가 ChatGPT보다 일관적으로 고품질(예: spatial reasoning) |

## 2. Stage 1 데이터 — CC3M 필터 (Appendix E 'CC3M')

| 규칙 | 내용 |
|---|---|
| 절차 | SpaCy noun-phrase 추출 → unique phrase 빈도 집계 |
| 저빈도 | 빈도 **< 3** 제외 |
| 고빈도 | 빈도 **> 100**이면 캡션별 무작위 100개만 |
| 결과 | **약 595K image-text pairs** — 빈도 ≥ 3 개념 커버리지 + 페어 수 축소 |

## 3. 아키텍처 + 2단계 학습 (§4.1 · §4.2 · §5 · Appendix C)

| 요소 | 제원 |
|---|---|
| 비전 | **CLIP ViT-L/14** 사전학습 — 그리드 피처(마지막 층 전/후 둘 다 고려, Table 8에서 전이 +0.96p) |
| 결합 | **단일 선형 projection W**: H_v = W·Z_v (식 1) — Q-former/gated cross-attention **없음**, 정교한 결합은 future work 명시 |
| LLM | **Vicuna** — 공개 체크포인트 중 지시수행 능력 최상 |
| Stage 1 | CC-595K · **θ={W}만** · 1 epoch · lr **2e-3** · batch **128** · ≤**4h** |
| Stage 2 | LLaVA-Instruct-158K (또는 ScienceQA) · **θ={W, φ}** · 비전 고정 · 3 epochs · lr **2e-5** · batch **32** · ≤**10h** |
| 공통 | **8×A100** · Vicuna 하이퍼파라 상속 · Adam(no weight decay) · cosine LR · **warmup 3%** · FSDP + gradient checkpointing · BF16/TF32 |
| 학습률 축소비 | Stage 1→2 = **100×** (2e-3 → 2e-5) — 역할 전환(정합 → instruction tuning)의 지표 |

## 4. LLaVA-Bench (COCO) — Table 4 상대 점수 [%] (기준: 텍스트 전용 GPT-4)

구성: COCO-Val-2014 임의 30장 × 질문 3종 = **90 questions**

| 훈련 데이터 | Conv | Detail | Complex | All |
|---|---|---|---|---|
| **Full data (best)** | 83.1 | 75.3 | 96.5 | **85.1** |
| Detail + Complex | 81.5 (−1.6) | 73.3 (−2.0) | 90.8 (−5.7) | 81.9 (−3.2) |
| Conv + 5% Detail + 10% Complex | 81.0 (−2.1) | 68.4 (−7.1) | 91.5 (−5.0) | 80.5 (−4.4) |
| Conversation만 | 76.5 (−6.6) | 59.8 (−16.2) | 84.9 (−12.4) | 73.8 (−11.3) |
| **No Instruction Tuning** | 22.0 | 24.0 | 18.5 | **21.5** |

- 지시튜닝 효과: **21.5 → 85.1 (+63.6)** — 논문 자체 요지 "50점 이상 향상"
- 공동 확장: Complex 소량 혼입이 Conv까지 끌어올림(73.8 → 80.5) / 3종 전부 최고(85.1)

## 5. LLaVA-Bench (In-the-Wild) — Table 5 상대 점수 [%] (mean ± std)

| 모델 | Conv | Detail | Complex | All |
|---|---|---|---|---|
| OpenFlamingo | 19.3 ±0.5 | 19.0 ±0.5 | 19.1 ±0.7 | 19.1 ±0.4 |
| BLIP-2 | 54.6 ±1.4 | 29.1 ±1.2 | 32.9 ±0.7 | 38.1 ±1.0 |
| **LLaVA** | 57.3 ±1.9 | 52.5 ±6.3 | 81.7 ±1.8 | **67.3 ±2.0** |
| LLaVA† (GPT-4 3회 재평가 일관성) | 58.8 ±0.6 | 49.2 ±0.8 | 81.4 ±0.3 | 66.7 ±0.3 |

- [조건] 상대점수 = 정답 캡션·박스를 직접 본 **텍스트 전용 GPT-4** 기준 (1~10점 심판 평균)
- 난도 근거: 라면 사진(다국어 지식)·냉장고 세부(고해상도 세부) — §5.1 한계 절이 벤치 설계로 이어짐

## 6. ScienceQA — Table 7 [%]

| 모델 | All |
|---|---|
| Human | 88.40 |
| GPT-3.5 | 73.97 |
| GPT-3.5 w/ CoT | 75.17 |
| GPT-4† (text-only · 2-shot · **저자 자체 eval**) | 82.69 |
| LLaMA-Adapter | 85.19 |
| MM-CoT Large | 91.68 (당시 SoTA) |
| **LLaVA** (before-last 피처 · reason-first · 12ep) | **90.92** |
| LLaVA + GPT-4 (complement) | 90.97 |
| **LLaVA + GPT-4 (judge)** | **92.53 — new SoTA** |

- 데이터: **21K 다중선택 · 12726/4241/4241 분할**
- 앙상블 2형식: (i) complement — GPT-4 실패 시 LLaVA 예측 대체(90.97) (ii) judge — 답이 다르면 GPT-4가 최종 판정(**92.53**)
- 논문 문구: "first time that GPT-4 is used for model ensembling"
- 특기: **84×12 epoch 학습 ≤4h** 가 단독 90.92의 예산

## 7. 설계 소거 — Table 8 [%]

| 항목 | 변형 | 값 (최적 대비) |
|---|---|---|
| 비전 피처 (best) | **before last layer** | 90.92 |
| 비전 피처 | last layer | 89.96 (−0.96) |
| CoT (best) | answer-first 12ep | 89.77 (−1.15) |
| CoT | reason-first = **6ep로 동일 도달 89.77** — 이후 부가 개선 없음(24ep도 무개선) | — |
| 사전 | Stage 1 생략(from scratch) | **85.81 (−5.11)** |
| 크기 | 7B(같은 세팅) | 89.84 (−1.08) |

- 논문 귀결: ① reason-first가 수렴 속도를 크게 개선하나 최종 성능 기여는 작음 ② **사전 스킵 패널티가 최대(−5.11p)** — Stage 1 정합의 필요성의 근거. 데모 V1의 σ=0 복원 항등식이 이 경제의 구조를 말한다.

## 8. 데모 실측 요약 — `llava_demo.py` V1~V4 (합성 세계, seed 42)

| 검증 | 핵심 실측 | PASS 근거 |
|---|---|---|
| V1 Stage 1 정합 | σ=0 복원 resid **0.0** · 잔차 직교성 **0.0** · 모듈 비트 동일 · heldout cos **0.668 vs probe 0.048** | 항등식 + 정리 + 방향 |
| V2 데이터 혼입 | 루프=폐형 **0.0 편차** · All 순서 == Table 4 · 혼입 25.8765 > 대화전용 25.8553 (Conv 축) | 항등식 + 방향 |
| V3 loss 마스크 | 마스크 항등식 exact · instr 기울기 행 **정확히 0** · dW == 답변 rows dW · 위치9 CE **0.775 < 0.943 < 2.773** | 항등식 + 구조 + 방향 |
| V4 judge 앙상블 | 분해 항등식 exact · **0.9253 > 0.908** · 호출률 0.2481==0.2475 · MC dev 1e-4 | 항등식 + 통계 + 방향 |

> 최종 PASS 4/4 — 세부는 `03_실습.md`.

## 9. 이웃 모델과의 구조 대응 — 직접 비교 ❌, 구조 대응 ◯

| 관점 | CLIP (D12) | Flamingo | BLIP-2 | **LLaVA (D13)** |
|---|---|---|---|---|
| 비전-언어 결합 | (모달 결합 없음 — 대조 정렬) | **gated cross-attention** (tanh 게이트) | **Q-former** (learned queries) | **단일 선형 projection W** |
| 학습 전략 | 양방향 대조(사전학습 1단계) | 언어 모델 사전 + 어댑터 | 2단계 동결(대조→생성) | **2단계: 정합(W만) → end-to-end(W+φ)** |
| 데이터 축 | WIT 400M 캡션형 쌍 | 대규모 인터리브 이미지-텍스트(세부 수치는 본 스터디 미확인 — [라벨: 일반 지식]) | 129M 캡션 사전학습(세부 수치는 본 스터디 미확인 — [라벨: 일반 지식]) | **CC-595K → GPT-4 합성 158K** |
| 말이 만드는 것 | 분류기가 말에서 옴 | few-shot 대화 | 생성 캡션/QA | **지시 데이터 문체의 3종 분리** |
| 서로 조인 없음 | — 사전학습 신호·데이터·평가 모두 달라 **Table 수치 직접 비교 금지** (Flamingo·BLIP-2 수치는 Table 5의 동일 벤치(LLaVA-Bench) 한정에서만 대응 열로 존재) | | | |

| 관점 | GLIP (D10) | SAM (D10) | **LLaVA (D13)** |
|---|---|---|---|
| 텍스트의 역할 | 프롬프트 = **토큰 피처**(로짓 스왑) | (텍스트 없음 — 포인트/박스) | **토큰 열 자체가 입력 시퀀스** — 시각 토큰이 어휘 자리에 참석 |
| box-as-language | ✅ ( GLUT 단일층) | ❌ | **(box를 GPT-4에 기호로 넘겨 지시 데이터를 합성 — §3)** |
| 학습 예산 관례 | O365+GoldG 등 grounding 쌍 | SA-1B 1.1B 마스크 | **≤14h (Stage 1 ≤4h + Stage 2 ≤10h · 8×A100)** |

## 10. LLaVA-1.5(D14)로 넘기는 다리

| 이 논문(v1)에서 | 다음 논문(2310.03744, abs 기준) |
|---|---|
| 단일 선형 W | **2층 MLP connector** (Δ: 파라미터 구조만 — Q-former 대신) |
| CLIP ViT-L/14 | CLIP-ViT-L-**336px** |
| 지시 데이터 158K | 공개 데이터 **~1.2M** + 학술 과제 대상 VQA + response-formatting 프롬프트 |
| 11 벤치마크 SOTA 주장 | (1.5 본문 확보 시 D14 세션 대조) |
