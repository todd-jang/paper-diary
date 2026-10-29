---
layout: default
title: "Day 13 LLaVA — 01 · 코드워크스루 — LLaVA (§3 데이터 파이프라인 → §4 모델·학습 → §5 평가)"
paper_id: llava
date: 2026-10-26
---

# 01 · 코드워크스루 — LLaVA (§3 데이터 파이프라인 → §4 모델·학습 → §5 평가)

> 논문: Visual Instruction Tuning (2304.08485 · NeurIPS 2023 Oral)
> 성격: 논문 파이프라인·수식·표의 구조워크스루. 레포 코드 복원이 아니라 원문 절의 직접 대조.
> 근거: 00_검증팩트시트.md (arXiv HTML v2 전문 조회 2026-10-26)

## 1. 데이터 파이프라인 — "텍스트 전용 GPT를 teacher로 쓴다"

### 1.1 문제 (§3 서두)

멀티모달 instruction-following 데이터가 부족하다.
사람에게 수집은 시간이 들고 절차가 불명확하다.
그래서 **ChatGPT/GPT-4로 image-text pair를 instruction-following 형식으로 변형**한다.

### 1.2 이미지를 GPT에게 넘기는 방법 (§3)

GPT는 이미지를 직접 볼 수 없다. 그래서 기호(symbolic)로 바꿔서 넘긴다.

- **(i) Captions** — 여러 시점의 장면 문장.
- **(ii) Bounding boxes** — 개체 유형 + 위치. 원문 예: `person: [0.681, 0.242, 0.774, 0.694]`.

### 1.3 3종 응답 (Table 1 · §3)

| 유형 | 내용 |
|---|---|
| conversation | 사진에 대한 다양한 질문 — 개체 유형/개수/동작/위치/상대 위치. "definite answer"만 |
| detailed description | 질문 리스트를 GPT-4가 먼저 정리 → 1개를 표본으로 상세 서술 생성 |
| complex reasoning | 심층 추론 질문 — 단계적 논리를 요구하는 답변 |

- seed 예시 몇 개 = 수집 중 **유일한 사람 주석** (in-context seed)

### 1.4 산출 (§3 마지막 문단)

- 총 **158K unique samples** = conversation **58K** + detailed **23K** + complex reasoning **77K**
- GPT-4 대 ChatGPT: GPT-4가 일관되게 고품질 (spatial reasoning 개선 관측)
- 온전 확장(§3 서두의 naive expansion)은 "lacks diversity and in-depth reasoning" → 위 3종이 보완

### 1.5 CC3M 필터 (Appendix E 'CC3M')

1. SpaCy로 전체 CC3M 캡션에서 noun-phrase 추출, unique 빈도 집계.
2. 빈도 < 3 → 제외 ("rare combinations concept and attributes").
3. 빈도 > 100 → 무작위 100개만.
4. 나머지를 저빈도부터 채워 넣음 → **약 595K pairs**.
- 목적: "concept coverage와 training efficiency의 균형" (§4.2 Stage 1 본문).

## 2. 아키텍처 (§4.1)

- 비전: **CLIP ViT-L/14** 사전학습. 피처는 Z_v = g(X_v).
- 결합: **단일 선형 projection W** — H_v = W·Z_v (식 1). LLM 워드 임베딩 차원과 같아야 함.
- 그리드 피처: **마지막 Transformer 층 이전과 이후, 둘 다 고려** — Table 8에서 이전(before last)이 0.96p 더 좋음.
- LLM: **Vicuna** — 공개 체크포인트 중 지시수행 능력 근거로 선택.
- 설계 논거(원문): projection이 lightweight → "iterate data centric experiments quickly". Flamingo식 gated cross-attention, BLIP-2식 Q-former는 future work.

## 3. 학습 (§4.2)

```
Stage 1 — Pre-training for Feature Alignment
  데이터: CC-595K (naive expansion, 단일 턴)
  고정:   비전 인코더, LLM
  학습:   projection W 만  (lr 2e-3 · batch 128 · 1 epoch · ≤4h)

Stage 2 — Fine-tuning End-to-End
  데이터: LLaVA-Instruct-158K (chatbot 시나리오) 또는 ScienceQA
  고정:   비전 인코더
  학습:   W + LLM(φ)      (lr 2e-5 · batch 32 · 3 epochs · ≤10h)
```

- Stage 1의 내용(원문): "training a compatible visual tokenizer for the frozen LLM"
- Stage 2 두 시나리오: ① 158K로 챗봇 튜닝(3종 uniform sampling) ② ScienceQA (단일 턴 구조)
- 공통: 8×A100 · Vicuna 하이퍼파라 상속 · Adam(no weight decay) · cosine LR · warmup 3% · FSDP + gradient checkpointing · BF16/TF32 (§5 서두 + Appendix C)

## 4. 학습 목표 — 식 (2)~(3)

- 턴 구조(식 2): 첫 턴은 [질문, 이미지] 또는 [이미지, 질문] 중 무작위, 이후 턴은 질문만.
- 시퀀스(Table 2): `system-message <STOP> Human: instruct <STOP> Assistant: answer <STOP> ...` — 아이콘은 녹색(답변) 토큰만 loss에 들어감.
- 목표(식 3): 답변 토큰의 자기회귀 확률을 최대화. 이미지 X_v는 모든 조건부에 명시적으로 걸려 있음 (grounding 강조).
- loss 마스크: instruction·system-message 토큰은 예측 대상이 아님 → 02_수식유도.md의 항등식 검사 V3에서 재확인.

## 5. 평가 (§5)

| 실험 | 설정 | 결과(Table) |
|---|---|---|
| Chatbot — 정량 | LLaVA-Bench (COCO): 30장 × 3종 = 90 | Full data **85.1** · No-tuning **21.5** (Table 4) |
| Chatbot — 실란트 | In-the-Wild: 라면·냉장고 등 난도 | LLaVA **67.3** vs BLIP-2 38.1 · OpenFlamingo 19.1 (Table 5) |
| ScienceQA | 21K 다중선택 · 4241 test | LLaVA 90.92 · **+GPT-4 judge 92.53 (SoTA)** (Table 7) |
| 설계 ablation | 피처 위치 · CoT 순서 · 사전 스킵 · 7B | from-scratch **−5.11** 최대 패널티 (Table 8) |

 평가자(§5.1): 답변·GT·질문을 **텍스트 전용 GPT-4 심판**이 1~10점 채점 → "relative score" w.r.t. 텍스트 전용 GPT-4.

## 6. GPT-4 앙상블 (§5.2)

- (i) complement: GPT-4가 실패(memory/context 부족 보고)하면 LLaVA 예측으로 대체 → 90.97
- (ii) judge: 두 모델 답이 다르면 GPT-4가 최종 판정 → **92.53, new SoTA**
- 의미(논문): "first time that GPT-4 is used for model ensembling."
- 시사점: **텍스트 전용 judge가 이미지 문항까지 개선** — 이미지 불필요 문제를 식별하는 능력.

## 7. 한 줄 요지

1. 사람 주석을 seed 몇 개만 두고 GPT-4로 지시 데이터를 만들었다.
2. 결합은 단순 선형 projection — 학습 저력(≤14h 총량)으로 'data-centric' 반복이 가능했다.
3. 3종 데이터를 섞어야 지시·추론·상세가 동시에 좋아졌다 (Table 4).

## 8. 오늘 세션의 두 방향 (다음 세션으로 이월)

- 다음 D14: LLaVA-1.5 (별도 논문 — 협력 힌트: "LLaVA-1.5 = 데이터 강화 + CLIP resolution 업그레이드").
- 오늘의 관통 질문 재확인 — Day13_LLaVA.md '🎯 오늘의 관통 질문'과 Q1~Q3 채점 완료(95.6%).
