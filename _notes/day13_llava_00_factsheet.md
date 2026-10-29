---
layout: default
title: "Day 13 LLaVA — 00 · 검증 팩트시트 — LLaVA (Visual Instruction Tuning)"
paper_id: llava
date: 2026-10-26
---

# 00 · 검증 팩트시트 — LLaVA (Visual Instruction Tuning)

> **검증 방법**: arXiv HTML 실험판 v2 전문(`arxiv.org/html/2304.08485`, 2023-12-11)을 직접 조회·대조. 수치는 원문 라인 그대로 인용.
> **원칙**: 모든 수치는 논문 본문·표·초록의 직접 출처만. 레포·모델카드 수치는 별도 라벨. 미확인 항목은 배제.

## 1.1 신원

- arXiv: [2304.08485](https://arxiv.org/abs/2304.08485) — 제출 2023-04-17(v1), 최종 개정 2023-12-11(v2). [arXiv 제출기록 직접 확인]
- 저자: Haotian Liu · Chunyuan Li · Qingyang Wu · Yong Jae Lee (UW–Madison · Microsoft Research · Columbia)
- Venue: **NeurIPS 2023 Oral** — arXiv 페이지 Comments 필드 직접 확인(2026-10-26)
- 공식 레포: [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — **Apache-2.0** (GitHub API license 필드 직접 확인)
- **범위 주의**: 이 키트는 논문 v1/v2(LLaVA v1)만 다룬다. **LLaVA-1.5는 별도 논문 — D14 범위.**

## 1.2 핵심 수치 — 원문 위치 표기

### (a) GPT-assisted instruction 데이터 (§3)

| 항목 | 수치 |
|---|---|
| 총량 | **158K unique language-image instruction-following samples** |
| 3종 분해 | conversation **58K** · detailed description **23K** · complex reasoning **77K** |
| teacher | **language-only GPT-4/ChatGPT** — 이미지 직접 입력 불가 → 기호적 표현만: (i) captions (ii) bounding boxes |
| 사람 주석 | 수집 중 유일한 사람 주석 = 휴먼 작성 seed 소수 예시(in-context seed) |
| 품질 관측 | GPT-4가 ChatGPT보다 일관적으로 고품질(예: spatial reasoning) — §3 얼리 실험 관측 |

합산 검산: 58 + 23 + 77 = 158 ✅

### (b) CC3M 필터 → Stage 1 데이터 (Appendix E 'CC3M')

| 항목 | 내용 |
|---|---|
| 절차 | SpaCy로 CC3M 전체 캡션에서 noun-phrase 추출 → unique phrase 빈도 집계 |
| 규칙 1 | 빈도 **< 3**인 phrase는 제외(rare 조합은 다른 캡션이 이미 커버) |
| 규칙 2 | 빈도 **> 100**이면 그 캡션들에서 **무작위 100개**만 선택 |
| 결과 | **약 595K image-text pairs** — 빈도 ≥ 3 개념 커버리지 유지 + 페어 수 축소 |
| 참고 그림 | noun-phrase 통계 전/후 비교 — 부록 Figure 7 |

### (c) 아키텍처 (§4.1)

| 요소 | 제원 |
|---|---|
| LLM | **Vicuna** — 당시 공개 체크포인트 중 지시수행 능력 최상(논문 근거 문장) |
| 비전 | **CLIP ViT-L/14** 사전학습 — 피처 `Z_v = g(X_v)` |
| 결합 | **단일 선형 projection W**: `H_v = W·Z_v` (식 1) — LLM 워드 임베딩 차원으로 투영 |
| 피처 위치 | **마지막 Transformer 층 이전/이후 그리드 피처** 둘 다 고려(§4.1) |
| 설계 논거 | 단순 projection = "lightweight → 데이터 중심 실험을 빨리 반복"; 정교한 결합(gated cross-attention · Q-former)은 future work |

[라벨-b] ViT-L/14 토큰 수(24²=576)는 **논문 본문 미명시** — CLIP config 본검증 필요. 본 키트는 "그리드 피처" 문장까지만 인용.

### (d) 2단계 학습 (§4.2 · §5 · Appendix C)

| 단계 | 데이터 | 학습 | 학습률 | 배치 | 소요 |
|---|---|---|---|---|---|
| Stage 1 (feature alignment) | CC-595K | **1 epoch** | **2e-3** | **128** | **≤ 4h** |
| Stage 2 (end-to-end) | LLaVA-Instruct-158K | **3 epochs** | **2e-5** | **32** | **≤ 10h** |
| ScienceQA 밀리 | ScienceQA | **12 epochs** | (Appendix C 세팅) | — | **≤ 4h** |

- Stage 1: 비전 인코더 + LLM **전부 고정**, 학습 파라미터 = **projection W만** — 논문 표현: "training a compatible visual tokenizer for the frozen LLM"
- Stage 2: 비전 고정, **W + LLM(φ) 갱신**
- 공통 하드웨어·옵티마: **8×A100** · Vicuna 하이퍼파라 상속 · Adam(**no weight decay**) · cosine LR · **warmup 3%** · FSDP + gradient checkpointing · BF16/TF32

### (e) LLaVA-Bench (COCO) — Table 4 상대 점수 [%] (기준: 텍스트 전용 GPT-4)

구성: COCO-Val-2014 임의 30장 × 질문 3종 = **90 questions**

| 훈련 데이터 | Conv | Detail | Complex | All |
|---|---|---|---|---|
| **Full data (best)** | 83.1 | 75.3 | 96.5 | **85.1** |
| Detail + Complex | 81.5 (−1.6) | 73.3 (−2.0) | 90.8 (−5.7) | 81.9 (−3.2) |
| Conv + 5% Detail + 10% Complex | 81.0 (−2.1) | 68.4 (−7.1) | 91.5 (−5.0) | 80.5 (−4.4) |
| Conversation만 | 76.5 (−6.6) | 59.8 (−16.2) | 84.9 (−12.4) | 73.8 (−11.3) |
| **No Instruction Tuning** | 22.0 (−61.1) | 24.0 (−51.3) | 18.5 (−78.0) | **21.5 (−63.6)** |

논문 자체 요지: ① 지시수행이 지시튜닝으로 **50점 이상** 향상 ② reasoning 데이터 소량 혼입이 대화 능력까지 끌어올림 ③ 3종 전부가 최고(85.1).

### (f) LLaVA-Bench (In-the-Wild) — Table 5 상대 점수 [%] (mean ± std)

| 모델 | Conv | Detail | Complex | All |
|---|---|---|---|---|
| OpenFlamingo | 19.3 ±0.5 | 19.0 ±0.5 | 19.1 ±0.7 | 19.1 ±0.4 |
| BLIP-2 | 54.6 ±1.4 | 29.1 ±1.2 | 32.9 ±0.7 | 38.1 ±1.0 |
| **LLaVA** | 57.3 ±1.9 | 52.5 ±6.3 | 81.7 ±1.8 | **67.3 ±2.0** |
| LLaVA† (GPT-4 3회 재평가 일관성) | 58.8 ±0.6 | 49.2 ±0.8 | 81.4 ±0.3 | 66.7 ±0.3 |

### (g) ScienceQA (§5.2 · Table 7) [%]

- 데이터: **21K 다중선택 문제** — 3 subjects · 26 topics · 127 categories · 379 skills; 분할 **12726 / 4241 / 4241** (train/val/test)

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

- 논문 서술: **"To the best of our knowledge, this is the first time that GPT-4 is used for model ensembling."**
- 이미지 문맥 문제에서 텍스트 전용 GPT-4가 성능을 끌어올리는 이유 — **이미지 없이도 답되는 문제를 식별·교정**하기 때문
- GPT-4 단독 실패 원인 관측: "insufficient context (이미지·플롯)" 보고

### (h) 설계 선택 ablation — Table 8 [%]

| 항목 | 변형 | 값 (최적 대비) |
|---|---|---|
| 비전 피처 (best) | **before last layer** | 90.92 |
| 비전 피처 | last layer | 89.96 (−0.96) |
| CoT (best) | answer-first 12ep | 89.77 (−1.15) |
| 수렴 관측 | reason-first는 **6ep**에서 89.77 동일 도달 — 이후 부가 개선 없음(24ep도 무개선) | — |
| 사전 | Stage 1 생략(from scratch) | **85.81 (−5.11)** |
| 크기 | 7B(같은 세팅) | 89.84 (−1.08) |

논문 귀결: ① reason-first가 수렴 속도를 크게 개선하나 최종 성능 기여는 작음 ② **사전 스킵 패널티가 최대**(−5.11p) — Stage 1 정합의 필요性의 근거.

### (i) 한계 (§5.1 Limitations · Appendix A)

- Knowledge coverage / 다국어(라면 사진 예) · 고해상도 세부(요구르트 브랜드 예) — In-the-Wild 벤치 난도 근거
- Appendix A 'Hallucination' 항목에 환각 관측 명시

## 1.3 라벨 정리

- [라벨-b] ViT-L/14 토큰 수(24²=576) · 어휘·스펙류 — CLIP/Vicuna config 본검증 필요(논문 밖 사양)
- [라벨-b·저자 eval] ScienceQA의 GPT-4 82.69 — 원문 공식 스코어가 아니라 본 저자가 수행한 평가("our eval")
- [관측] "~80K unique images"(§5.1) — 샘플 158K와 상이(이미지 수 vs 지시 샘플 수는 다른 축)

## 1.4 검증 판정선

- 초록 85.1%(상대 점수) = Table 4 'All' 85.1 ✅ 일치
- 초록 ScienceQA 92.53% = Table 7 judge 92.53 ✅ 일치
- Stage 구조(595K → 158K) §4.2·§5·Appendix C 상호 일치 ✅
