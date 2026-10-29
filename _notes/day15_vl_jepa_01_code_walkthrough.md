---
layout: default
title: "Day 15-16 VL-JEPA — 01 · 코드워크스루 — VL-JEPA 아키텍처 및 텐서 파이프라인"
paper_id: vl-jepa
date: 2026-11-02
---

# 01 · 코드워크스루 — VL-JEPA 아키텍처 및 텐서 파이프라인

> **대상**: VL-JEPA (Joint Embedding Predictive Architecture for Vision-Language)  
> **세션 일자**: 2026.11.02(월) ~ 11.03(화)

---

## 1. 아키텍처 전체 흐름

```
[비전 입력 (Frames / Patches)] ──> [Context Encoder (Online)] ──> s_x (컨텍스트 임베딩)
                                                                     │
[언어 컨디셔닝 (Prompt/Action)] ──> [Text Encoder / Embed] ─────┼──> [Predictor] ──> ŝ_y (예측된 타깃 표현)
                                                                     │                      ▲
                                                                     ▼                      │  Loss D(ŝ_y, s_y)
[타깃 비전 (Future / Masked)] ───> [Target Encoder (EMA)] ────────────────────────────> s_y (실제 타깃 표현)
```

---

## 2. 핵심 모듈별 라인바이런 (학습 세션 중 분석)

1. **Context Encoder**: 마스킹되지 않은 시각 패치를 표현 벡터 $s_x$로 인코딩
2. **Target Encoder (EMA)**: 마스킹된 영역을 인코딩하되 gradient backprop 차단 (`stop_gradient`)
3. **Predictor**: 가벼운 트랜스포머 네트워크로 언어 지시문 $z$와 $s_x$를 결합하여 $s_y$를 예측
4. **EMA Updater**: 타깃 인코더 파라미터 $\theta_{target} \leftarrow \tau \theta_{target} + (1-\tau) \theta_{online}$
