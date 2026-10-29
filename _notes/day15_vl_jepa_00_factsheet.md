---
layout: default
title: "Day 15-16 VL-JEPA — 00 · 검증 팩트시트 — VL-JEPA (Vision-Language Joint Embedding Predictive Architecture)"
paper_id: vl-jepa
date: 2026-11-02
---

# Day 15-16 검증 팩트시트 — VL-JEPA (Vision-Language Joint Embedding Predictive Architecture)

> **논문**: arXiv [2512.10942](https://arxiv.org/abs/2512.10942)  
> **저자**: Meta FAIR & HKUST (Yann LeCun 등)  
> **목적**: 2026.11.02~11.03 세션을 위한 논문 수치 및 스펙 사전 검증

---

## 1. 모델 핵심 사양 (검증 대상)

- **Vision Encoder**: ViT-H/14, ViT-L/16 등 (동결 여부 및 사전학습 가중치 출처 확인)
- **Predictor**: 가벼운 트랜스포머 레이어 (표현 공간에서의 다음 상태/누락 블록 예측)
- **Language Conditioning**: Cross-Attention 또는 프리픽스 토큰 결합 방식
- **Loss**: L1 / L2 또는 코사인 거리 기반 Representation Prediction Loss (픽셀 디코더 미사용)

---

## 2. 팩트 대조 체크리스트 (세션 당일 채우기)

- [ ] 타깃 인코더 EMA 모멘텀 스케줄링 값
- [ ] 마스킹 전략: 비디오 시공간 튜브 마스킹 vs 블록 마스킹
- [ ] 텍스트 주입 지점: 인코더 내부 vs 예측기(Predictor) 입력단
- [ ] 다운스트림 평가 벤치마크 (비디오 질의응답, 액션 예측, 제로샷 전이)
