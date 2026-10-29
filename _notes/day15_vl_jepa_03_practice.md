---
layout: default
title: "Day 15-16 VL-JEPA — 03 · 실습 — VL-JEPA 축소 데모 및 Representation Alignment 실측"
paper_id: vl-jepa
date: 2026-11-02
---

# 03 · 실습 — VL-JEPA 축소 데모 및 Representation Alignment 실측

> **데모 스크립트**: `vl_jepa_demo.py` (세션 당일 구현)  
> **세션 일자**: 2026.11.02(월) ~ 11.03(화)

---

## 1. 실습 목표 및 4가지 검증 항목

1. **V1. Latent Space Prediction**:
   - 픽셀 복원 디코더 없이 타깃 임베딩 $\hat{s}_y$와 $s_y$의 거리가 단조 감소하는지 검증
2. **V2. Language Conditioning Effect**:
   - 올바른 언어 프롬프트 $z$가 주어졌을 때와 셔플된 프롬프트가 주어졌을 때의 예측 오차 갭 실측
3. **V3. Collapse 방지 검증 (Rank / Variance Check)**:
   - 표현 벡터들의 공분산 행렬 랭크가 붕괴하지 않고 유지되는지 확인 ($\text{Cov}(S)$ 유효 차원 수)
4. **V4. EMA 타깃 인코더의 역할**:
   - EMA 계수 $\tau=0.99$ vs $\tau=0$ (온라인 공유) 환경 비교 실측

---

## 2. 실행 명령어

```bash
python3 day15_vl_jepa/vl_jepa_demo.py --epochs 20 --device mps
```
*(결과 로그는 `day15_vl_jepa/runs/vl_jepa_demo_output.txt`에 기록)*
