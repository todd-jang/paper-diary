# Day 15-16 · VL-JEPA: Vision-Language Joint Embedding Predictive Architecture (월드모델 2일 트랙)

> **일정:** 2026.11.02(월) ~ 11.03(화) (2일간 집중 트랙)  
> **논문:** arXiv [2512.10942](https://arxiv.org/abs/2512.10942) — *VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language*  
> **저자/기관:** Meta FAIR + HKUST (Yann LeCun 저자 참여)  
> **핵심 키워드:** World Models, Self-Supervised Learning, Latent Space Prediction, Non-Generative Multimodal Learning

---

## 🧪 심화 산출물 키트 (`day15_vl_jepa/`)

- 📜 [`00_검증팩트시트.md`](https://github.com/todd-jang/paper-diary/blob/main/day15_vl_jepa/00_검증팩트시트.md) — 논문 원문 스펙·하이퍼파라미터·평가 프로토콜 대조
- 🧭 [`01_코드워크스루.md`](https://github.com/todd-jang/paper-diary/blob/main/day15_vl_jepa/01_코드워크스루.md) — 인코더-예측기(Predictor) 파이프라인 및 멀티모달 컨텍스트 주입 흐름
- 📐 [`02_수식유도.md`](https://github.com/todd-jang/paper-diary/blob/main/day15_vl_jepa/02_수식유도.md) — 에너지 기반 JEPA 목적함수, Collapse 방지 EMA 정합
- 💻 [`03_실습.md`](https://github.com/todd-jang/paper-diary/blob/main/day15_vl_jepa/03_실습.md) — `vl_jepa_demo.py` 축소 구현 및 Representation Alignment 검증
- 📊 [`04_비교표.md`](https://github.com/todd-jang/paper-diary/blob/main/day15_vl_jepa/04_비교표.md) — MAE(픽셀 복원) vs LLaVA(토큰 생성) vs JEPA(표현 공간 예측) 3대 패러다임 비교
- 📚 [`06_Quest_안내.md`](https://github.com/todd-jang/paper-diary/blob/main/day15_vl_jepa/06_Quest_안내.md) — Socratic Quest 및 Feynman 설명 질문지

---

## 🎯 2일 집중 트랙 관통 질문

1. **왜 '생성(Generative)'이 아닌 '표현 공간 예측(JEPA)'인가?**
   - 픽셀 세부 복원(MAE)이나 오토리그레시브 토큰 예측(LLM/LLaVA)의 정보 낭비와 비효율을 어떻게 해결하는가?
2. **시각과 언어의 조인트 임베딩 공간 구성:**
   - 텍스트 컨디셔닝이 비디오/이미지의 미래 상태 예측(World Model)에 어떻게 작용하는가?
3. **Representation Collapse 방지 메커니즘:**
   - 음성 샘플(Negative pair) 없이 타깃 인코더의 EMA와 정규화만으로 무너짐(Collapse)을 막는 수학적 원리.

---

## 🗓️ 2일 학습 로드맵

- **Day 1 (11/02, 월): 이론 & 아키텍처 해부**
  - LeCun 월드모델 비전(AMI)과 JEPA 패러다임 이해
  - Vision-Language 컨디셔닝 메커니즘 분석
  - 팩트시트 작성 및 수식 유도 (00, 01, 02)
- **Day 2 (11/03, 화): 실습 & 비교 분석 & 퀘스트**
  - 파이토치 축소 데모 구현 및 임베딩 예측 검증 (03)
  - MAE vs LLaVA vs VL-JEPA 비교표 확정 (04)
  - 최종 Socratic Quest 및 답변 작성 (06)
