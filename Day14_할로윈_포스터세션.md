# Day 14 · 할로윈 지식공유 포스터 세션 (팀원 공유 arXiv 4편 리뷰/재포장)

> **일정:** 2026.10.30(금) 자율세션  
> **형식:** 팀원 공유 논문 리뷰 → 수치 검증 → A0/A1 할로윈 테마 포스터 제작  
> **범위:** 심화 키트 폴더(day14/) 없음 — 리뷰·포스터 기록만 진행

---

## 📌 오늘 다룬 4편 (모든 수치는 논문 고유 라벨 + 조건 첨부, arXiv abs 원문 직접 재검증 통과)

### 1. STAMPlus — 학생 학습 모델링 향상 (arXiv [2608.02791](https://arxiv.org/abs/2608.02791))
- 연속형/이진형 과제 아키텍처 통합 + 문맥 임베딩 모듈 컨텍스트 스위칭
- 검증: 주요 표 수치 원문 대조 32/32 셀 토큰 (Table I·X·XI 등, 조건 라벨 그대로)

### 2. Search-R1 — 검색 증강 LLM 강화학습 (arXiv [2503.09516](https://arxiv.org/abs/2503.09516))
- RL로 `<search>` 액션 트리거 학습 + 검색 인용 마스킹 손실
- **Table 3:** PPO 0.431 vs GRPO 0.350/0.396 (NQ, F1/em) / **Table 4 (마스킹):** w. mask 0.431 vs w.o. 0.343 (NQ F1) / **Table 2:** 7개 벤치마크 전 행 — 합계 33/33 셀 토큰 검증

### 3. Perception Encoder (PE) — 비전 인코딩 (arXiv [2504.13181](https://arxiv.org/abs/2504.13181))
- 멀티스케일 문맥 사전학습 0M→17M samples (Fig 6 캡션)
- Table 8/Fig 8/Figs 9–12 파생치 검산 일치 (+0.081, +0.088(25.7%) — 표 라벨 조건 첨부)

### 4. 좌우 대칭 깨짐 기제 연구 (arXiv [2601.12809](https://arxiv.org/abs/2601.12809))
- **Fig 4:** [M_B=M_rep=1, M_h=4] 구성 실험 (−61.8% 변화율 — 캡션/`math alttext` 직접 추출)
- **Fig 10:** [M_B=2, M_rep=2, M_h=4, d_head=32, d_MLP=512, d_model=128] 프로토콜 (+5.0 / −5.8 / −3.8 검산 일치)
- 구성값·정렬 기제 가시화

---

## 🎃 산출물: `poster/halloween/` (커밋 승인 완료)

- 🎨 [`halloween.css`](https://github.com/todd-jang/paper-diary/blob/main/poster/halloween/halloween.css) — 다크 퍼플/펌킨 테마 공통 스타일
- 🖨️ [`poster_stamplus.html`](https://github.com/todd-jang/paper-diary/blob/main/poster/halloween/poster_stamplus.html) · [`poster_search_r1.html`](https://github.com/todd-jang/paper-diary/blob/main/poster/halloween/poster_search_r1.html) · [`poster_pe.html`](https://github.com/todd-jang/paper-diary/blob/main/poster/halloween/poster_pe.html) · [`poster_symmetry.html`](https://github.com/todd-jang/paper-diary/blob/main/poster/halloween/poster_symmetry.html) — A0(정식 인쇄본)/A1(compact) 2 사이즈
- 📊 `poster/runs/poster_halloween-*_{A0,A1}.{pdf,png}` — 검증 완료 빌드 산출물 8쌍
- 자세한 내용은 [`poster/halloween/README.md`](https://github.com/todd-jang/paper-diary/blob/main/poster/halloween/README.md) 참조

---

## ✅ 세션 품질 게이트

- 모든 수치: **arXiv abs 원문 직접 대조**(표 셀 토큰 단위) + 파생치 검산
- 조건·라벨 첨부 확인 / 게재 불가 수치: 없음
- 건너뛴 것: 심화 키트 폴더(day14/) 실습·퀘스트 — 기록만 진행
