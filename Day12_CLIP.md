# Day 12 — CLIP: 자연어가 분류기가 되는 방식 (대조 사전학습 · zero-shot · 프롬프트 앙상블)

> **2026-10-23 (논문 읽기 — 1일 당김, 2026-10-23 확정)** · 원문: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (OpenAI · ICML 2021, PMLR v139) · 원문 텍스트: `day12_clip/src/clip_paper.tex`(LaTeX e-print — ar5iv/HTML 원문 미제공 대체)
> 오늘의 초점: **"말이 분류기"** — 대칭 교차엔트로피(두 방향이 같은 logits) · τ 클립(정보 없는 이동만 금지하는 이유) · 임베딩 앙상블(비용 상각 1:8) · 프롬프트 평균(≈1/K 분산 감소)이 각기 답하는 것.
> 진행 방식: 오전 자료 완성(심화자료 7종 + 데모 PASS 4/4 + 차트 4종 — 모든 수치는 LaTeX 원본 검증선 준수) → 저녁 Socratic Quest(5문항).
> 이전 Day 연결: **Day 10(GLIP/SAM)**에서 텍스트의 첫 접점(로짓 스왑/모호성 3분해)을 잡았고, **CLIP은 그 전제(텍스트 임베딩 공간)를 만든 논문**. 다음 D11 자율(10/24) → **D13 LLaVA(10/26)** — "말이 지시"로 확장. GLIP(텍스트=로짓) ↔ CLIP(텍스트=분류기) ↔ LLaVA(텍스트=지시)의 나선이 여기서 이어진다.

## 🧪 심화 산출물 (`day12_clip/`)
- 📜 [`00_검증팩트시트.md`](./day12_clip/00_검증팩트시트.md) — LaTeX 원본 e-print 검증. 원칙: **모든 수치는 논문 표/초록 또는 공식 README + 조건(분할·백본·해상도·GPU)**. 원문 불일치 명시(본문 BPE 어휘 "49,152" ↔ 부록 "49,408" 등) + 인용 금지 목록
- 🧭 [`01_코드워크스루.md`](./day12_clip/01_코드워크스루.md) — 공식 pseudocode 11줄 인용(l2_normalize · logits = I_e·T_eᵀ·exp(t) · loss = (loss_i+loss_t)/2) + 아키텍처·학습 제한 표
- 📐 [`02_수식유도.md`](./day12_clip/02_수식유도.md) — 대칭 CE 2축 분해 · τ 클립 항등식 · 1/K 분산 감소 정리
- 💻 [`03_실습.md`](./day12_clip/03_실습.md) — `clip_demo.py` **PASS 4/4**(합성 세계, 논문 수치 재현 아님 · `runs/clip_demo_output.txt`): V1 대칭 CE(분해 항등식 + 대각 행 최대 전제 + τ 비증가) · V2 logit 스케일 ×100 클립(항등식 exact, 랭킹 불변) · V3 임베딩 앙상블(상각 1:8, σ=0 동치, agreement 비증가) · V4 프롬프트 평균(ratio 0.1198 ≈ 1/8=0.125)
- 📊 [`04_비교표.md`](./day12_clip/04_비교표.md) — Table 1(`zeroshot_table` 라벨 인용) · few-shot/앙상블/linear probe · robustness · GLIP/SAM 구조 대응(직접 비교 ❌)
- 🛠️ [`05_실전_배포_가이드.md`](./day12_clip/05_실전_배포_가이드.md) — openai/CLIP 설치·API·함정 체크리스트
- 📚 [`06_Quest_안내.md`](./day12_clip/06_Quest_안내.md) — 저녁 세션용(질문 원본·준비물)
- 📊 **차트 4종** — `day12_clip/runs/figures/`: `01_clip_training_scale.png` · `02_clip_table1_vngrams.png` · `03_clip_dataset_deltas.png` · `04_clip_robustness.png` (생성: `make_charts.py`)
- 🔒 **입력 원본 확보** — `day12_clip/src/`(clip_paper.tex · 공식 README · LICENSE 등 40 files — 표 라벨(`zeroshot_table`, `table:robustness` 등)로 인용하는 근거 원본)

## 🎯 오늘의 관통 질문

**CLIP의 zero-shot 분류기가 27개 데이터셋 중 16개에서 "완전 지도 linear proxy"를 이겼다 — 학습 예제 0개를 쓰고. 그런데 지는 11/27은 "개념을 말로 전달"이 어려운 범주(계수·표지·거리)에 몰려 있다. "텍스트=분류기"의 강함과 약함이 같은 원리의 양면이라면, GLIP(텍스트=로짓 근사) ↔ CLIP(텍스트=분류기) ↔ LLaVA(텍스트=지시)의 나선에서 CLIP이 고정한 것과 여전히 열지 못한 것은 각각 무엇인가?**

## 📖 읽기 범위

| 자료/절 | 무엇을 볼 것인가 |
|---|---|
| `00_검증팩트시트.md` | 표 번호·라벨·조건 — 모든 수치의 유일한 출처 창구(컴파일 번호 인용 금지) |
| pseudocode (§2.1 Fig. 1 우측, 11줄) | l2_normalize · logits = I_e·T_eᵀ·exp(t) · loss = (loss_i+loss_t)/2 — "대칭"의 정체 |
| §3.1.1 | 프롬프트 앙상블 — ImageNet 단일 +1.3% → 80개 앙상블 +3.5% · "cache a single set of averaged text embeddings" |
| Table 1 (라벨 `zeroshot_table`) | 27 데이터셋 zero-shot vs fully supervised linear proxy — 16/27 승 |
| `table:robustness` | robustness 벤치 격차 — 라벨·조건은 팩트시트로 추적 |
| §7 (스케일) | zero-shot 오류 44× compute 구간 log-log linear · "전반 SOTA"에는 ~1000× · 12.8B passes = 405년 |
| `02_수식유도.md` · `03_실습.md` | 2축 분해 · τ 클립 항등식 · 1/K 분산 감소 + V1~V4 실측(논문 수치와 직접 비교 금지) |
| `04_비교표.md` · `05_실전_배포_가이드.md` | 표 수치 총정리 · openai/CLIP 실전 |

## 🧠 오늘의 Socratic Quest (5문항 — 논문서술 형식)

> **작성 형식.** 문항당 **하나의 단락**: ① **주장** → ② **근거**(식·표·캡션 출처 직접 인용 — 출처 없는 숫자는 인정하지 않음) → ③ **경계 조건**. 힌트는 [`day12_clip/06_Quest_안내.md`](./day12_clip/06_Quest_안내.md).

**Q1. "대칭" 교차엔트로피 — 대칭이 아니어도 학습은 되지 않나?**
- **핵심**: pseudocode의 `loss_i = CE(logits, labels, axis=0)`(텍스트 방향)와 `loss_t = CE(logits, labels, axis=1)`(이미지 방향)의 차이, 그리고 이 두 방향이 **같은 logits 행렬**을 두 번 쓰는 이유.
- **참조**: `03_실습.md` V1(분해 항등식 exact, 정렬 배치 τ 비증가) / `02_수식유도.md` §2.

**Q2. τ를 "log 파라미터"로 여기고 ×100 클립을 두는 이유?**
- **핵심**: (i) 클립 구간(`exp(τ)>100`)은 **정보 없는 이동**만 발생(항등식 — `03_실습.md` V2), (ii) 공통 양수 배율은 argmax·랭킹을 못 바꾼다(정리), (iii) 그러므로 τ가 관리하는 것은 **분류 경계**가 아니라 **확률의 날카로움(신호 강도)**.
- **함정 질문**: "클립을 빼면 학습이 안 되나?" — 본 데모는 V2 실측으로 방향 확인(합성 세계).

**Q3. 프롬프트 앙상블을 왜 확률이 아니라 임베딩 공간에서 하나?**
- **핵심**: (i) 캐시 1세트 → 추론 비용 1회 인코딩(확률 앙상블은 K회 — 비용 구조 `03_실습.md` V3), (ii) K개 프롬프트 평균은 잔여 노이즈 비율 ≈ 1/K(`03_실습.md` V4: ratio 0.1198 vs 1/8 = 0.125), (iii) σ=0 동치 항등식 + σ>0에서 agreement 비증가.
- **참조**: §3.1.1 원문 "cache a single set of averaged text embeddings ... when amortized" / ImageNet 단일 +1.3% → 80개 앙상블 +3.5%.

**Q4. zero-shot 16/27 승리 — 대조 조건이 뭐였고, 왜 일부에서 지는가?**
- **핵심**: 대조군 = "supervised RN50 특성에 **fully supervised 정규화 로지스틱 회귀**"(27 데이터셋 각각, 16/27 승) — 학습 예제 0개를 쓰고 이긴 것이 포인트. 반대로 지는 곳(EuroSAT·RESISC45·PatchCamelyon·CLEVRCounts·GTSRB·KITTI Distance)은 **"개념을 말로 전달"이 어려운 범주/계수/표지/거리** 종류 — 약점 데이터셋 목록을 원인과 엮어 토론.
- **확장**: STL10 **99.3%**는 "학습 예제 0개로 신기록"의 사이드 케이스.

**Q5. 훈련 스케일 — 44×·1000×·405년은 각각 무엇을 말하나?**
- **핵심**: (i) 8모델 spanning "~2 orders of magnitude of compute", zero-shot 오류는 **44× compute 구간에서 log-log linear**(5 모델 · 39 evals · 36 datasets), (ii) 그런데 "zero-shot CLIP이 전반적 SOTA에 도달"에는 **~1000×**가 필요(§7 — 무한 스케일 규약이 아님), (iii) 400M×32 epochs = **12.8B passes**, 1장/초면 **405년**.
- **확장**: MNIST zero-shot 88%와 raw-pixel 로지스틱 회귀 대조의 의미("기능이 아니라 감독 신호"의 좋은 반례).

> **📝 채점자에게 (2026-10-23 저녁 세션):** D13 LLaVA(10/26) 대비 — CLIP은 "말을 분류기로", LLaVA는 "말을 지시로": 이 관점 차이를 학생이 스스로 짚을 수 있으면 통과 · 좋은 답변의 조건: (1) 검증된 숫자(팩트시트의 셀)를 조건을 붙여 인용했는지 (2) 표 인용의 라벨(`zeroshot_table`/`table:robustness`/나머지는 라벨) 추적 여부 (3) 합성 데모 실측값과 논문 수치를 직접 비교하지 않았는지 · 채점이 어려우면 `04_비교표.md` §9(GLIP/SAM과의 구조 대응)의 "직접 비교 ❌" 항목을 같이 읽게 할 것.

## ✍️ 나의 답변 (본인이 작성 — **논문서술 형식**, Q1~Q5 순서)

<!-- (미작성) 2026-10-23 저녁 세션에서 채움 — 형식: 문항당 하나의 단락 ①주장 ②근거 ③경계. 준비물·힌트는 day12_clip/06_Quest_안내.md -->

## 🔍 자기점검 (답변 직후 스스로)

- **Q1.** (i) 두 CE 방향의 의미를 구분했는가 (ii) 같은 logits 행렬 2회 사용 이유를 설명했는가 (iii) V1 분해 항등식·τ 비증가를 근거로 붙였는가
- **Q2.** (i) 클립 구간이 "정보 없는 이동"임을 항등식으로 보였는가 (ii) 공통 배율=argmax 불변 정리를 서술했는가 (iii) τ의 역할을 "신호 강도"로 규정했는가
- **Q3.** (i) 비용 구조 1:K를 서술했는가 (ii) V4 ratio 0.1198≈0.125 인용했는가 (iii) σ=0 동치·σ>0 비증가를 근거로 썼는가 (iv) ImageNet +1.3→+3.5를 인용했는가
- **Q4.** (i) 대조군( RN50+fully supervised 로지스틱 회귀)을 정확히 규정했는가 (ii) 0 예제 승리의 의미를 설명했는가 (iii) 약점 6셋을 "말로 전달 불가" 원인과 엮었는가 (iv) STL10 99.3% 케이스를 언급했는가
- **Q5.** (i) 44× log-log linear 근거를 인용했는가 (ii) ~1000×의 조건(SOTA 도달)을 구분했는가 (iii) 12.8B·405년 계산을 보였는가 (iv) MNIST/raw-pixel 반례의 의미를 서술했는가

## 🔬 Feynman 설명

<!-- 답변 후 준비 -->

## ❓ 의문점 / 헷갈린 것

<!-- 답변 후 기록 -->

## ✅ 채점 & 교정 (선생님이 채움)

<!-- 답변 후 -->
