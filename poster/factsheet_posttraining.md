# LLM 사후강화학습 3편 검증 팩트시트 (2026-11-04 자율세션)

> **출처 검증 원칙**: 모든 수치·공식은 원문(arXiv HTML/ar5iv HTML)에서 직접 추출.
> InstructGPT: ar5iv 2203.02155 (NeurIPS 2022) · DPO: arXiv HTML 2305.18290v1 + ar5iv (NeurIPS 2023 Oral) · GRPO 원전: arXiv HTML 2402.03300v2 (DeepSeek-AI).
> 수치 인용 시 반드시 이 파일의 (식번호/표번호/절번호) 태그를 함께 기입.

---

## 1. InstructGPT — "Training language models to follow instructions with human feedback" (Ouyang et al., 2203.02155, NeurIPS 2022)

### 1.1 파이프라인 (§3.1, Fig. 2)
- 3단계: **① SFT(시연 데이터) → ② RM(비교 데이터) → ③ PPO(RM 대상 RL)**. Steps 2–3은 반복 가능(대부분 비교 데이터는 SFT 정책에서, 일부는 PPO 정책에서 수집).
- 배지 환경: "고객 프롬프트 하나 제시 → 응답 → RM 보상 → 종료" (§3.5 RL).
- 라벨러: **약 40명의 contractor** (§3.4). 교육 라벨러 상호 동의 **72.6±1.5%**, hold-out **77.3±1.3%**.

### 1.2 데이터 (§3.2)
- **SFT ~13k**, **RM 33k**, **PPO 31k** 학습 프롬프트 (Table 6 상세).
- 모델: GPT-3 아키텍처 **1.3B / 6B / 175B**.

### 1.3 학습 설정 (§3.5)
- SFT: **16 epochs**, 코사인 LR 감쇠, dropout 0.2. 1 epoch 이후 검증 손실 과적합이나 RM 점수/인간 선호는 향상.
- RM: **6B만 사용** — 175B RM은 학습 불안정 → RL value function으로 부적합 (부록 C).
  - 고정 셔플로 한 번 훑으면 과적합 → **프롬프트별 (K choose 2) 비교를 배치 1단위로 학습** (K=4..9).
- RM 손실 — **식 (1)**: `loss(θ) = −1/C(K,2) * E_{(x,y_w,y_l)~D}[ log σ( r_θ(x,y_w) − r_θ(x,y_l) ) ]`
- RL 목적식 — **식 (2)** (PPO-ptx):
  `objective(φ) = E_{(x,y)~D}[ r_θ(x,y) − β log( π_RL(y|x)/π_SFT(y|x) ) ] + γ E_{x~D_pretrain}[ log π_RL(x) ]`
  - "PPO"는 γ=0. 이 논문의 **"InstructGPT"는 PPO-ptx**를 지칭.
  - 보상에 **per-token KL 페널티**(SFT 대비)로 RM 과최적화 완화.

### 1.4 대표 수치 (§4 / 초록 / Fig. 1)
- **1.3B InstructGPT > 175B GPT-3** (선호율 — 100× 작아도).
- **175B InstructGPT vs 175B GPT-3: 85±3%**, vs few-shot GPT-3: **71±4%**.
- SFT 겨피선 대비 승률: InstructGPT **73.4±2%**, T0 26.8±2%, FLAN 29.8±2% (§1).
- 진실성: TruthfulQA에서 참·정보성 답변 **약 2×**. 폐쇄 도메인 환각율 **21% vs 41%** (절반 이하).
- 독성: RealToxicityPrompts "존중하라" 프롬프트 시 **~25% 감소**. Winogender·CrowS-Pairs는 유의미한 개선 없음.
- **Alignment tax**: SQuAD/DROP/HellaSwag/WMT15-FR 회귀 → **PPO-ptx가 회귀를 크게 줄임**.

### 1.5 인용 영향력 (Semantic Scholar, 2026-11-02 기준)
- 인용 **25,614** / 영향력 인용 **2,541**.

---

## 2. DPO — "Direct Preference Optimization" (Rafailov et al., 2305.18290, NeurIPS 2023 Oral)

### 2.1 아이디어 (§3–4)
- Bradley-Terry 선호모델 — **식 (1)**: `p*(y1≻y2|x) = exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2)))`
- RM 최대우도 손실 — **식 (2)**: `L_R = − E[ log σ( r_φ(x,y_w) − r_φ(x,y_l) ) ]`
- KL-제약 보상 최대화 — **식 (3)**: `max_{π_θ} E[r_φ(x,y)] − β D_KL[π_θ ‖ π_ref]`
- KL-제약 목적의 해석해(최적 정책) — **식 (4)**: `π_r(y|x) = (1/Z(x)) π_ref(y|x) exp( r(x,y)/β )`
- 재매개화 — **식 (5)**: `r(x,y) = β log( π_r(y|x)/π_ref(y|x) ) + β log Z(x)`
- BT에 대입하면 Z(x) 소멸 → 선호확률이 정책만으로 — **식 (6)**.
- **DPO 손실 — 식 (7)**:
  `L_DPO(π_θ; π_ref) = − E_{(x,y_w,y_l)~D}[ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`
- 기울기 해석: 암묵보상 `r̂_θ = β log(π_θ/π_ref)`이 **y_l을 y_w보다 높게 매기는(틀린) 예에 큰 가중**(σ항) → y_w 좋아요·y_l 싫어요. 이 가중 없는 단순 비율 목적은 TM 퇴화 유발 (부록 Table 2/3).
- **Theorem 1** (§5.1, 식 (8)): 등가 클래스 내 모든 보상은 `r(x,y) = β log(π(y|x)/π_ref(y|x))`로 표현 가능 — 일반성 손실 없음 ("당신의 LM은 몰래 보상모델").
- **식 (10)** (§5.2): PPO 손 배우자 알고리즘의 불안정 원인 — 정규화항 f(r_φ,π_ref,β)(참조정책의 soft value) 부재 시 정책기울기 분산 증가. DPO 재매개화는 baseline 불필요.

### 2.2 대표 수치 (§6.1–6.3, Fig. 2/3, Table 1)
- 감정 제어 reward-KL frontier: **DPO가 PPO-GT(정답 보상 접근 PPO)보다도 좋은 frontier** (Fig. 2 좌, 22회 런 스윕).
- TL;DR 요약 (GPT-J 6B, GPT-4 평가자): 승률 **T=0.0에서 ~61%** vs PPO 최적 온도 T=0.0의 **57%**. DPO는 β 튜닝 거의 없이, 온도 변화에 더 로부스트. Preferred-FT는 SFT 대비 유의미 개선 없음.
- 인간 직접평가(head-to-head): **DPO(T=0.25)가 PPO(T=0) 샘플을 58% 선호** (§6.2).
- Table 1 (인간 연구): 인간 승률 **DPO 58 / SFT 43 / PPO-1 17%**; GPT-4(S) 승률 47/27/13; GPT-4–인간 동의 70% vs 인간–인간 동의 65%.
- Anthropic-HH 1턴 대화 (Pythia-2.8B + Preferred-FT 참조): **표본이 개선된 유일한 연산 효율 방법**, Best of 128과 동등 이상. 외부 PPO 모델(reciprocate/ppo_hh_pythia-6B)은 베이스 Pythia-2.8B를 넘는 온도·프롬프트를 찾지 못함.
- 모델 크기: **최대 6B**까지 평가. β 스윕 {0.05, 0.1, 1, 5}.

### 2.3 인용 영향력 (Semantic Scholar, 2026-11-02 기준)
- 인용 **11,437** / 영향력 인용 **2,369**.

---

## 3. GRPO 원전 — "DeepSeekMath" §4.1 (Shao et al., 2402.03300v2)

### 3.1 PPO 출발점 (§4.1.1)
- PPO 목적 — **식 (1)**: 클립된 surrogate (GAE 기반 â²t, 학습 value V_ψ).
- 보상에 per-token KL — **식 (2)**: `r_t = r_φ(q,o≤t) − β log( π_θ(o_t|...)/π_ref(o_t|...) )` — InstructGPT(Ouyang et al. 2022) 방식 인용.
- **critic/value 모델이 정책과 맞먹는 크기** → 메모리·계산 부담 + RM은 마지막 토큰만 보상이라 토큰별 value 학습이 어려움 → Figure 4.

### 3.2 GRPO (§4.1.1, Fig. 4, Algorithm 1)
- **식 (3)**: 질문 q마다 그룹 {o_1..o_G} 샘플 → 클립 surrogate − **KL을 손실에 직접 추가**(보상에 섞지 않음):
  `J_GRPO(θ) = E[q, {o_i}~π_old] (1/G) Σ_i (1/|o_i|) Σ_t { min[ρ_i,t Â_i,t, clip(ρ_i,t, 1−ε, 1+ε) Â_i,t] − β D_KL[π_θ ‖ π_ref] }`
- KL 추정 — **식 (4)** (Schulman 2020, 항상 양의 부호): `D_KL = π_ref/π_θ − log(π_ref/π_θ) − 1`
- 결과보감(outcome) advantage: **Â_{i,t} = r̃_i = ( r_i − mean(r) ) / std(r)** — 그룹 표준화를 모든 토큰에 적용 (§4.1.2).
- 절차보감(§4.1.3): 스텝 끝마다 보상 → 같은 group mean/std 정규화.
- **Algorithm 1 (Iterative GRPO)**: I번 반복 — 매 반복 ref 갱신, M스텝, **μ번의 GRPO 업데이트**, RM은 replay로 계속 학습.

### 3.3 대표 수치 (§4.2 / abstract / Table 5)
- 사전학습: DeepSeek-Coder-Base-v1.5 7B + **120B math tokens** (Common Crawl, Minerva 대비 7×, OpenWebMath 대비 9×).
- Base: GSM8K **64.2%**, MATH **36.2%** → Instruct: 82.9 / 46.8 → **RL(GRPO): GSM8K 88.2 / MATH 51.7** (영어 서브셋만으로 RL).
- CMATH 84.6 → **88.8**. MATH 51.7%를 외부 도구·투표 없이 자율주행 SOTA 도달(Gemini-Ultra/GPT-4 접근), SC@64 **60.9%**.
- 통합 패러다임(§5.2.1): **RFT·DPO·PPO·GRPO 전부 "직접 또는 단순화된 RL"**이라는 통합 관점.

### 3.4 인용 영향력 (Semantic Scholar, 2026-11-02 기준)
- 인용 **10,603** / 영향력 인용 **3,104** (3편 중 최고 — 실무 구현 인용).

---

## 4. 3계열 품은 지도 (포스터 스토리 라인)

| | InstructGPT (2022) | DPO (2023) | GRPO (2024) |
|---|---|---|---|
| 전제 | RLHF의 원형: RM 학습 + PPO | 같은 목적을 **재매개화로 closed-form** | PPO는 살리고 **critic을 그룹통계로 대체** |
| 선호데이터 사용 | RM 학습(비교) → RM이 RL 보상 | **직접** 정책 손실로(오프라인) | RM 학습 → 그룹 보상 → advantage |
| 보상모델 | 명시적 (6B) | **암묵적**(정책=보상, Thm.1) | 명시적 (critic 없음) |
| 학습 시 샘플링 | 필요 (온라인 RL) | **불필요** | 필요 (그룹 G개) |
| KL 제약 위치 | 보상에 per-token penalty | 목적식 내 참조비율 (식7 내장) | **손실에 직접** (식 (3) 마지막항) |
| 황금 수치 | 1.3B > GPT-3 175B; 85±3% | TL;DR 61% vs 57%; 인간 58/43/17 | MATH 46.8→51.7, GSM8K 88.2 |

- 연결점(당신 일기): Day06 DeepSeek-R1 = GRPO의 대규모 확장(2501.12948) — 본 포스터는 **원전 DeepSeekMath의 GRPO 서론부**(공식·critic 제거 근거)를 다룸.
- 외부 맥락(원문 통합 아님): 이후 DAPO/RLOO 등 파생은 본 팩트시트 범위 외 — 인용 시 별도 검증 필요.
