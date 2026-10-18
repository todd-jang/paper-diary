# Day 04 — DeepSeek-V2: MLA와 DeepSeekMoE (효율의 재설계)

- **DeepSeek-V2** — DeepSeek-AI · 2024 · arXiv:**2405.04434**
- **한 줄:** 어텐션을 **MLA(Multi-head Latent Attention, 저랭크 KV 결합 압축 + decoupled RoPE)** 로, FFN을 **DeepSeekMoE(fine-grained + shared expert)** 로 교체해 **236B 총 / 21B 활성**으로 8.1T 토큰을 학습 — **KV cache −93.3%, throughput ×5.76, 학습비 −42.5%**, 128K 문맥.

> Day 02–03은 **"표현을 어떻게 배우나"**(MIM/자기증류)를 다뤘다면, Day 04는 축을 바꾼다 — **"같은 성능을 어떻게 훨씬 싸게 돌리나"**.
> 관통 질문: **"128K 문맥에서 KV cache를 어떻게 98% 줄이면서 성능은 오히려 올리나?"**

---

## 🧪 심화 산출물 (day04_deepseek_v2/)
- 📜 [`01_코드워크스루.md`](./day04_deepseek_v2/01_코드워크스루.md) — MLA(저랭크 KV 압축·흡수·decoupled RoPE) + DeepSeekMoE 라인바이런
- 📐 [`02_수식유도.md`](./day04_deepseek_v2/02_수식유도.md) — 식 (9)–(22): 압축, 흡수 항등식 `qᵀk=(c^Q)ᵀA_i c^KV`, decoupled RoPE, MoE 게이트
- 💻 [`03_실습.md`](./day04_deepseek_v2/03_실습.md) — `mla_demo.py`로 KV cache·흡수·RoPE 흡수파괴 **수치 검증**(MPS 실측)
- 📊 [`04_비교표.md`](./day04_deepseek_v2/04_비교표.md) — MHA/MQA/GQA/**MLA** + Dense/MoE/**DeepSeekMoE** 비교

---

## 📖 읽기 가이드 (염두에 두고 읽기)
- **핵심 대비 3가지**: ① KV 압축 방식(헤드 공유 GQA vs **저랭크 압축 MLA**) ② RoPE 위치(**decoupled 소채널**) ③ FFN(**잘게 + 공유**)
- MLA가 GQA와 다른 점은 "무엇을 줄이느냐"가 아니라 **"어떻게 줄이느냐"** — 왜 저랭크가 표현력 손실이 더 작은가?
- **흡수(absorption)** `A_i = W^UQ_iᵀ W^UK_i` 가 왜 "K,V를 계산조차 안 하게" 만드는가?
- RoPE를 `k^C`에 걸면 왜 흡수가 **깨지는가**? (행렬곱의 **비가환성**과 연결)
- **decoupled RoPE**가 왜 `d_h^R=64`라는 **작은 별도 채널**로 충분한가?
- DeepSeekMoE의 **fine-grained**와 **shared expert**가 각각 무엇을 해결하나(중복 제거 / 조합 폭발)?

---

## 🎯 오늘의 Socratic Quest (읽고 답하세요)

**Q1.** MLA는 KV를 **저랭크 잠재벡터 `c_t^KV`** 로 압축한다(식 9). GQA가 **KV 헤드를 공유**해 cache를 줄이는 것과 비교할 때, MLA의 **저랭크 압축**이 **표현력 손실을 더 작게** 만드는 이유는 무엇인가? "헤드를 묶는 것"과 "공통 저랭크 공간으로 접는 것"의 차이로 설명하라.

**Q2.** **흡수(absorption)** `q_t^Cᵀ k_s^C = (c_t^Q)ᵀ (W^UQ_iᵀ W^UK_i) c_s^KV` 가 성립한다. 이 항등식이 왜 **"추론 시 K, V를 materialize하지 않아도 됨"** 을 함의하는가? (그리고 W^UV는 어디로 흡수되나?)

**Q3.** RoPE를 `k^C`에 그대로 걸면 왜 위 흡수가 **깨지는가**? `RoPE(k_s^C) = R_s·W^UK c_s^KV` 로 썼을 때 `A_i` 가운데에 무엇이 끼어들어 흡수를 막는지 서술하라.

**Q4.** **decoupled RoPE**는 위치정보를 `q^R, k^R`(각 `d_h^R=64`) **소채널에만** 싣는다. 이 설계가 **(i) 흡수는 유지**하면서 **(ii) 위치정보는 보존**하는 이유는? 왜 `d_h^R`가 `d_h`(=128)보다 작아도 되는가?

**Q5.** **DeepSeekMoE**의 두 아이디어 — **fine-grained expert segmentation**(잘게 쪼개기)과 **shared expert isolation**(공유 전문가 분리) — 이 각각 어떤 문제를 해결하는가? (힌트: "조합 폭발"과 "공통지식 중복")

**Q6 (심화).** MLA는 KV cache를 줄이면서 **MHA보다 성능이 낮지 않다**(논문). 이 결과가 **"표현력과 효율은 트레이드오프"** 라는 통념에 던지는 반례는 무엇인가? KV의 **저랭크 구조**가 오히려 **정규화(regularization)처럼** 작동할 수 있다는 관점으로 설명하라.

---

## ✍️ 나의 답변 (본인이 채우세요)
- Q1.
- Q2.
- Q3.
- Q4.
- Q5.
- Q6.

---

## 🔬 Feynman 설명 (선생님께 설명하듯, 본인이 작성)
> MLA와 DeepSeekMoE가 무엇이고 왜 "더 싸게 더 강한가"를, 모르는 사람에게 5문장 이내로 설명한다면?

**(학생):**
1.
2.
3.
4.
5.

**선생님 모범 5문장:** (Quest 진행 후 채움)

---

## ❓ 의문점 / 헷갈린 것 (본인이 작성)



---

## ✅ 채점 & 교정 (선생님이 채움)

(Quest 진행 후 채움)

---

## 🧩 핵심 메모 (정답 요약 — 답변 후 대조용)

**① MLA = MHA + low-rank KV joint compression**
- KV를 **공통 저랭크 공간** `c_t^KV`(512)로 다운프로젝션(식9) → 업프로젝션으로 K,V 복원(식10-11).
- GQA(헤드 공유)와 달리 **헤드별 표현력 유지** → cache↓ + 성능↑.
- cache/토큰 = `(d_c + d_h^R)·l = 576×60 = 34,560` (MHA `1,966,080` 대비 **−98.2%**; 논문 headline −93.3%는 DeepSeek 67B 대비).

**② 흡수(absorption)**
- `A_i = W^UQ_iᵀ W^UK_i` 를 미리 접으면 `qᵀk = (c^Q)ᵀ A_i c^KV` → **K,V 미계산**.
- `W^UV`는 출력 측 `W^O`로 흡수. 실측 항등식 오차 `~1e-11`.

**③ decoupled RoPE**
- `k^C`에 RoPE 걸면 `R_s`가 `A_i` 가운데 끼어 흡수 불가(실측 오차 `4.6e-05` ≫ 0).
- 해법: RoPE를 **별도 `q^R,k^R`(`d_h^R=64`)** 에만 → C-부분 흡수 유지(실측 일치 `~3e-11`).
- `q=[q^C;q^R]`, `k=[k^C;k^R]`, 스코어 = `q^Cᵀk^C + q^Rᵀk^R`.

**④ DeepSeekMoE**
- `h' = u + Σ N_s shared + Σ g_{i,t} routed`, `g=Top-K(Softmax(uᵀe_i))`.
- **fine-grained** → 조합 폭발(표현력↑, 같은 활성 파라미터). **shared** → 공통지식 전담(라우팅 중복 제거).
- V2: N_s=2, N_r=160, K_r=6, expert hidden 1536 → **236B/21B**.

**⑤ 대표 수치**
- KV cache **−93.3%**, throughput **×5.76**, 학습비 **−42.5%**, 8.1T 토큰, 128K 문맥, 236B/21B.

---

## 🏁 Day 4 종료 요약 (답변·채점 후 채움)

**Quest 결과:** (진행 중)

**다음:** Day 5 = **DeepSeek-V3** (2412.19437, MoE + FP8 + MTP)

> **Day 5 방침 (2026-10-18, 학생 확정):** 비교표는 **true ablation** 중심 — 같은 DeepSeek-V3 실험 환경에서 변수만 켜고 끄기: **aux-loss-free balancing on/off · MTP on/off · FP8 on/off · shared-expert 수 변화**. 도메인 교차 숫자 비교(vision ↔ NLP)는 금지. (도메인 내 비교표는 허용 — Day 3 vision SSL 5-way, Day 4 attention/MoE 계보처럼)
