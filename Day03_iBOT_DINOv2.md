# Day 03 — iBOT & DINOv2: 온라인 토크나이저와 자기증류의 융합

- **iBOT** — Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, Tao Kong (ByteDance & JHU) · ICLR 2022 · arXiv:**2111.07832**
- **DINOv2** — Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, … , Patrick Labatut, Armand Joulin, Piotr Bojanowski (Meta AI) · TMLR 2024 · arXiv:**2304.07193**
- **한 줄:** iBOT = BEiT의 **고정 dVAE 토크나이저를 student의 EMA teacher(온라인 토크나이저)로 교체**해 가린 패치를 **soft 분포에 증류** / DINOv2 = **DINO+iBOT 손실 + KoLeo + Sinkhorn centering + 데이터 큐레이션(LVD-142M)** 을 스케일한 **범용 frozen 특징**.

> 두 논문은 **같은 질문**에서 출발한다 — "BEiT의 이산 토큰 타깃은 꼭 필요하며, 그 토크나이저를 따로 학습해야 하나?"
> iBOT는 **"토크나이저를 student 자신에서 만들어내면 된다"**(온라인·soft·1단계)고 답하고, DINOv2는 **"그 답을 DINO와 합치고 데이터로 밀면 범용 특징이 된다"**고 밀어붙인다.

---

## 🧪 심화 산출물 (day03_ibot_dinov2/)
- 📜 [`01_코드워크스루.md`](./day03_ibot_dinov2/01_코드워크스루.md) — iBOT(온라인 토크나이저+두 손실) / DINOv2(DINO+iBOT+KoLeo) 라인바이런 해설
- 📐 [`02_수식유도.md`](./day03_ibot_dinov2/02_수식유도.md) — MIM=KD 정식화, L[CLS]/L[MIM], EMA, Sinkhorn-Knopp, KoLeo 유도
- 💻 [`03_실습.md`](./day03_ibot_dinov2/03_실습.md) — iBOT 축소판 직접 구현·학습 (+ `ibot_demo.py`, MPS 실측)
- 📊 [`04_비교표.md`](./day03_ibot_dinov2/04_비교표.md) — MAE vs BEiT vs SimMIM vs **iBOT** vs **DINOv2** (5자 심층 비교)

---

## 📖 읽기 가이드 (염두에 두고 읽기)
- **핵심 대비 3가지**: ① 토크나이저(오프라인 dVAE vs **온라인 EMA teacher**) ② 타깃(hard 토큰 vs **soft 분포**) ③ 단계 수(2단계 vs **1단계**)
- iBOT의 teacher가 곧 **"온라인 토크나이저"** 라는 말의 정확한 의미는? (마스크 없는 뷰 → 패치별 확률분포)
- 왜 iBOT는 손실을 **두 개**(전역 `L[CLS]` + 국소 `L[MIM]`) 쓰나? 하나로는 왜 부족한가?
- `L_MIM`의 타깃이 **왜 soft 분포**여야 하나? hard argmax로 하면 무엇이 깨지나?
- **EMA + stop-grad**가 없으면 왜 붕괴(collapse)하는가?
- DINOv2는 왜 DINO와 iBOT을 **합쳤고**, 거기에 **KoLeo**와 **Sinkhorn-Knopp**을 더했나?
- **LVD-142M**(자기지도 검색으로 큐레이션한 142M)이 "더 많은 데이터"보다 이긴 이유(ImageNet-A 73.9 vs 59.4)?

---

## 🎯 오늘의 Socratic Quest (읽고 답하세요)

**Q1.** iBOT는 BEiT의 **오프라인 dVAE 토크나이저**를 **student의 EMA teacher(온라인 토크나이저)** 로 교체했다. 왜 이 교체가 가능하며, 이것이 **2단계 파이프라인을 1단계로** 만드는 이유는 무엇인가?

**Q2.** iBOT의 두 손실 **`L[CLS]`**(전역)와 **`L[MIM]`**(국소 패치)의 역할 분담은 무엇인가? 왜 **둘 다** 필요한가 — 하나만 쓰면 무엇이 부족한가?

**Q3.** `L_MIM`에서 teacher의 타깃을 **hard argmax(BEiT의 이산 토큰)가 아니라 soft 확률분포**로 쓰는 이유는 무엇인가? patch의 **의미 모호성(ambiguity)** 관점에서 설명하라.

**Q4.** teacher를 **EMA로 갱신 + stop-grad** 하는 것이 왜 붕괴(collapse) 방지에 결정적인가? teacher를 student와 같은 그래디언트로 함께 학습하면 왜 안 되는가?

**Q5.** DINOv2 = **DINO + iBOT + KoLeo + Sinkhorn-Knopp + (LVD-142M + 증류)**. 절제실험 결과(MIM 제거 → 분할 47.1→44.2 / KoLeo 제거 → 검색 63.9→55.6)를 근거로, **각 요소가 어느 능력을 담당**하는지 짝지어라.

**Q6 (심화).** DINOv2의 **LVD-142M 큐레이션**이 "그냥 더 많은 uncurated 데이터"를 **이겼다**(ImageNet-A 73.9 vs 59.4). 이 결과가 시사하는 바를 **"데이터의 양 vs 품질"** 관점에서 설명하고, 큐레이션 절차(중복제거 → 자기지도 임베딩 검색)를 한 문장으로 요약하라.

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
> iBOT이 BEiT와 무엇이 다른지, 그리고 DINOv2가 왜 강한지 모르는 사람에게 5문장 이내로 설명한다면?

**(학생):**
1.
2.
3.
4.
5.



---

## ❓ 의문점 / 헷갈린 것 (본인이 작성)



---

## ✅ 채점 & 교정 (선생님이 채움)

**Q1 (2026-10-18)** — 학생 답변 없이 **수업(모범답안 제시) 완료**, 자기말 재시도 대기.
- 핵심: BEiT의 2단계는 **(i) 이산 타깃(미분 불가)** + **(ii) 사전학습·고정 토크나이저** 라는 두 제약에서 *강제*된 것.
- iBOT는 두 제약을 동시에 제거 → **(i) soft 분포 타깃(미분 가능)** + **(ii) EMA teacher(자기생성·진화)**.
- 따라서 토크나이저 학습(① `log p(x|z)`)과 표현 학습(② `log p(z|x)`)이 **한 루프로 합쳐진다** → 1단계.
- `L[CLS]`가 teacher의 전역 의미를 떠받쳐 "정답지 품질"을 유지(품질 보증 장치).
- ↳ **재시도 (2026-10-18):** ① "왜 2단계가 강제되나" → **모름**(재교육) / ② "왜 1단계로 접히나" → **부분 정답 (½)**.
  - ② ✓ 정확한 절반: soft 분포 → logits의 매끄러운 함수 → **미분 가능**, `argmax` 소멸. (= **이산** 제약 해소)
  - ② ✗ 빠진 절반: 정답 생성기를 **외부 고정 모듈 → student의 EMA(`θ'`)** 로 바꿨다는 부분. 이게 "별도 사전학습·동결 불필요"를 만든다. 즉 1단계는 **두 제약(이산 + 고정 토크나이저)이 *동시에* 제거**될 때 성립.
  - ① 재교육(정답): 이산 타깃 = `argmax` = **미분 불가** → 정답 생성기(z 만들기)와 예측기를 **한 그래디언트로 묶을 수 없음** → ① dVAE `log p(x|z)` 먼저 학습·고정 → ② MIM `log p(z|x_visible)`. **"이산이라서"가 2단계의 원인.**
  - ⚠️ **반복 지점:** Day 02 봉합 포인트 #3("이산 z는 미분 불가 → 2단계 강제")과 **동일** — iBOT에서 또 막힘. 최우선 봉합 대상.

**Q2 (2026-10-18)** — 답변: "L만 쓰면 사실상 암기인데 MIM만 쓰면 문맥 픽셀 전체를 이해하고, 품질은 student가 한 것을 teacher가 보증?" → **부분 정답 (방향 ½, 교정 2건)** ✓
- ✓ 옳은 감각: MIM이 **국소+문맥**을 담당한다는 방향 ✓. 두 손실이 **서로 다른 축**을 담당한다는 큰 그림 ✓.
- ✗ 교정 ①: **"L[CLS]만 쓰면 사실상 암기"** → 부정확. L[CLS]만 = 사실상 **DINO**. 배우는 건 **전역(이미지 수준) 의미 + 뷰 불변성** → **분류엔 강, 검출/분할엔 약함**. "암기"가 아니라 **국소 감독 부재**가 문제. 빠진 것 = **패치 수준 / 공간 구조**.
- ✗ 교정 ② (**방향 반대, 감점 포인트**): **"품질을 teacher가 보증"** → 보증 관계가 뒤집힘. teacher는 student의 **EMA**라 student를 *따라갈* 뿐 **검증자가 아님**. 실제 구조: **teacher가 student에게 정답지(soft 타깃)를 준다** ← 그 **teacher의 품질을 `L[CLS]`가 보증**(전역 의미 앵커). 즉 "L[CLS]가 teacher를 의미 있게 유지 → teacher가 student에게 좋은 타깃 제공".
- 한 줄: **"L[CLS]=전역(앵커·분류) · L[MIM]=국소·문맥(분할/검출). MIM만 쓰면 타깃이 저수준으로 표류할 압력이 생기는데, CLS가 그걸 의미 있게 붙잡는다."**

> Q3–Q6 학생 답변 대기 중. 답변 제출 후 이 칸을 채웁니다.

---

## 🧩 핵심 메모 (정답 요약 — 답변 후 대조용)

**① iBOT = "온라인 토크나이저"**
- BEiT: **고정 dVAE**(DALL-E, 코드북 8192)로 하드 토큰 라벨 → **2단계**.
- iBOT: teacher(=student의 **EMA**)의 patch head = **online tokenizer**. 마스크 없는 뷰 → 패치별 **soft 분포**. **1단계**.
- 핵심: 토크나이저를 **따로 학습하지 않고** student 자신의 안정적 버전(teacher)으로 **계속 진화**시킨다.

**② 손실 = L[CLS] + L[MIM]**
- `L[CLS]` : 두 뷰 [CLS] 교차 자기증류(DINO식) → **전역 의미** + 토크나이저 품질 부트스트랩.
- `L[MIM]` : 가린 패치를 teacher soft 분포에 CE → **국소(패치) 의미** → dense 전이(검출/분할) 강함.

**③ soft 타깃의 이유**
- 패치는 경계·배경에서 **의미가 모호** → hard 라벨로 강제하면 노이즈. 분포로 맞히면 모호함을 반영.
- 정식화: **MIM = 지식증류(KD)**, teacher(`θ'`)를 online으로 두면 KD가 됨.

**④ EMA + stop-grad**
- `θ' ← λθ' + (1−λ)θ` (λ≈0.996), teacher 출력은 항상 **stop-grad**.
- teacher = **천천히 변하는 안정적 정답지** → 자기확증 붕괴 방지. (같이 그래디언트 학습하면 collapse.)

**⑤ DINOv2 = 융합 + 스케일**
- 손실: `L_DINO + L_iBOT + λ·L_Koleo`, centering을 **SwAV Sinkhorn-Knopp(3 iter)** 로.
- **KoLeo**(`−1/n Σ log d_n,i`) = 특징을 고르게 퍼뜨림 → **검색**(Oxford-M 55.6→63.9).
- **MIM** = **분할**(ADE-20k 44.2→47.1). **LVD-142M** = **강건성**(INet-A 73.9 vs 59.4).
- 대형 `ViT-g/14`(1B) → **증류**로 S/B/L. iBOT 대비 **~2× 빠르고 ⅓ 메모리**.

**⑥ 대표 성능**
- iBOT: INet-1K **82.3% linear / 87.8% fine-tune**.
- DINOv2: **frozen linear probe** S 81.1 / B 84.5 / L 86.3 / g **86.5**, INet-A(g) **75.9%**.

---

## 🏁 Day 3 종료 요약 (답변·채점 후 채움)

- **Quest 결과:** (대기)
- **🔁 봉합 포인트:** (채점 후 기입)
- **성취:** (채점 후 기입)
- **다음:** Day 4 = **DeepSeek-V2** (2405.04434, MLA)
