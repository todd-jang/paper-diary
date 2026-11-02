# Day 15-16 · 05 · 심화 Socratic 12문항 — arXiv 2512.10942 v2 원문 대조 답변

> **원문**: arXiv [2512.10942 **v2**](https://arxiv.org/abs/2512.10942) (2025-12-11 제출, **2026-02-02 개정**), Meta FAIR + HKUST (Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, ..., Allen Bolourchi, Yann LeCun, Pascale Fung).
> **작성**: 2026-11-02~03 세션 대비. 심화 Socratic 질문(Part 1~3)의 전제 검증 → 논증 → 근거 수치 → 방어선.
> **독법**: 각 문항 = ① 전제 검증 (질문 전제와 원문 사실의 일치 여부) → ② 핵심 논증 → ③ 근거 수치 → ④ 방어선 (꼬리 질문 대응).

---

## ⚠️ 0. 전제 정정 요약 — 이 문서가 존재하는 이유

심화 질문 다수가 I-JEPA/V-JEPA 계열(비디오 미래 상태 예측 JEPA)의 프레임으로 작성되었으나, **VL-JEPA 논문 본문은 전제가 다르다**:

| # | 질문의 전제 | 원문의 사실 | 판정 |
|---|---|---|---|
| 1 | 텍스트 조건으로 **비디오의 미래 상태**를 예측하는 월드모델 | Predictor는 **타깃 *텍스트* 임베딩 Ŝ_Y**를 예측 (§1: "predicts continuous embeddings of the target texts") | **수정 필요** |
| 2 | Collapse 방지 = EMA 타깃 인코더 + L1/L2 | **양방향 InfoNCE**(정렬항+균일화항) 채택 (§2). EMA·frozen Y-encoder는 대안으로 언급 | **수정 필요** |
| 3 | 학습 중 시공간 마스킹 | VL-JEPA 학습에 마스킹 없음(전체 프레임 입력). 마스킹은 동결된 V-JEPA 2 인코더의 사전학습 레시피 | **수정 필요** |
| 4 | 텍스트 주입 = Cross-Attention | **Predictor 입력단에 쿼리 토큰 결합** + 양방향 어텐션(causal mask 비활성화, §3.1) | **확정** |
| 5 | World model 논증은 플래닝 실험 기반 | 세계모델 증거 3종: **WorldPrediction-WM(역동학)** 65.7% SoTA + **§4.4 Action Anticipation(신규)** EK100/COIN + 스트리밍 selective decoding | 재해석 필요 |

**v2 개정 주요 변동 (v1 대비, 이 문서의 수치는 모두 v2 기준):**

| 항목 | v1 | v2 |
|---|---|---|
| BASE 평균 분류 (8셋 zero-shot) | 46.4 | **52.5** (vs PE-Core-G 44.7) |
| BASE 평균 검색 Recall@1 (8셋) | 58.4 | **63.7** |
| SFT 평균 분류 | 70.7 | **75.4** |
| VQA GQA / TallyQA | 60.8 / 67.4 | **61.5 / 69.9** |
| POPE / POPEv2 | 84.2 / 82.2 | **85.7 / 86.3** |
| 비디오 사전학습 | PLM-Video-Auto, Ego4D, Action100M; 16프레임 joint | **Action100M 단독**; 8프레임×60k iter → 32프레임×10k iter; 2주→**4주** |
| SFT | 35k 스텝 × 배치 6k | **83k 스텝 × 배치 3,072** (~2.5일) |
| §4 구성 | — | **§4.4 Action Anticipation 신규** (EK100 Anticipation + COIN 다음 스텝 예측) |

**아키텍처 확정 사실 (§3.1)**: X-Encoder = 동결 V-JEPA 2 ViT-L (304M, 256² 해상도, 이미지는 복제해 16프레임 형태 주입) · Predictor = Llama-3.2-1B 마지막 8층 초기화 (490M 학습가능, 쿼리 512토큰 상한, `[PAD]` 패딩, causal mask 비활성화, non-PAD 토큰 평균풀링) · Y-Encoder = EmbeddingGemma-300M 초기화 (LR multiplier ×0.05) · 양측 linear projection 뒤 **공유 임베딩 공간 1,536차원**에서 loss 산출.

---

# Part 1. 아키텍처 및 손실 함수 메커니즘

## [Q1-1] 바늘에 실 꿰기 같은 미세 조작에서 "고주파"가 핵심 시그널이면 어떻게 대응하는가?

**① 전제 검증.** 질문은 VL-JEPA가 액션조건부 *비디오 상태 예측* 모델이라는 전제를 담고 있다. 원문은 그렇지 않다: VL-JEPA는 텍스트 임베딩을 예측하며, 고해상도 인지는 이미 동결된 X-Encoder(V-JEPA 2 ViT-L)에서 수행된 뒤 토큰 시퀀스로 Predictor에 들어간다 (§3.1).

**② 핵심 논증 — 인지/서술의 계층 분리.**

1. **고주파 정보는 폐기되지 않고 보존 위치가 이동했다.** "픽셀을 버린다"의 정확한 의미는 *사후에 인지·제어에 사용될 표현*과 *서술(narrate)용 최종 출력*을 나누는 것이다. X-Encoder는 "이미지·비디오 양쪽 모두에서 탁월한 자기지도 비전 모델"(§3.1 원문)이며, 출력은 비압축 토큰 시퀀스(visual tokens) 전체가 Predictor로 전달된다 — 고주파 정보의 책임은 인코더 선(先)에 완성되어 있다.
2. **잠재 표현이 정보를 잃었는지는 실측으로 확인 가능**: §4.6 selective decoding 평가에서 segment average pooling이 "denoising and stabilization"을 제공하며 **uniform/adaptive 양쪽 전략 모두에 이득** — 즉 스트림 잠재 표헌에 고주파 노이즈가 존재하며, 결론적으로 표현은 정보를 *남기고* 있음(정주 vs 유실의 반대 방향 증거).
3. **범위 교정**: 서보 수준의 모터 제어는 VL-JEPA의 주장 범위 밖이다. 관련 연구(§5)는 액션조건부 predictor 계열을 별도 계열로 구분하며 "mazes or robotic pick-and-place 등 협소 도메인 제한"을 스스로 명시. 본 논문의 역동학 증거는 WorldPrediction-WM(전이 설명 동작 선택)이며, v2 신규 §4.4 Action Anticipation도 **의미 수준의 다음-동작 예측**(EK100, COIN)이다. 참고로 같은 인코더에서 V-JEPA 2 ViT-L-256px anticipation Recall@5(t=1s) 32.7에 대비 VL-JEPA 34.2 — 의미 조건화가 미세 동작 예상에서도 손해가 아님.

**④ 방어선.** "고주파가 핵심이면 JEPA가 무너진다"는 환원 오류: 질문이 *semantic 서술 모듈*에 *motor policy*의 요구를 던진다. 정교한 제어는 액션조건부 Predictor 모듈(V-JEPA-IC 계열)이 담당하며, 텍스트 임베딩 타입에 두 가지 책임을 부과하는 설계가 오히려 원문이 피하는 편향.

## [Q1-2] MAE가 고주파 노이즈에 용량을 낭비한다는 주장은 Linear Probing 격차로 증명되는가?

**① 전제 검증.** 질문의 비교 대상(MAE)은 본 논문과 다른 논문 — 증거축을 2개로 나눠 답한다: (A) MAE 논문 내부 증거의 해석 한계, (B) VL-JEPA 논문이 직접 공급하는 통제 실험.

**② 핵심 논증.**

1. **Linear probing 격차만으로는 불충분**: MAE는 fine-tuning에서 강하고 검출·분할 등 밀집 밀도(dense) 과제에서 오히려 좋은 비대칭을 보인다. Bordes et al. (2024) Guillotine regularization 계열의 해석에 따르면 이는 "고주파 정보 유실"이 아니라 **일반(abstract) 정보가 상위 계층에 고립되는 전이 프로토콜의 측정 혼동**이다. 즉 "용량 낭비"의 증명으로는 미달.
2. **통제 실험 축 (원문 주 증거, §4.5)**: 동일 동결 vision encoder(Perception Encoder ViT-L-14, 336², no tiling, 16프레임), 동일 데이터·배치(128)·학습률 스케줄. 유일한 차이 = **0.5B predictor의 잠재 공간 예측** vs **1B LLM의 token-space next-token prediction**.
   - 500K 샘플: 동률 — VL-JEPA CIDEr **1.23** vs VLM **1.35**, top-5 **14.9%** vs **14.0%**.
   - 15M 샘플: VL-JEPA **14.8 CIDEr / 41.0% top-5** vs VLM **7.1 / 27.2%** — 표본 효율과 절대 성능 모두에서 격차가 같은 방향.
3. **매개변수 축**: 동일 데이터·인코더에서 token-space 대비 학습가능 파라미터 **50% 이하**, 더 강한 성능 (§1 abstract).
4. **기제의 정보이론적 정식화**: 토큰 공간에서는 다른 plausible Y들이 one-hot으로 *직교*하지만 임베딩 공간에서는 근접 점으로 수렴(§2) — 타깃 분포의 분산 구조가 달라져 학습 신호의 엔트로피 자체가 감소한다. 이는 "고주파 노이즈"라는 은유보다 정확한 학술적 서술.

**④ 방어선.** "MAE가 낫다/나쁘다"가 아니다: MAE는 token-space 감독 안에서 올바른 설계다. 증명되는 것은 **타깃 공간 선택이 학습 효율의 인자로 직결**된다는 것이며, MAE linear-probing 격차는 불완전한 측도일 뿐.

## [Q2-1] X-Encoder가 언어를 몰라도 진정한 "비전-언어 정렬 표현"이라 할 수 있는가?

**① 전제 검증.** 사실이다: X-Encoder(동결 V-JEPA 2)는 언어 신호에 노출되지 않는다. 전제는 정확하다.

**② 핵심 논증 — 정렬 본체의 위치.**

1. **정렬 계약의 파라미터화**: 정렬은 (a) Predictor의 수렴 목표(Y-Encoder 생성 타깃 S_Y), (b) 양측 linear projection 뒤 **1,536차원 공유 공간**(§3.1)에서 성립한다. 비전-언어 정렬은 인코더 파라미터 교란이 아니라 **출력 공간의 대칭 제약**으로 구현된다.
2. **실측 증명 (인코더 blindness와 무관하게 정렬이 성립)**: BASE의 retrieval 순위 연산은 Ŝ_Y(captioning 프롬프트 조건)와 텍스트 쿼리 임베딩의 cosine 유사도 기준 — 8셋 평균 Recall@1 **63.7** vs PE-Core-G 58.1 (v2). 인코더가 language-blind이어도 **Predictor가 신호 경로를 완성**한다.
3. **동결의 방어적 가치**: 인코더를 언어 감독에 노출시키면 catastrophic interference 리스크가 생긴다. 동결은 이를 원천 차단하며, §5 world-model 계열(Zhou 2025 / Baldassarre 2025 / Assran 2025 / Terver 2025) 전반이 "frozen encoders + action-conditioned predictors"를 공용하는 설계 관행과 부합 — **같은 동결 X-Encoder를 액션조건부 모듈과 공유**할 수 있는 호환성 자산.
4. **Y-Encoder 쪽 정렬 품질은 독립 계측** (§4.7, v2 Table 6): SugarCrepe++/VISLA hard-negative triplet accuracy — **BASE 평균 49.8 / 42.9** vs CLIP ViT-L 37.6/31.3, SigLIP2 ViT-g 48.7/32.1, PE-Core-G 45.2/31.4. "VL-JEPA BASE has a Y-Encoder that is more resilient to text hard-negatives."

**④ 방어선.** "인코더는 정렬이 아니다"를 긍정한 뒤: "정렬 본체는 공유 공간 + Predictor이며, 인코더는 교란 없이 재사용 가능한 센서 전단" — 시스템 관점의 정렬 정의. motion-centric 벤치마크(BASE 우수) 실측으로 보강 (SSv2 16.1 등).

## [Q2-2] Predictor 출력이 특정 텍스트 조건에서 붕괴하거나 Mode averaging으로 수렴하는 것을 어떻게 방지하는가?

**① 전제 검증.** 유효한 질문. "What will happen here if I flip this light switch down?"에 대해 "the lamp is turned off"와 "room will go dark" 둘 다 정답이라는 예시는 §2에 원문이 직접 실어 놓은 논증.

**② 핵심 논증.**

1. **목표 분포의 근원 재형성**: 토큰 공간에서 다른 plausible Y들은 *직교*하지만 Y-Encoder가 이웃 점으로 모아 "compact unimodal distribution"을 만든다(§2) — **mode averaging 문제는 후처리가 아니라 학습 대상 자체를 단봉화함으로써 먼저 처리**된다.
2. **InfoNCE의 균일화 항**: InfoNCE = 정렬항 + 균일화항 (Wang & Isola 2020). 배치 내 네거티브 repulsion이 predictor 출력의 point/mean 수렴을 분포 수준에서 벌한다. 사전학습 이미지 국면 배치 **24k**가 이 정규화의 물리적 지원.
3. **추론 프로토콜의 이산화**: 생성형 샘플링이 아니라 **discriminative 프로토콜** — VQA는 후보 답변을 Y-Encoder로 인코딩해 최근접을 선택(§2 "Multi-tasking", §4.2). 이산 모드(yes/no 등)는 deterministic mean-pooled 예측이라도 후보 비교로 분리된다.
4. **잔여 리스크의 정직한 인정**: open-ended generation에서 conditional mean에 수렴하는 성향(다양성 저하)은 완전히 제거되지 않았다. 원문은 sampling 기반 다양성 평가를 수행하지 않았다 — 논문의 미비로 명시할 것.

**④ 방어선.** "해결"이 아니라 "재정식화" — mode 문제는 InfoNCE 균일화 + 큰 배치 + Y-encoder의 단봉 재형성 + discriminative 추론 프로토콜의 4중 안배로 접근.

---

# Part 2. 비대칭성 및 최적화 동역학

## [Q3-1] Predictor를 깊게/Encoder를 얕게 만들면 수치적으로 어떤 역효과가 나타나는가?

**① 전제 검증.** I-JEPA의 교훈("deep predictor가 shortcut 흡수")은 **양쪽 인코더도 학습가능하고 anti-collapse가 순수 구조적(stop-grad+EMA)일 때** 유효하다. VL-JEPA는 X-Encoder 동결 + anti-collapse를 손실 함수(InfoNCE)에 둔다 — 전제 이동 필요.

**② 핵심 논증과 원문 수치 (§4.8 Ablation).**

- **Predictor 층 수는 "많을수록 좋다"**: Layer 8-16 → 27.3 분류 / 30.2 검색 / 42.5 VQA vs **Layer 0-16 (전체 16층)** → 27.4 / 31.0 / **45.5**. 저자 서술: "more layers yield better performance, especially on VQA". **동결 인코더 + InfoNCE 환경에서 deep predictor 역효과는 관찰되지 않는다** — collapse/shortcut을 수용할 축이 손실 함수 쪽에 있기 때문.
- 관찰된 실패 모드는 층 수가 아니라 **어텐션 위상**: causal mask 유지 시 VQA **-1.9** — 쿼리 토큰이 시각 토큰 뒤에 붙으면 비전 토큰이 쿼리를 볼 수 없다. §3.1의 causal mask 비활성화가 이에 대한 직접 배려다.
- 초기화 trade-off (ablation (d)): random init 대비 **Llama-3 초기화는 VQA에 유익**, 분류/검색은 약간 하락(28.1 분류 +0.8, VQA 40.6 -1.9) — *예측 과제의 언어 의존성*에 따라 LM 초기화가 유리.

**④ 방어선.** "수치적 역효과"를 요구받으면 §4.8은 오히려 반대 방향(더 얕은 predictor → 하락, 42.5→45.5)을 보여준다. I-JEPA 교훈의 유효 조건(양측 학습 + 구조적 anti-collapse)을 명시하고 VL-JEPA는 이 조건 하에 있지 않음을 논증하는 것이 정답의 형태.

## [Q3-2] VL-JEPA의 Time-axis 마스킹 비율은 몇 %이며, 큰 마스킹이 비디오 연속성을 해치지 않는가?

**① 전제 검증.** 마스킹은 **VL-JEPA 학습 루프에 존재하지 않는다** — 쿼리 없는 사전학습과 SFT 모두 전체 프레임 입력. 마스킹은 동결된 X-Encoder의 *upstream recipe*(V-JEPA 2 사전학습) 소관이며 본문에 비율이 명시되지 않는다.

**② 핵심 논증.**

1. **책임 이동**: "비디오 연속성" 논증은 V-JEPA 2 사전학습 설계의 문제로 회부된다. VL-JEPA 본문이 관여하는 시간축 설계는 v2 §3.2의 **증분 노출 커리큘럼**(8프레임×60k iter → 32프레임×10k iter)뿐 — 전체 프레임 기반이다.
2. **동결의 철학**: §5 — "pretrained encoders are frozen", world-model 계열 전반이 동결 인코더 + 별도 predictor로 통일된 관행과 부합. upstream recipe는 고정 자산이다.
3. **v2 신규 시간축 실측 (§4.4)**: EK100 anticipation Recall@5가 t=1→10초에 걸쳐 34.2→11.7로 감소 — 마스킹 유무와 무관하게 **시간 갭 자체의 성능 감쇠**가 VL-JEPA의 직접 관측 대상이며, 같은 인코더의 V-JEPA 2 논문 수치 장기 갭(32.7/23.4/15.9/7.1)보다 모든 갭에서 우수.

**④ 방어선.** "몇 %냐"에 정답은 "VL-JEPA 학습 자체에 마스킹이 없다"(원문). 질문 전제가 I-JEPA framing의 관점 편향임을 학술적으로 교정.

## [Q4-1] EMA τ=1 고정(동결 타깃)이어도 비붕괴 학습이 가능한가? Random projection 타깃과의 차이는?

**① 전제 검증.** τ 논정은 이 논문에 부적합 — **EMA를 쓰지 않는다**. §2 원문: "정규화 항은 EMA(Assran 2025)나 frozen Y-Encoder(Zhou 2025) 같은 다른 anti-collapse 전략으로 대체 가능"하며 본 연구는 **양방향 InfoNCE로 Predictor·Y-Encoder joint-training**. 다만 질문의 본질(동결 타깃 + 비붕괴)은 ablation이 직접 다룬다.

**② 핵심 논증.**

1. **동결 타깃 + InfoNCE는 비붕괴 가능**: ablation (c) — InfoNCE 23.3 분류/30.3 검색/44.3 VQA vs cosine 16.5/20.2/**46.6**, L1 14.8/15.5/41.9, L2 13.5/11.7/43.7. 저자 서술: "**only InfoNCE has the anti-collapse regularization** and can be applied with unfrozen Y-Encoder". 즉 *stabilizer 축(EMA/동결)*과 *regularizer 축(InfoNCE)*은 독립된 두 축이며, 동결 시 비붕괴 방어는 균일화 항에 위임된다.
2. **Random projection과의 본질적 차이**: random target은 비붕괴만 만족하지만 **의미적 neighborhood 기하를 잃는다** — 다양한 plausible Y를 근접 점으로 모아 타깃 분포를 단일화한다는 §2의 표본 효율성 논거는 Y-encoder의 *의미적 cluster 기하*를 요구한다. ablation (e)가 파장을 직접 계측: Y-encoder 교체 시 PE-Core-G → 분류 **+14.4** (29.4 vs 19.5). 타깃의 의미 기하가 분류/검색 성능에 직행.
3. **실 레시피는 hard-freeze도 full-update도 아닌 **LR multiplier ×0.05** (ablation (b): sweet spot 0.05~0.10, 빠르게/느리게 둘 다 하락) — "타깃 안정성 vs 의미적 적합"의 튜닝된 조화점.

**④ 방어선.** Random projection = 상수해 방지의 좁은 축만 해결. 의미적 근린 거리 저장 구조는 Y-encoder의 기하이며, 교체 실측(±10pt급)으로 그 영향이 정량화됨.

## [Q4-2] 명시적 정규화 없이 순수 L1/L2가 차원 붕괴(Dimensional collapse)를 완전히 배제할 수 있는가?

**① 전제 검증.** 원문은 이 가정을 채택하지 않았다. §2: "More advanced non-sample-contrastive regularization such as **VICReg, SIGReg**를 미래연구로 미룬다" — 저자 스스로 분포 정규화의 가치를 명시하며 InfoNCE가 그 자리를 채웠다.

**② 핵심 논증.**

1. **InfoNCE의 이론적 방어 범위**: InfoNCE = 정렬 + 균일화(Wang & Isola 2020). 균일화 항은 정규화 hypersphere 위 포인트 분산을 강제(rank-1 붕괴 방지에 유효)하나, **rank-k 차원 붕괴를 명시적으로 금지하지는 않는다** — 배크 rank에 비례하는 방향 지원으로 부분적으로만 완화.
2. **규모 조건**: 1,536차원 공유 공간 + 이미지 국면 배치 24k — in-batch 균일화의 Rank 지원이 크다. SFT 국면 배치 3,072로 상대적으로 축소됨.
3. **실측 구도 (ablation (c))**: InfoNCE가 cosine·L1·L2보다 전반적으로 우수(유일 예외: cosine VQA +2.3). 이는 *순수 거리 loss가 정규화를 결여해 열세임을 실측으로 입증*하는 동시에, VICReg/SIGReg와의 비교는 수행되지 않은 **명시적 한계**로 남아 있다. 답변의 선: "InfoNCE 수준의 분포 정규화는 최소 요건 — 더 높은 수준은 저자도 미개척으로 명시".

**④ 방어선.** "완전 배제"라는 전제 자체를 교정한 뒤 InfoNCE의 부분적 방어 + 저자의 미래연구 남김을 학술적 선(선) 상에서 정리.

---

# Part 3. 월드 모델 및 플래닝 한계

## [Q5-1] 잠재 롤아웃의 컴파운딩 에러(Compounding error)를 VL-JEPA는 어떻게 완화하는가?

**① 전제 검증 (v2에서 강화).** VL-JEPA는 비디오 잠재 상태를 롤아웃하지 않는다. 그러나 v2 신규 **§4.4 Action Anticipation**이 *미래 동작 예상*의 직접 실측을 제공한다.

**② 핵심 논증.**

1. **제공되는 두 종류의 세계모델 평가**:
   - (a) **WorldPrediction-WM (§4.3)**: 초기/최종 상태 이미지 쌍 → 후보 action 클립 4개 중 전이를 설명하는 것 선택 — **역동학(inverse dynamics)**, 단일 스텝. 63.9(BASE)/**65.7%(SFT)** SoTA (GPT-4o/Claude-3.5/Gemini-2 초과).
   - (b) **Action Anticipation (§4.4, v2 신규)**: EK100 Recall@5: t=1s **34.2** / t=2s 26.0 / t=4s 19.4 / t=10s 11.7 (동일 인코더 V-JEPA2 ViT-L-256px 대비 32.7/23.4/15.9/7.1 — **모든 시간 갭에서 우월**; ViT-g-384px(39.7) 대비는 t=1s 아래, 긴 갭(t=4s,10s)에서 역전: 19.4 vs 19.3, **11.7 vs 8.2**). COIN 다음-스텝 예측 SoTA **56.2** (VideoLLM-online 49.1, VideoLLM-MoD 49.7, ProVideLLM 53.6).
2. **컴파운딩 에러 완화 장치의 3층**: (i) *이산 의미 후보 조건의 단일-스텝 예측* — multi-step 롤아웃 구조 자체를 회피; (ii) **selective decoding** (§4.6): Ward-distance 클러스터 기반 시점 선택, 0.35Hz에서 uniform 1Hz 등가 → **2.85× 절감** — 장시간 스트리밍 잠재 스트림의 신뢰 가능한 단조 운영; (iii) **segment average pooling** "denoising and stabilization" — 스트림 노이즈 상쇄.
3. **정직한 한계**: 저자 서술(§4.4 결과문) — "VL-JEPA can be finetuned to handle **semantic uncertain prediction tasks**". *연속 상태의 다단계 롤아웃*은 본 논문 범위 밖이며 §5의 action-conditioned predictor 계열(Zhou/Baldassarre/Assran/Terver 2025)은 "narrow domains like mazes or robotic pick-and-place" 제한을 저자 스스로 명시.

**④ 방어선.** "compounding error 완화"는 실제로 *(i) 이산 의미 후보 예측으로 회피 + (ii) 스트리밍 표현 denoising(평균풀링, selective decoding) + (iii) 프로그램 수준에서 H-JEPA(계층화, time-scale별 JEPA)로 연기* — "해결했다"고 주장하는 것만큼 학술적으로 위험한 서술.

## [Q5-2] 연속 제어값(토크, 6-DoF)과 이산 자연어 명령 간 표현 간극을 어떻게 좁히는가?

**① 전제 검증.** VL-JEPA의 컨디셔닝 입력은 텍스트 쿼리 토큰(512 상한) — 연속 제어값 인터페이스는 **논문에 존재하지 않는다**. v2 §4.4의 액션도 이산 동작 라벨(EK100 verb 후보, COIN 스텝 라벨) 예측이다.

**② 핵심 논증.**

1. **프로그램 수준 해법**: LeCun 프로그램의 표준 해법은 **계층적 H-JEPA** — 텍스트는 상위 부분목표(subgoal) 지시, 연속 제어는 하위 액션조건부 시각 JEPA(V-JEPA-IC 계열)가 처리. VL-JEPA의 텍스트 임베딩 스트림은 상-하위 인터페이스의 후보이며 selective decoding과 궁합이 좋다.
2. **아키텍처 확장 경로** (논문 외 논증, 그다지 명시한 대로): (a) 연속 액션 → 이산 토큰화 후 쿼리 토큰 열 삽입 — §3.1의 512토큰 유연 구조 그대로 수용 가능; (b) 액션 임베딩 스트림을 predictor 입력에 결합(causal mask 비활성화 필요); (c) WorldPrediction-WM의 방식과 동일하게 액션 후보를 Y-encoder로 인코딩하는 **액션 공간 동일 파라미터화**.
3. **현상의 정직한 인정**: 본 논문은 연속 제어 없이 순수 텍스트 컨디셔닝만 평가했다 — 간극은 남아 있으며, "무대응"이 아니라 **다른 모듈 계열과의 조합 설계로 위임**된 것이 원문 관행의 암묵적 해법.

**④ 방어선.** "표현 간극"은 본 논문 단독 문제가 아니라 시스템 수준 문제로 접을 볼 것 — 논문을 *단일 자율 모델*이 아닌 *모듈 시리즈의 일부*로 읽는 것이 원문 §5의 정통 해석.

## [Q5-3] 잠재 경로에 대한 보상 함수(Reward function)의 원천 — 순환 논리에 빠지지 않는가?

**② 핵심 논증.** 비판(자기 인증 순환)은 정당하나, 원문의 실제 설계는 외부 감독에 의존한다:

1. **본 논문의 "보상" = 외부 공급**: 모든 평가(분류/검색/VQA/WP-WM/EK100/COIN)는 *후보 집합 + 정답 라벨*이 곧 보상이다. Y-encoder 공유 공간의 cosine 최근접 선택은 학습된 보상 모델이 아니라 **판별 평가 protocol의 일부**(§4.2: "encode candidate answers... selecting the answer that minimizes the distance"). 순환은 *이 범위에서* 발생하지 않는다 — SFT의 25M VQA 사람 라벨이 최종 감독 원천.
2. **개방형 플래닝으로 확장하면 비판이 유효해진다**: 원문은 open-ended RL 인센티브 인터페이스 없이 "predictor stream 정량 모니터링 + 필요시 디코딩(2.85×)"으로 *인간 검증 비용*을 파형 저감하는 절차를 제시 — *인간이 순환 안에 남아* 있는 의존 설계다. 이 점을 정직하게 명시하는 것이 답변의 등급 차이.
3. **원칙적 대안 (LeCun 2022 프로그램)**: (i) goal-conditioned 텍스트 임베딩과의 거리 = *distance-to-goal energy*; (ii) 외부 감독으로 학습된 critic — 순환은 critic 학습 신호가 전적으로 내부 생성에 의존할 때만 닫힌다.

**④ 방어선.** "순환 부정"이 아니라 "본 논문의 보상은 외부 공급 + 검증 가능 후보 집합"; 순환 비판의 유효 범위(open-ended RL)와 원문의 상관 없음을 정확히 분리.

## [Q5-4] Sora급 생성 모델이 픽셀 예측의 일관성을 극복한다면 JEPA의 이론적 비교우위는 유지되는가?

**② 핵심 논증.**

1. **토큰-생성 병목은 데이터 규모에 무관한 구조적 성질**: (i) §2의 타깃 분포 분리 논증(직교 token → 근접 embedding)은 *토큰 통계의 성질*로 데이터량이 흡수하지 못한다; (ii) selective decoding은 *자기회귀 모델은 지원 불가, 외부 메모리/KV-cache 최적화가 필요*한 구조적 우위(§2 원문). 통제 실험의 샘플 효율 격차(15M: 14.8 vs 7.1 CIDEr, 41.0% vs 27.2%)는 동일 데이터 하 직접 실측.
2. **v2 신규 지연시간 논증 (§4.5 끝)**: latency 비교 실험에서 **두 모델의 텍스트 생성 지연은 비슷** — 차이는 **프롬프트 인코딩/Encoder+Predictor와 텍스트 디코더의 decoupling** 구조. 모델 전면부만으로 retrieval·모니터링 수행 가능 → online video inference의 확장성이 생성형과 다르다. "픽셀 예측의 일관성"과 무관한 **구조적 우위**.
3. **교체 계층 성능 종합 (v2)**: 학습 파라미터 50% 이하 + selective decoding 2.85× + 1.6B 규모로 CLIP/SigLIP2/PE-Core 평균 초과 (분류 52.5 / 검색 63.7) + InstructBLIP/Qwen-VL급 VQA (61.5/69.9/85.7/86.3) + WP-WM SoTA 65.7% + COIN anticipation SoTA 56.2%.
4. **정직한 수론**: 생성 진영이 *reward-on-rollout*(자체 시뮬레이션으로 판단)을 성공적으로 도입하면 단기 우위는 상대 평가의 달관으로 이동 가능. LeCun의 주장은 "생성은 쓸모없다"가 아니라 **지능 예산의 배분 축**(예측 대상의 추상 수준 최적화)이며, 본 논문의 기여는 그 원리를 일반 VL 과제에 최초 실측한 것.

---

## 📊 사후 요약 — 12문항 전제 점검 대시보드

| 문항 | 전제와 원문 일치? | 답변 요지 |
|---|---|---|
| Q1-1 | △ (모듈 범위 오판) | 고주파 인지는 X-Encoder(V-JEPA 2) 소관; 본 논문은 semantic 서술 계층 |
| Q1-2 | ○ (증거 축 한계 지적) | §4.5 통제 실험 + 50% 파라미터; linear probing은 불완전 측도 |
| Q2-1 | ○ (사실 명시적) | 정렬 본체 = 1,536d 공유 공간 + Predictor; Y-encoder hard-negative 강성 (49.8 avg) |
| Q2-2 | ○ | InfoNCE 균일화 + 배치 24k + Y-encoder의 단봉 재형성 + discriminative 프로토콜 |
| Q3-1 | ✗ (조건 이동 필요) | 동결 인코더+InfoNCE에서 deep predictor 역효과 없음(§4.8: 16층이 최고) |
| Q3-2 | ✗ | 본 논문 학습에 마스킹 없음; 비율은 V-JEPA 2 upstream 이슈 |
| Q4-1 | ✗ (EMA 미사용) | InfoNCE 균일화로 동결/soft-freeze(×0.05) 가능; random target은 의미 기하 상실 (+14.4 실측) |
| Q4-2 | ✗ (전제 미채택) | InfoNCE로 rank-1 방지, rank-k는 미보장 — 저자도 VICReg/SIGReg 보류 |
| Q5-1 | △ | 역동학(65.7% SoTA)+anticipation(v2 신규, 56.2% SoTA) 단일-스텝; 다단계 롤아웃은 범위 밖 |
| Q5-2 | △ | 연속 제어 인터페이스 부재; H-WM 계층 분업이 프로그램 해법 |
| Q5-3 | ○ | 본 논문 보상 = 외부 공급(후보+라벨); 순환은 critic 내부폐쇄 시에만 정당한 비판 |
| Q5-4 | ○ | 토큰 직교성·selective decoding·decoder decoupling은 scale 무관 구조 우위 |

### 세션 전 확인 목록 — ✅ 2026-11-02 원문 교차 검증으로 전량 해소

- [x] **V-JEPA 2 인코더의 마스킹 비율**: V-JEPA 2 (arXiv 2506.09985) 본문은 비율을 재기술하지 않고 "동일한 multiblock masking strategy (Bardes et al., 2024)"로 회부. V-JEPA (arXiv 2404.08471) 원문의 확정값: **3D Multi-Block masking** — short-range(스페이셜 스케일 0.15, 8블록 union, 전체 시간축 튜브) + long-range(스페이셜 스케일 0.70, 2블록 union)의 멀티마스킹, **평균 마스킹 비율 ≈ 90%** (encoder 측). tubelet 2×16×16, 블록 종횡비 0.75~1.5.
- [x] **표 1 "Samples seen" 정의**: BASE 3.3B = 이미지 구간 2.4B(100k×24k) + 비디오 구간 ≈0.9B(8fr×60k + 32fr×10k); **SFT 3.6B = 누적** 기준(3.3B + 83k×3,072≈0.25B ≈ 3.55B→3.6B — §4.1 본문 "only 3.6B in comparison with PE-Core-G's 86B"과 부합). 표의 수치는 **누적 시행 샘플 수**로 해석.
- [x] **ablation (c) 캡션 해소**: v2 표 7(c)의 전체 캡션은 **"(c) Loss function (with no projection head on top frozen text encoder)"** — 4개 loss 로우 모두 *동결 텍스트 인코더 + 프로젝션 헤드 없음* 조건에서 비교. InfoNCE가 이 동결 조건조차 **cosine(-6.8 분류, -10.1 검색)·L1·L2를 앞서므로**, 균일화항의 이득이 동결 타깃에서도 별개로 실측된다. "(unfrozen 가능)" 서술은 §4.8 본문의 별도 주장.
- [x] **(신규 확보) ablation (b) LR multiplier 전체 행** — Q4-1의 직접 실측 근거: **multiplier=0.00(완전 동결)에서도 분류 20.0/검색 25.9로 비붕괴 성능 관측** (기본 ×0.05 대비 -7.3/-4.3 — 의미 적합 부재 비용), ×1.00도 하락(-3.6/-1.4/-1.8) → sweet spot 0.05~0.10.

> **출처**: arXiv 2512.10942 v2 (2026-02-02) §1–§5; **V-JEPA 2 (arXiv 2506.09985, §2.1·§2.2·§4)**, V-JEPA (arXiv 2404.08471, §3.2·Appendix C·Fig.8); Wang & Isola (2020); He et al. (2022) MAE; Assran et al. (2023) I-JEPA; Bordes et al. (2024) Guillotine Regularization; Zhou et al. (2025), Baldassarre et al. (2025), Terver et al. (2025); LeCun (2022).
