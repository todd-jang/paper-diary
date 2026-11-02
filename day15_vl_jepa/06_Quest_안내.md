# 06 · 저녁 세션용 Quest 안내 (Day 15-16 VL-JEPA)

> **세션 일자**: 2026.11.02(월) ~ 11.03(화)  
> **진행 방식**: 티쳐 출제 Socratic 질문 → 학생 학술적(Academic) Feynman 답변 및 실측 근거 → 심층 꼬리 질문 방어  
> **✅ 2026-11-02 원문(v2) 대조 교정 완료**: 본 논문이 예측하는 것은 **타깃 텍스트 임베딩**(비디오 미래 상태 아님), collapse 방지는 **양방향 InfoNCE**(EMA 아님), 세계모델 증거는 WorldPrediction-WM 65.7% SoTA + v2 §4.4 Action Anticipation (EK100 34.2 / COIN 56.2 SoTA). 교정 근거·수치·방어선은 [`05_심화_Quest_12문항_답변.md`](05_심화_Quest_12문항_답변.md), 확정 스펙은 [`00_검증팩트시트.md`](00_검증팩트시트.md) 참조.

---

## 🧠 Socratic Quest & Academic Drafts (5문항)

---

### Q1 — 왜 생성(Generative)이 아니라 잠재 공간 예측(JEPA)인가?

#### 1. 문제 제기 및 이론적 배경
기존 시각 자기지도학습(SSL) 및 비디오 파운데이션 모델은 주로 **생성 패러다임(Generative Paradigm)**—원시 픽셀을 복원하는 MAE 계열이나 확산(Diffusion)/자기회귀(Autoregressive) 모델—에 의존해왔다. 그러나 Yann LeCun이 제안한 World Model 이론에 따르면, 물리 세계의 현상은 정보 엔트로피와 인과적 예측 가능성 관점에서 세 가지 수준으로 분할된다:
1. **예측 불가능한 고주파 미세 디테일 (Irreducible Noise)**: 잔물결, 나뭇잎의 개별 떨림, 센서의 양자 노이즈 등 물리적 미시 상태는 결정론적으로 예측이 불가능하다. 픽셀 공간 생성 모델(MAE 등)은 이 무의미한 세부사항을 손실함수($\mathcal{L}_{\text{MSE}}$)에 강제로 반영해야 하므로 모델의 표현 용량(Capacity) 상당수를 고주파 노이즈 피팅에 낭비한다.
2. **단기적 가예측성과 장기적 확률적 불확실성 (Short-term Determinism vs Long-term Multimodality)**: 시간 흐름에 따라 미래 궤적은 다수의 그럴듯한 모드(Multimodal distribution)로 분기한다. 생성 모델은 이를 표현하기 위해 $O(N)$ 스텝의 반복적 디노이징(Diffusion)이나 토큰별 순차 샘플링(Autoregressive)을 수행해야 하며, 이는 막대한 계산 복잡도를 유발한다.
3. **추상적 거시 수준의 예측 가능성 (Predictable Abstract Dynamics)**: 물리 법칙, 물체의 영속성, 객체 간 인과적 상호작용 등 고차원적 시맨틱은 고도로 예측 가능하다.

#### 2. JEPA의 해법: 잠재 표현 공간에서의 예측
JEPA(Joint Embedding Predictive Architecture)는 원시 관측치 $y$를 픽셀 단위로 렌더링하지 않고, 인코더가 추상화한 잠재 공간(Latent Space) $s_y = \text{Enc}_{\bar{\theta}}(y)$ 상에서 예측 오차를 최소화한다:
$$\mathcal{L} = D\big(\text{Pred}_\phi(s_x, z), s_y\big)$$
표현 공간의 불변성(Invariance) 덕분에 불필요한 고주파 노이즈는 인코더 단계에서 자연스럽게 필터링되며, 예측 불확실성은 잠재 변수 공간의 분산(Variance/Manifold)으로 효율적으로 흡수된다.

#### 3. 핵심 반론 격파: "LLM의 토큰 생성은 성공했는데, 왜 비디오 픽셀 생성은 부적절한가?"
- **언어(Text)의 엔트로피 특성**: 자연어는 인간의 인지 필터를 통과하여 상징화(Symbolic abstraction)된 **극도로 낮은 엔트로피(Low-entropy)**의 이산적(Discrete) 신호이다. 모든 토큰 하나하나가 시맨틱을 가지므로 차례대로 생성하는 것이 최적의 정보 처리 경로이다.
- **비디오(Video)의 엔트로피 특성**: 비디오는 물리적 연속 공간을 무차별적으로 샘플링한 **극도로 높은 엔트로피(High-entropy)**의 연속형 신호이다. 인접 픽셀 및 프레임 간의 정보 잉여(Redundancy)가 극심하며, 픽셀 값의 90% 이상은 인과적 의미가 없는 배경 잡음에 불과하다. 따라서 비디오를 토큰화하여 픽셀/패치 단위로 생성하려는 시도는 저엔트로피 상징 체계의 성공 방식을 고엔트로피 물리 신호에 잘못 투사한 범주 오류(Category Error)이다.

---

### Q2 — 텍스트와 비디오는 어떻게 정렬되는가 (생성 없이)?

#### 1. 비생성적 정렬(Non-generative Alignment)의 본질
전통적인 멀티모달 모델은 두 가지 방식으로 정렬을 달성한다:
1. **판별적 대조 정렬 (CLIP)**: 이미지와 텍스트를 단일 공유 벡터 공간에 매핑하여 코사인 유사도를 최대화한다($\cos(u_I, v_T)$). 이는 정적 객체 인식에는 유효하나, 시간에 따른 물리적 동역학이나 행동 변화를 예측하는 능력이 결여된다.
2. **생성적 디코딩 (LLaVA/Video-LLM)**: 비전 토큰을 LLM에 주입하여 텍스트 토큰을 자기회귀적으로 생성하도록 크로스엔트로피를 최적화한다.

반면 VL-JEPA에서의 비전-언어 정렬은 **"동역학 예측을 위한 조건부 정보성(Conditional Informativeness)"**으로 정의된다. 텍스트 $z$는 단순한 매칭 대상이 아니라, 월드 모델의 전이 함수(Transition Function)를 조향하는 **제어 입력(Control Input / Action Parameter)** 역할을 수행한다.

#### 2. 정보 흐름과 수식 메커니즘
현재 시각 컨텍스트 표현 $s_x = \text{Enc}_\theta(x)$와 언어 지시문 $z = \text{TextEnc}(w)$가 주어졌을 때, 정렬은 Predictor $\text{Pred}_\phi$가 미래 표현 $s_y$를 정확히 복원하도록 유도하는 최적화 과정에서 암묵적으로 형성된다:
$$\min_{\theta, \phi} \mathbb{E}_{(x, w, y)} \left[ D\Big(\text{Pred}_\phi\big(\text{Enc}_\theta(x), \text{TextEnc}(w)\big), \text{Enc}_{\bar{\theta}}(y)\Big) \right]$$
텍스트가 미래 시각 상태의 불확실성을 유의미하게 해소하지 못하면 Predictor는 $\hat{s}_y$를 타깃 $s_y$에 근접시킬 수 없다. 즉, **"텍스트가 비전의 물리적 미래 변화를 올바르게 규명할 때만 손실이 최소화"**되므로, 픽셀 생성이나 인위적 contrastive negative 없이도 자연스러운 심층 시맨틱 정렬이 강제된다.

#### 3. 세션 팩트시트 확인 과제: 텍스트 주입 지점
- **가설 1 (Predictor 입력단 결합)**: Context visual embedding $s_x$와 text embedding $z$를 Concat 또는 Cross-Attention으로 Predictor에만 공급. (인코더는 순수 비각 표현의 일반성을 보존하므로 이론적으로 가장 순수한 JEPA 월드모델 구조)
- **가설 2 (인코더 내부 조기 융합)**: Vision Encoder 중간 레이어에서 Text-Vision Cross Attention 수행. (정렬 강도는 높아지나 시각 인코더의 독립적 표현 전이성은 저하)
> *세션 당일 논문 원문의 Architecture Diagram 및 Implementation Detail을 확인하여 주입 위치를 확정할 것.*

---

### Q3 — Predictor의 용량과 역할 분담 (계산 비대칭성)

#### 1. 계산 비대칭성(Computational Asymmetry)의 구조
I-JEPA 및 VL-JEPA 계열의 지배적 아키텍처 규칙은 **"Heavy Encoder, Lightweight Predictor"**이다. (예: Vision Encoder는 ViT-Huge/Large, Predictor는 6~12 레이어의 얕은 Transformer 블록). 이러한 비대칭성은 다음 두 가지 핵심 공학적/이론적 필연성에 기인한다.

#### 2. 비대칭성이 제공하는 이중 이점
1. **표현의 범용성 및 특이성 분리 (Separation of Generality and Specificity)**:
   - **Context Encoder**는 임의의 다운스트림 태스크에 전이 가능한, 시각 세계의 보편적이고 정보량이 풍부한 불변 표현(Invariant semantic manifold)을 학습해야 한다.
   - **Predictor**는 특정 마스킹 패턴, 특정 시간 간격($\Delta t$), 특정 텍스트 프롬프트에 종속된 조건부 변환(Task-specific mapping)을 처리하는 가변적 모듈이다.
   - 만약 Predictor의 용량이 너무 크면, 인코더가 거시적 의미를 구조화하지 않고 원시 정보를 방치하더라도 Predictor 혼자서 복잡한 비선형 매핑을 통해 타깃을 맞춰버리는 병목 우회(Shortcut learning)가 발생한다. 반대로 Predictor를 극도로 제한하면, 인코더는 Predictor가 선형/준선형 결합만으로도 미래를 유추할 수 있도록 **가장 정제되고 선형 분리 가능한(Linearly separable) 고품질 표현**을 산출하도록 강제된다.
2. **표현 붕괴(Collapse) 방지를 위한 유도 바이어스**:
   - 예측기의 용량 제한은 온라인 네트워크와 타깃 네트워크 간의 정보 대칭성을 깨뜨리는 물리적 지연 장치로 작용하여 상호 동기화에 의한 자명해 수렴을 지연시킨다.

#### 3. 절제 실험 대비: "Predictor를 완전히 제거하면($\text{Identity}$) 어떻게 되는가?"
Predictor가 제거되면 $\text{Enc}_\theta(x)$ 자체가 $\text{Enc}_{\bar{\theta}}(y)$와 직접 일치해야 한다. 이는 공간적/시간적으로 다른 두 패치가 동일한 벡터로 축약되도록 강제하므로, 인코더는 위치 정보와 시간적 변화량을 소거하고 모든 입력을 평균 벡터로 매핑하는 심각한 표현 퇴화(Semantic dilution / Position collapse)를 겪게 된다.

---

### Q4 — Negative Pair 없이 Collapse를 어떻게 막는가 (수학적/동역학적 근거)

#### 1. Information Collapse의 수학적 정의
대조 학습(Contrastive Learning)과 달리 JEPA는 음성 쌍(Negative pairs)을 분모에 두는 InfoNCE 류의 반발력(Repulsive force)을 사용하지 않는다. 따라서 목적함수 $\mathcal{L} = \|\hat{s}_y - s_y\|_2^2$는 모든 입력 $x, y$에 대해 $\text{Enc}(x) = \text{Enc}(y) = \mathbf{c}$ (상수 벡터)로 붕괴할 때 손실이 완전한 $0$이 되는 자명해(Trivial Solution)의 위험을 본질적으로 안고 있다.

#### 2. 다층 방어선 (Three Defensive Pillars)
VL-JEPA는 다음 세 가지 장치의 상호작용을 통해 동역학적으로 Collapse를 억제한다:
1. **타깃 인코더의 역전파 차단 ($\text{stop\_gradient}$)**:
   $$\theta_{\text{target}} \not\leftarrow \nabla_{\theta_{\text{target}}} \mathcal{L}$$
   온라인 인코더와 타깃 인코더가 동일 목적함수를 공유하여 동시에 상수로 붕괴하는 양방향 경사 하강 피드백 루프를 차단한다.
2. **지수 이동 평균(EMA)을 통한 완만한 가중치 갱신**:
   $$\bar{\theta}_t = \tau \bar{\theta}_{t-1} + (1 - \tau) \theta_t \quad (\tau \to 1)$$
   EMA 타깃 인코더는 온라인 인코더의 파라미터 궤적에 대한 저주파 통과 필터(Low-pass filter)로 작동한다. 이는 타깃 표현 공간의 급격한 변동을 억제하며, 표현 공간이 한 점으로 축약되는 동역학적 속도보다 훨씬 느리게 기준 좌표계를 갱신함으로써 붕괴 궤적으로의 수렴을 저지한다.
3. **구조적 비대칭성 ($\text{Predictor}$의 존재)**:
   Predictor $\text{Pred}_\phi$의 개입으로 인해 온라인 경로의 출력과 타깃 경로의 출력 간에 수학적 비대칭성이 보장된다.

#### 3. 엄밀한 학술적 평가 (Feynman Perspective)
- "이 3가지 장치가 정보 붕괴가 불가능함을 수학적으로 엄밀히 증명(Rigorous Proof)하는가?"에 대한 답은 **"아니다(No)"**이다. BYOL, SimSiam, I-JEPA 계열의 비붕괴 현상은 주로 실측된 최적화 동역학(Empirical Optimization Dynamics)에 기초한다.
- 그러나 정보 이론적으로 완전한 붕괴 방지를 보장하기 위해, 공변량 행렬의 대각 성분을 1로 유지하고 비대각 성분을 0으로 강제하는 **VICReg(Variance-Invariance-Covariance Regularization)** 손실을 보조 정규화 항으로 결합할 수 있으며, 실제 대규모 학습 안정성은 EMA 스케줄링($\tau$ annealing)의 정밀한 튜닝에 크게 의존한다.

---

### Q5 — 로보틱스 및 액션 플래닝(World Model)으로의 확장성

#### 1. 잠재 공간 플래닝(Latent Space MPC)의 계산적 우위
로봇 공학 및 엠바디드 AI에서 월드 모델의 핵심 임무는 미래 상태 시뮬레이션을 통한 최적의 행동 궤적 탐색(Model Predictive Control, MPC)이다.
- **기존 비디오 생성 월드 모델 (Sora, Diffusion World Models)**:
  미래 궤적을 평가하기 위해 매 스텝마다 수백만 픽셀을 렌더링해야 하므로, 수백~수천 개의 잠재 행동 후보를 실시간으로 롤아웃(Rollout)하는 것이 계산적으로 불가능하다.
- **VL-JEPA 월드 모델**:
  언어 조건 $z$를 로봇의 제어 액션 $a_t \in \mathcal{A}$로 치환하면, Predictor는 완벽한 **잠재 동역학 전이 모델(Latent Dynamics Simulator)**이 된다:
  $$\hat{s}_{t+1} = \text{Pred}_\phi(\hat{s}_t, a_t)$$
  픽셀 렌더링 없이 경량 Predictor 포워드 패스만으로 수천 번의 롤아웃을 수 밀리초(ms) 내에 수행할 수 있어 실시간 고차원 궤적 최적화(CEM, MCTS)가 가능해진다.

#### 2. 정직한 불확실성(Honest Uncertainty)의 표현
물리적 상호작용 시 객체의 가려짐(Occlusion)이나 미끄러짐과 같은 확률적 결과에 대해, 픽셀 모델은 흐릿한 픽셀(Blur)을 생성하거나 비현실적인 환각(Hallucination)을 발생시킨다. 반면 JEPA는 잠재 표현 공간 내의 에너지 표면(Energy Landscape)이나 분산으로 불확실성을 표현하므로, 플래닝 알고리즘이 위험 영역을 신뢰성 있게 회피할 수 있다.

#### 3. 결정적 한계와 비판적 검토 (The Achilles' Heel)
- **개입 가시화성 및 검증 가능성(Verifyability/Interpretable Grounding)의 부재**:
  JEPA는 픽셀을 렌더링하지 않으므로, 에이전트가 상상한 미래 $\hat{s}_t$가 물리적으로 타당한지 인간 조작자가 직관적으로 검증하거나 디버깅하기 극히 어렵다.
- **표현 드리프트(Latent Compounding Error)**:
  장기 롤아웃(Long-horizon planning) 시 디코더를 통한 물리적 앵커링이 없으므로, 잠재 공간에서의 미세한 예측 편향이 누적되어 실제 물리 역학에서 완전히 벗어난 매니폴드로 발산할 위험이 상존한다.

---

## 🎯 세션 당일 즉각 채워 넣을 검증 체크리스트 (Factsheet Checklist)

| 검증 항목 | 후보 값 / 가설 | 세션 실측 확인값 (당일 기입) | 논증 영향도 |
| :--- | :--- | :--- | :--- |
| **EMA 모멘텀 $\tau$** | Cosine schedule $0.996 \to 1.0$ vs 고정 $0.999$ | `[ ]` | Q4 Collapse 방지 동역학의 안정성 입증 |
| **비디오 마스킹 전략** | Space-time Tube Masking vs 3D Block Masking | `[ ]` | Q1 고주파 시간 노이즈 제거 효과 분석 |
| **텍스트 주입 지점** | Predictor 전용 Cross-Attention vs ViT Encoder 내부 | `[ ]` | Q2 비전 인코더의 독립적 표현 순도 보존 여부 |
| **다운스트림 벤치마크** | VideoQA (ActivityNet), Action Anticipation (Epic-Kitchens) | `[ ]` | Q5 월드모델 전이 성능 실측 지지 |

---

## ⚔️ 티쳐/리뷰어 관점 심층 꼬리 질문 세트 (12 Killer Questions)

### Part 1. 아키텍처 및 손실 함수 메커니즘 (Q1, Q2 연계)
1. **[Q1-1]** "픽셀을 버리면 고주파 노이즈를 피한다고 했는데, 미세한 조작(Fine-grained manipulation)이 필요한 로봇 태스크(예: 바늘에 실 꿰기)에서 그 '고주파'가 핵심 시그널이면 JEPA는 어떻게 대응하는가?"
2. **[Q1-2]** "MAE는 디코더를 떼어내면 인코더 표현이 다운스트림에서 훌륭히 동작한다. MAE가 고주파 노이즈에 용량을 낭비한다는 주장이 단순한 직관을 넘어 실제 Linear Probing 성능 격차로 증명되는가?"
3. **[Q2-1]** "텍스트 $z$가 Predictor에만 들어간다면, Context Encoder는 언어의 존재를 전혀 모른 채 학습된다. 그렇다면 이 비전 인코더가 진정한 '비전-언어 정렬 표현'을 가졌다고 볼 수 있는가?"
4. **[Q2-2]** "단일 비디오 클립에 대해 여러 가지 언어적 해석(다양한 프롬프트)이 가능할 때, Predictor의 출력이 특정 텍스트 조건으로 인해 붕괴되거나 모드 평균(Mode averaging)으로 수렴하는 것을 어떻게 방지하는가?"

### Part 2. 비대칭성 및 최적화 동역학 (Q3, Q4 연계)
5. **[Q3-1]** "Predictor를 깊게(Deep) 만들고 Encoder를 얕게 만들면 정확히 어떤 역효과가 수치적으로 나타나는가? 저자들이 관찰한 실패 모드는 무엇인가?"
6. **[Q3-2]** "I-JEPA 논문에서는 Target block과 Context block 간의 의미적 중복을 피하기 위해 마스킹 크기를 극단적으로 키운다. VL-JEPA에서 시간축(Time-axis) 마스킹 비율은 몇 %이며, 지나치게 큰 마스킹이 비디오 연속성을 해치지 않는가?"
7. **[Q4-1]** "EMA 모멘텀 $\tau=1$로 고정하면 타깃 인코더가 초기화 가중치로 동결된다. 이 경우에도 비붕괴 표현 학습이 가능한가? 가능하다면 랜덤 네트워크 타깃(Random projection)과의 차이는 무엇인가?"
8. **[Q4-2]** "DINOv2는 Sinkhorn-Knopp 중심화 및 샤프닝을 명시적으로 사용한다. VL-JEPA는 이러한 명시적 정규화 없이 순수 $L_1/L_2$ 거리만으로 고차원 공간 상의 차원 붕괴(Dimensional collapse)를 완전히 배제할 수 있는가?"

### Part 3. 월드 모델 및 플래닝 한계 (Q5 연계)
9. **[Q5-1]** "잠재 공간 롤아웃에서 오차가 누적되는 컴파운딩 에러(Compounding error)를 완화하기 위해 VL-JEPA가 채택한 메커니즘(예: 다단계 예측, 정규화)은 무엇인가?"
10. **[Q5-2]** "비디오-텍스트 정렬에서 텍스트가 행동(Action)을 지시할 때, 연속적인 물리 제어값(예: 로봇 토크, 6-DoF 포즈)과 이산적 자연어 명령 간의 표현 간극을 어떻게 좁힐 것인가?"
11. **[Q5-3]** "픽셀 렌더링이 불가능하다면 플래닝된 잠재 경로 $\hat{s}_1, \dots, \hat{s}_T$에 대한 보상 함수(Reward function)는 어디서 가져오는가? 보상 모델 역시 잠재 공간 상에 사전 정의되어야 하는 순환 논리에 빠지지 않는가?"
12. **[Q5-4]** "Sora나 Runway 같은 거대 생성 모델 진영이 막대한 데이터와 연산량으로 픽셀 예측의 일관성을 극복해 낸다면, 표현 예측(JEPA) 패러다임이 여전히 압도적인 이론적 비교우위를 유지할 수 있는가?"
