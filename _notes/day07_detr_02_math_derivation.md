---
layout: default
title: "Day 07 DETR — 02 · 수식유도 — DETR 매칭·손실부터 RT-DETR 불확실성까지"
paper_id: detr
date: 2026-10-20
---

# 02 · 수식유도 — DETR 매칭·손실부터 RT-DETR 불확실성까지

> 원문 텍스트: `day07_detr/src/paper.txt` (DETR) · `day07_detr/src/rtdetr.txt` (RT-DETR) — 모든 인용은 이 두 파일에서 확인
> 표기: $\mathcal{L}$ = 손실, $\hat y$ = 예측, $y$ = 정답(GT), $\sigma$ = 순열, $\varnothing$ = no-object

---

## 1. 식 (1) — 양자택일 매칭을 "순열 탐색"으로 쓰는 이유

논문은 다음을 최적화한다.

$$\hat\sigma=\arg\min_{\sigma\in\mathfrak S_N}\sum_{i=1}^{N}\mathcal L_{\rm match}\big(y_i,\hat y_{\sigma(i)}\big)\tag{1}$$

### 1.1 왜 순열로 써야 하는가

- $\hat y$는 **집합(set)** 이다. 집합에는 원소의 순서가 없으므로, GT $y_i$와 "어떤 $\hat y$가 짝인지"를 **결정해야** 손실을 정의할 수 있다. 순열 $\sigma$가 그 결정이다.
- 반대로, 순열에 집합을 대응시키는 게 아니라 **집합에 순열을 대응**시키면 "각 GT가 정확히 하나의 예측과 짝"이라는 조건이 자동으로 성립한다. 이것이 논문 §2.1의 요지다: *"The usual solution is to design a loss based on the Hungarian algorithm, to find a bipartite matching between ground-truth and prediction. This enforces permutation-invariance, and guarantees that each target element has a unique match."*
- NMS를 쓰는 탐지기는 다대일(many-to-one) 할당을 허용한다. 그래서 중복이 생기고, NMS로 사후 제거해야 한다. 식 (1)은 **one-to-one**이므로 중복 자체가 구조적으로 불가능하다.

### 1.2 탐색 비용: $N!$ vs $O(N^3)$

- 순열 개수는 $|\mathfrak S_N| = N!$. DETR의 $N=100$이면 $100!\approx 9.3\times10^{157}$ — 완전 탐색은 불가능하다.
- 그런데 이 문제는 **할당 문제(assignment problem)** 이다. 비용 행렬 $C\in\mathbb R^{N\times N}$ 에 대해 최소 비용 perfect matching을 찾는 헝가리아리안 알고리즘은 $O(N^3)$ 이다.
- 논문 §3.1: *"This optimal assignment is computed efficiently with the Hungarian algorithm, following prior work."* 코드도 정확히 이 알고리즘을 쓴다(`models/matcher.py:81` — `scipy.optimize.linear_sum_assignment`).
- $\hat y$가 $N$개, GT가 $M(\le N)$개일 때: GT가 없는 슬롯은 $\varnothing$로 채워 $N$개로 맞춘다(논문 §3.1: *"we consider y also as a set of size N padded with ∅"*). **즉 $\varnothing$는 "예측"이 아니라 "매칭될 대상이 없음"을 나타내는 패딩**이다.

### 1.3 매칭 비용의 구조

$$\mathcal L_{\rm match}\big(y_i,\hat y_{\sigma(i)}\big)=-\mathbb 1_{\{c_i\neq\varnothing\}}\,\hat p_{\sigma(i)}(c_i)+\mathbb 1_{\{c_i\neq\varnothing\}}\,\mathcal L_{\rm box}(b_i,\hat b_{\sigma(i)})$$

- $\mathbb 1_{\{c_i\neq\varnothing\}}$: GT가 실재할 때만 비용이 생긴다. **$\varnothing$ 매칭의 비용은 예측과 무관한 상수**다. 논문 §3.1이 명시한다: *"the matching cost between an object and ∅ does not depend on the prediction, which means that in that case the cost is a constant."*
- 왜 상수가 되는가: $\varnothing$는 "이 슬롯은 아무것도 아니다"는 라벨이므로, 어떤 예측이 이 슬롯에 붙어도 예측 측의 정보(확률·박스)는 소모되지 않는다. 비용 행렬의 해당 열은 **모든 행이 같은 값**을 갖는다.
- 부수 효과: $\varnothing$ 열이 상수이므로 그 열들의 배정 순서는 총비용을 바꾸지 않는다. 즉 **"어떤 슬롯이 ∅가 될지"는 학습되지 않는다**(수량만 학습된다). 이는 §3.2의 aux loss가 "객체 수"를 맞추는 것과 정합한다.

---

## 2. 식 (2) — 매칭은 확률, 손실은 로그 우도

$$\mathcal L_{\rm Hungarian}(y,\hat y)=\sum_{i=1}^{N}\Big[-\log \hat p_{\hat\sigma(i)}(c_i)+\mathbb 1_{\{c_i\neq\varnothing\}}\,\mathcal L_{\rm box}(b_i,\hat b_{\hat\sigma(i)})\Big]\tag{2}$$

### 2.1 왜 둘이 다른 함수인가

- **매칭 단계**는 "어떤 GT를 어떤 예측에 붙일지"를 정하는 **비교 연산**이다. 비교 대상인 박스 항 $\mathcal L_{\rm box}$는 **유계**다(정규화된 좌표에서 L1 $\in[0,4]$, GIoU 손실 $\in[0,2]$).
- 클래스 항을 $-\log \hat p$로 쓰면 $\hat p\to0$일 때 $\to\infty$ 로 발산해, 유계인 박스 항을 압도한다. 즉 매칭 단계에서 클래스 항이 **무한히 큰 스케일**을 갖게 되어, 어떤 GT가 "아주 확신 없는 예측"과 짝이 되도록 억지로 당겨진다. 이건 의도된 "확실한 짝"과 반대 방향이다.
- **확률** $\hat p\in[0,1]$을 쓰면 클래스 항은 $[-1,0]$으로 **박스 항과 같은 스케일**이 된다. 논문 §3.1: *"In the matching cost we use probabilities $\hat p(c_i)$ instead of log-probabilities. This makes the class prediction term commensurable to $\mathcal L_{\rm box}(\cdot,\cdot)$ (described below), and we observed better empirical performances."*
- **손실 단계**는 매칭이 정해진 뒤의 **최대 우도 추정**이다. 분류 손실은 NLL $=-\log \hat p$ 가 표준이고(식 2의 첫 항), 이 단계에서는 발산을 허용해야 확신이 낮은 예측이 강하게 교정된다. 그래서 **매칭과 손실이 다른 확률 함수를 쓰는 것은 모순이 아니라 역할 분담**이다.

### 2.2 순열 불변성

식 (2)는 $\hat\sigma$가 최적이므로, 예측 슬롯의 **순서를 임의로 섞어도** 총 손실 값은 같다(매칭이 다시 최적 순열을 찾기 때문). 실습 [A]의 순열 불변성 측정($|L-L'|=0.00e{+}00$)이 이 성질의 직접 검증이다. 이것이 §2.2의 "we can emit them in parallel"의 근거다 — 순서를 정하는 장치가 손실 안에 이미 들어 있다.

---

## 3. $\varnothing$ 하향 10의 산술

논문 §3.1: *"In practice, we down-weight the log-probability term when $c_i=\varnothing$ by a factor 10 to account for class imbalance."*

- 코드: `models/detr.py:104-106` — `empty_weight[-1] = self.eos_coef`(기본 0.1). 즉 $\varnothing$ 클래스 항에 **0.1배**가 곱해진다.
- **왜 필요한가**: $N=100$ 슬롯, COCO 평균 7객체(§4: *"There are 7 instances per image on average"*) → 한 장의 GT는 7개, 나머지 **93개 슬롯은 $\varnothing$**.
- 매칭된 7쌍의 분류 손실을 $c_{\rm pos}=-\log \hat p_{\rm cat}$, $\varnothing$ 슬롯의 분류 손실을 $c_{\rm bg}=-\log \hat p_{\varnothing}$라 두면, 한 장의 분류 손실 합은

$$L_{\rm cls}\approx 7\,c_{\rm pos}+93\cdot 0.1\,c_{\rm bg}$$

  (하향 전: $7c_{\rm pos}+93c_{\rm bg}$). 실습 [C]의 예시값 $\hat p_{\varnothing}=0.015$, $\hat p_{\rm cat}=0.90$이면 $c_{\rm bg}=4.20$, $c_{\rm pos}=0.105$ → **하향 전후 39.9x → 4.0x**.
- **해석(경계 포함):** 하향은 *균형*을 잡을 뿐 *해소*하지 않는다. 93개 슬롯이 여전히 손실 합의 98.1%를 차지한다(실습 [C]). 그래서 DETR는 $\varnothing$ 예측을 **"지금 없다"가 아니라 "지금 없다"는 확신**으로 학습한다. 이 발상 자체가 세트 예측의 핵심이다: 빈 슬롯은 에러가 아니라 정상 출력이다.
- 대응되는 기존 관례: 논문 §3.1 *"This is analogous to how Faster R-CNN training procedure balances positive/negative proposals by subsampling."* — Faster R-CNN은 양성 proposal을 적게 **뽑고**(subsampling) 비율로 균형을 맞췄고, DETR는 $\varnothing$ **손실 가중치**로 같은 일을 한다.

---

## 4. 식 (9)–(10) — 박스 손실: L1과 GIoU

$$\mathcal L_{\rm box}\big(b_{\sigma(i)},\hat b_i\big)=\lambda_{\rm iou}\,\mathcal L_{\rm iou}(b_{\sigma(i)},\hat b_i)+\lambda_{\rm L1}\,\big\|b_{\sigma(i)}-\hat b_i\big\|_1 \tag{9}$$

- $\lambda_{\rm L1}=5$, $\lambda_{\rm iou}=2$ (부록 0.A.4: *"$\lambda_{L1}=5$ and $\lambda_{iou}=2$"*).
- 정규화: *"All losses are normalized by the number of objects inside the batch."* (부록 0.A.2) — 코드 `models/detr.py:228-232`는 `all_reduce`로 **전체 GPU**의 객체 수로 정규화한다.

### 4.1 GIoU 정의 (식 10)

$$\mathcal L_{\rm iou}(b_{\sigma(i)},\hat b_i)=1-\left(\frac{|b_{\sigma(i)}\cap\hat b_i|}{|b_{\sigma(i)}\cup\hat b_i|}-\frac{|B(b_{\sigma(i)},\hat b_i)\setminus b_{\sigma(i)}\cup\hat b_i|}{|B(b_{\sigma(i)},\hat b_i)|}\right)\tag{10}$$

기호를 정리하면: $A=b_{\sigma(i)}$, $B'=\hat b_i$, $C=B(A,B')$(둘을 포함하는 최소 박스).

$$\text{IoU}(A,B')=\frac{|A\cap B'|}{|A\cup B'|},\qquad \text{GIoU}=\text{IoU}-\frac{|C\setminus(A\cup B')|}{|C|}$$

- $C\setminus(A\cup B')$는 **"포함은 되지만 어느 한쪽에도 속하지 않는" 빈 영역**이다. 즉 GIoU는 IoU에 **"둘 다 덮지 못한 여백"**을 감점한다.
- 두 박스가 겹치지 않으면 IoU $=0$이지만, 여백 비율이 남아 GIoU $<0$이 될 수 있다 → 그래디언트가 계속 존재한다($\text{IoU}$는 겹치지 않는 경우 0에서 미분 불가).
- 논문 부록 0.A.2: *"The areas of unions or intersections are computed by min/max of the linear functions … which makes the loss sufficiently well-behaved for stochastic gradients."*

### 4.2 스케일 불변성 증명

**주장.** 임의의 $\alpha>0$, 평행이동 $t$에 대해 $\text{GIoU}(\alpha A+t,\alpha B'+t)=\text{GIoU}(A,B')$.

**증명.** $\alpha A+t=\{x:\ x=\alpha a+t,\ a\in A\}$로 두면 측도(넓이)가 $\alpha^2$배 되고, 교집합·합집합도 같은 방식으로 변환된다.

- 교집합 넓이: $|\alpha A+t\cap(\alpha B'+t)|=\alpha^2|A\cap B'|$
- 합집합 넓이: $|\alpha A+t\cup(\alpha B'+t)|=\alpha^2|A\cup B'|$

따라서

$$\text{IoU}'=\frac{\alpha^2|A\cap B'|}{\alpha^2|A\cup B'|}=\text{IoU}$$

포함 박스 $C=B(A,B')$도 $C'=\alpha C+t$가 되고($\min/\max$가 아핀변환과 교환됨),

$$|C'|=\alpha^2|C|,\qquad |C'\setminus(\alpha A+t\cup(\alpha B'+t))|=\alpha^2|C\setminus(A\cup B')|$$

이므로

$$\text{GIoU}'=\text{IoU}'-\frac{\alpha^2|C\setminus(A\cup B')|}{\alpha^2|C|}=\text{IoU}-\frac{|C\setminus(A\cup B')|}{|C|}=\text{GIoU}\qquad\blacksquare$$

**반면 L1은 불변이다:**

$$\|\alpha(A-B')\|_1=\sum_{k=1}^{4}|\alpha\,\Delta b_k|=\alpha\sum_{k=1}^{4}|\Delta b_k|=\alpha\,\|\Delta b\|_1$$

즉 $\alpha\neq1$이면 값이 $\alpha$배로 변한다. **같은 상대 오차여도 L1 손실은 상자 크기에 비례해 커진다**(실습 [B]: 15% 상대 오차에서 L1 비율 9.44x, GIoU 비율 1.000x).

**왜 둘이 필요한가 (Table 4, 300ep).**

| 구성 | AP | Δ | APS | APM | APL |
|---|---|---|---|---|---|
| class + L1 | 35.8 | −4.8 | 13.7 | 39.8 | 57.9 |
| class + GIoU | 39.9 | −0.7 | 19.9 | 43.2 | 57.9 |
| class + L1 + GIoU | 40.6 | — | 19.9 | 44.3 | 60.2 |

- **GIoU만**으로 거의 다 온다(−0.7). 스케일 불변이라 **크기 스펙트럼 전체를 균등하게** 다룬다.
- **L1만**이면 스케일 잭 때문에APS가 13.7로 처참하고 APL은 57.9로 GIoU와 동일하다 — 즉 L1은 **큰 상자만** 잘 맞추는 항(정확도 항)이었다.
- **둘 다**면 APL이 60.2로 오른다(분류에 미처 분류가 아닌 **정밀한 정렬** 기여). 두 손실은 "크기 균형(GIoU)"과 "정밀 좌표(L1)"을 나누어 담당한다.
- 경계: 논문도 스스로 짚는다 — *"We only studied simple ablations of different losses (using the same weighting every time), but other means of combining them may achieve different results."* 가중치 탐색이 아니라 **구성** ablation이다.

---

## 5. RT-DETR — 실시간을 위한 두 개의 수식

### 5.1 식 (1) 하이브리드 인코더

$$\mathcal Q=\mathcal K=\mathcal V=\texttt{Flatten}(\mathcal S_5),\quad \mathcal F_5=\texttt{Reshape}(\texttt{AIFI}(\mathcal Q,\mathcal K,\mathcal V)),\quad \mathcal O=\texttt{CCFF}(\{\mathcal S_3,\mathcal S_4,\mathcal F_5\})\tag{R1-1}$$

- $\mathcal S_5$는 백본 마지막 stage(해상도가 가장 낮고 의미가 가장 풍부한 레벨). $\mathcal S_3,\mathcal S_4$는 더 높은 공간 해상도·얕은 의미.
- **AIFI는 $\mathcal S_5$에만** self-attention을 건다. 논문 §4.2 근거: *"applying the self-attention operation to high-level features with richer semantic concepts captures the connection between conceptual entities … the intra-scale interactions of lower-level features are unnecessary due to the lack of semantic concepts and the risk of duplication and confusion."*
- **비용 계산(입력 640×640 기준)**: $\mathcal S_5$는 $20\times20=400$ 토큰, 세 스케일 전체는 $80^2+40^2+20^2=8400$ 토큰. self-attention의 attention 행렬 비용은 토큰 수의 **제곱**에 비례하므로, 8400 토큰 전체에 self-attention을 걸면 $400$ 토큰의

$$\frac{8400^2}{400^2}=441$$

  배다. 하위 레벨의 self-attention을 "필요 없다"가 아니라 **441배 비싸다**가 논문의 논리다. ablation D → D_S5: **지연 −35%, AP +0.4**(46.4 → 46.8).
- **CCFF**는 conv 기반이라 위치와 채널을 밀도 있게 처리한다. $\mathcal O$는 세 스케일 모두 256ch로 정렬된 채 디코더로 간다.

### 5.2 식 (2)–(3) 불확실성 최소화 쿼리 선택

$$\mathcal U(\hat{\mathcal X})=\big\|\mathcal P(\hat{\mathcal X})-\mathcal C(\hat{\mathcal X})\big\|,\qquad \hat{\mathcal X}\in\mathbb R^{D}\tag{R1-2}$$

$$\mathcal L(\hat{\mathcal X},\hat{\mathcal Y},\mathcal Y)=\mathcal L_{\rm box}(\hat b,b)+\mathcal L_{\rm cls}\big(\mathcal U(\hat{\mathcal X}),\hat c,c\big)\tag{R1-3}$$

- $\mathcal P$ = **위치 예측의 분포**, $\mathcal C$ = **분류 예측의 분포**. 둘의 **차이(불일치)**가 "이 피처가 무엇인지에 대한 확신이 얼마인지"의 대리표(proxy)다. 즉 하나의 잠재 변수(피처의 질)를 두 관측(카테고리·위치)으로 나눠 본 뒤, 둘이 어긋나는 정도를 **epistemic uncertainty**로 정의한다.
- 기존 쿼리 선택은 **분류 점수만**으로 top-K를 뽑았다. 그런데 탐지기는 카테고리와 위치를 **동시에** 모델링해야 하므로, "분류는 자신 있지만 위치는 틀린" 피처가 선택되면 그 피처는 좋은 초기 쿼리가 아니다.
- 선택 기준이 바뀌면 **같은 top-K 안에서 내용이 달라진다**: 분류 점수만으로는 못 골랐던 "분류·위치 모두 확실한" 피처를 우선한다. RT Table 4가 이걸 정량화한다: high-cls 비율 0.35% → **0.82%**, "둘 다 > 0.5" 비율 0.30% → **0.67%**, AP 47.9 → **48.7**(+0.8).
- **경계:** 이 효과는 RT-DETR-R50의 1× 설정 ablation이며, K=300이라는 특정 쿼리 수와 640×640 입력에 대한 값이다. 또한 공개 v1 PyTorch 코드의 top-K는 분류 점수 단독(`rtdetr_decoder.py:491`)이라, 논문의 불확실성 개념과 구현 선택을 **구분해서** 인용해야 한다(01_코드워크스루 §6.2).

### 5.3 디코더 깊이 = 속도 손잡이 (Table 5)

| 사용 층 | AP | 지연(ms) |
|---|---|---|
| 6층(전부) | 53.1 | 9.3 |
| 5층에서 중단 | 53.0 | 8.8 |
| 4층 | 52.7 | 8.3 |
| 3층 | 52.4 | 7.9 |
| 1층 | 49.1 | 7.0 |

- 6 → 5층은 **−0.1 AP로 지연 0.5ms 절약**이지만, 1층까지 낮추면 −4.0 AP다. 즉 **위쪽 층은 거의 버리면서 비용을 못 잡는** 구조다(상위 층이 정밀도를 만든다).
- **재학습 없이 조절 가능한 이유**: 원문 DETR의 aux decoding loss(§3.2)는 **모든 디코더 층에 예측 FFN과 헝가리아 손실을 붙인다**. 따라서 "5층에서 멈춘 출력"도 **손실을 받아 학습된 예측**이다(코드 `models/detr.py:240-250`의 층별 매칭). RT-DETRv2도 이 구조를 유지한다(보충 자료 Fig 5의 블록별 loss 스택).
- 그러므로 속도 조정은 "학습 재수행"이 아니라 **forward에서 어디서 끊을지**만 바꾸는 일이다.

---

## 6. Attention 복잡도 (부록 0.A.3)와 DC5

| 부분 | 복잡도 | 비고 |
|---|---|---|
| 인코더 self-attention | $\mathcal O(d^2HW+d(HW)^2)$ | attention 행렬 항이 $(HW)^2$ |
| 디코더 self-attention | $\mathcal O(d^2N+dN^2)$ | $N=100$ |
| 디코더 cross-attention | $\mathcal O(d^2(N+HW)+dN\cdot HW)$ | $N\ll HW$ 이므로 인코더보다 작음 |

- 인코더 항을 분해하면: head당 $d'=d/M$, M개 헤드 → attention 가중치 계산은 $M\cdot d'\cdot(HW)^2 = d(HW)^2$, QKV 투영은 $d^2HW$.
- **DC5의 비용**: 마지막 stage에 dilation을 걸어 해상도를 2배로 하면 $HW\to4HW$, 따라서 $(HW)^2$ 항은 **16배**가 된다. 논문 §4: *"at the cost of a 16x higher cost in the self-attentions of the encoder, leading to an overall 2x increase in computational cost."*
  - 왜 전체는 2배뿐인가: attention 항이 전체를 지배하지는 않고(백본 backbone·FFN·cross-attention 등 다른 항이 남아), 16배가 걸린 attention 항도 전체의 일부이기 때문이다. Table 1에서 DETR 86 GFLOPs → DETR-DC5 187 GFLOPs.
- **RT-DETR의 AIFI는 이 항을 구조적으로 줄인다**: 640 입력에서 attention은 400 토큰(20×20)만이 대상. DETR-DC5는 800×1333 입력에서 $\mathcal S_5$가 25×42=1050 토큰, DC5까지면 50×84=4200 토큰 — attention 비용은 $4200^2$ vs $400^2$ 로 **110배** 차이가 나기 때문에, RT-DETR는 "해상도를 올리는 대신 의미가 있는 S5에 attention을 몰아넣고 나머지는 conv로 처리"하는 선택을 한다.

---

## 7. 왜 이 조합이 "세트 예측"이 되는가 — 한 문단 요약

순열 $\sigma$를 최적화하는 손실(식 1)은 **예측의 순서에 무관**한 학습 신호를 만든다(식 2). 그 무관함 덕분에 슬롯 $N=100$개를 **한 번에(병렬)** 예측해도 각 슬롯이 "자기 역할"을 찾을 수 있고, one-to-one 제약이 중복을 학습 단계에서 제거하므로 NMS가 필요 없다. 여기에 **스케일 불변(증명한) GIoU**가 크기 스펙트럼 전체를 균등하게 다루고(식 9·10), **∅ 하향 10**이 93개의 빈 슬롯과 7개의 실객체 사이의 불균형을 조절한다. RT-DETR은 이 구조를 실시간 예산 안에 넣기 위해 (i) self-attention을 의미가 가장 농한 한 스케일에만 적용하고(AIFI), (ii) 쿼리 초기화의 불확실성을 줄이며(식 2·3), (iii) 학습된 모든 디코더 층을 속도 손잡이로 재사용한다(aux loss).
