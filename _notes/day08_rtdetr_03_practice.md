---
layout: default
title: "Day 08 RT-DETR — 03 · 실습 — `rtdetr_demo.py` 4검증 (CPU 실측)"
paper_id: rt-detr
date: 2026-10-21
---

# 03 · 실습 — `rtdetr_demo.py` 4검증 (CPU 실측)

> 스크립트: `day08_rtdetr/rtdetr_demo.py` · 로그: `runs/rtdetr_demo_stdout.txt` · 수치: `runs/rtdetr_demo_metrics.json`
> **환경:** 합성 데이터, CPU, 시드 20261021, **학습 없음** — RT-DETR의 네 가지 논문 주장(§3.1·§4.2·§4.3·§5.4)을 **구조·비용·퇴화 방향**측면에서 재현한다.
> **차트 4종**(`runs/chart_*.png`, `make_charts.py`로 재생성): `chart_nms_cost.png` · `chart_attention_scaling.png` · `chart_encoder_variants.png` · `chart_uncertainty_degeneracy.png`

## 0. 실행 방법

```bash
cd /Users/robotjang/paper-diary
.venv/bin/python day08_rtdetr/rtdetr_demo.py    # 약 1분
.venv/bin/python day08_rtdetr/make_charts.py   # 차트 4종 재생성
```

- **이 데모가 하는 일**: RT-DETR 논문의 *수치*를 재현하는 것이 아니라, **논문이 내린 설계 판단의 구조적 근거**(NMS 비용, attention 토큰 스케일, 변형별 비용 순서, uncertainty 항의 퇴화)를 **자체 실측**으로 확인한다. 따라서 아래 수치는 **논문 수치와 직접 비교 대상이 아니다**(도메인·하드웨어·구현이 모두 다르다). 비교 가능한 것은 **방향과 순서**다.

> **⚠️ 재현성 주의 (2026-10-21 추가)**: 이 데모의 수치에는 **결정론적 값**과 **벽시계(wall-clock) 측정값**이 섞여 있다. 2회 실행 대조 결과:
> - **결정론적(재실행해도 동일)**: NMS 전후 박스 수(1897·1750·945 / 1493·1761·1969), **AP proxy**(1.000·0.983·0.933 / 0.967·1.000·1.000), **변형 A~E의 지연 순서**(A < D_S5 < E < B < D < C = 논문과 동일), uncertainty 궤적의 최종값(s 0.690 · q 0.693 / s 0.794 · q 0.793), attention 시간의 **비율**(400→3200 = 42.3배, 재실행 42.1배)
> - **변동(머신 부하에 따라 흔들림)**: **지연의 절대값** — NMS 57.03 → 62.03ms(conf 0.001), 21.29 → 22.18ms(conf 0.25), IoU 0.5 43.05 → **74.06ms**, attention 3200토큰 100.19 → **202.79ms**(약 2배), 변형별 ours ms(C 2.55 → 2.83, D 1.03 → 1.95)
> - **따라서**: 아래 본문의 ms 값은 **커밋된 `runs/rtdetr_demo_stdout.txt` 기준**이며, 재실행하면 달라진다. **비율·순서·AP proxy만** 논문과 대조해 쓸 것.

---

## 1. 검증 A — NMS 비용 모델 (§3.1 · Table 1)

### 방법
- YOLO 계열을 흉내낸 검출 출력 2 000개(GT 60개, GT마다 중복 2~5개 + 배경 오탐) 합성.
- conf 임계값을 바꿔가며 **NMS 전 박스 수**를 재고, 표준 greedy NMS(numpy 직접 구현)의 **실제 실행 시간**과 **AP proxy**(NMS 후 남은 박스로 GT를 최대한 매칭했을 때의 재현율)를 측정.
- conf sweep: 0.001 / 0.01 / 0.05 / 0.25 (IoU 0.7 고정) · IoU sweep: 0.5 / 0.6 / 0.8 (conf 0.001 고정).

### 실측
```
conf=0.001   입력 2000 → NMS 후 1897   시간 57.03ms   AP proxy 1.000
conf=0.01    입력 2000 → NMS 후 1897   시간 57.23ms   AP proxy 1.000
conf=0.05    입력 1847 → NMS 후 1750   시간 50.47ms   AP proxy 0.983
conf=0.25    입력 1007 → NMS 후  945   시간 21.29ms   AP proxy 0.933
IoU=0.5      NMS 후 1493   시간 43.05ms   AP proxy 0.967
IoU=0.6      NMS 후 1761   시간 52.09ms   AP proxy 1.000
IoU=0.8      NMS 후 1969   시간 59.95ms   AP proxy 1.000
```

![NMS 비용 모델](https://github.com/todd-jang/paper-diary/blob/main/day08_rtdetr/runs/chart_nms_cost.png)

### 읽는 법
- **conf↑ → 시간↓(57.03 → 21.29ms)·AP proxy↓(1.000 → 0.933)** — 논문 Table 1의 conf 열(2.36 → 1.06ms, 52.9 → 51.2 AP)과 **같은 방향**. 어렵고 저점수인 GT가 임계값 위로 올라오지 못해 사라지는 것이 정확도의 대가다.
- **IoU↑ → 시간↑(43.05 → 59.95ms)·AP proxy↑(0.967 → 1.000)** — 논문 Table 1의 IoU 열(2.24 → 2.46ms, 52.1 → 52.8 AP)과 같은 방향. 중복이 많이 살아남을수록 정확도는 오르지만 시간이 든다.
- **결론**: NMS는 "임계값 하나로 정확도·시간을 동시 조율"하는 **hand-tuned 딜emma**이고, 두 임계값을 없앤 NMS-free는 이 딜emma 자체를 제거한다. RT-DETR의 §3.2 벤치마크(NMS 플러그인 포함)가 이 제거의 효과를 측정하는 장치다.
- **경계**: AP proxy는 **재현율 대용 지표**이지 COCO AP가 아니다(합성 GT·합성 오탐). 방향 확인용으로만 쓸 것.

---

## 2. 검증 B — Attention 토큰 스케일링 (§4.2, AIFI의 근거)

### 방법
- d=256, head=8로 scaled dot-product attention만 반복 실행하며 토큰 수 M을 400 → 800 → 1600 → 3200으로 늘려 **실측 시간** 측정. 1×1 conv(256ch, 80×80) 도 비교.

### 실측
```
M=  400 토큰   attention  2.37ms
M=  800 토큰   attention  8.28ms
M= 1600 토큰   attention 25.41ms
M= 3200 토큰   attention 100.19ms
→ 400 → 3200(8배) 에서 시간 42.3배 (이차 법칙 이론 64배, 실측은 메모리 한계로 저하)
→ 1x1 conv 80x80x256: 2.24ms   (attention 대비 파레토하게 저렴)
→ RT-DETR v1: 후보 8400 vs S5 400 = 441배 토큰 차
```

![Attention 토큰 스케일링](https://github.com/todd-jang/paper-diary/blob/main/day08_rtdetr/runs/chart_attention_scaling.png)

### 읽는 법
- 실측은 **이차 법칙을 대체로 따르되(42.3배 < 이론 64배)**, 이론보다 완만하다. 이론이 $\propto M^2$인 이유는 attention 가중치 행렬이 $M\times M$이라서다. 실측이 이론보다 낮은 이유는 큰 행렬의 메모리 대역폭·캐시 효율이 떨어지기 때문이다.
- **AIFI 결정의 체감 스케일**: 640×640 입력에서 후보 8 400 토큰(80²+40²+20²) vs AIFI가 보는 S5의 400 토큰(20²) = **441배**. attention 비용이 $M^2$이므로 **19만 배** 차이가 날 수 있다는 계산이 논문의 "인코더가 병목(49% GFLOPs, 11% AP)"이라는 진단과 정확히 맞물린다.
- **conv는 이 게임의다른 면**: CCFF가 conv 기반이라는 것은 비용이 $\propto HW$(선형)이어서 attention의 $M^2$와 **같은 해상도에서 파레토하게 싸다**는 의미다. 실측 1×1 conv 2.24ms vs 3 200토큰 attention 100.19ms.

---

## 3. 검증 C — 하이브리드 인코더 변형 A~E의 비용 순서 (§5.2 · Table 3)

### 방법
- 논문의 변형 A~E를 **미니 구현**(3스케일 합성 feature, d=64, 1 transformer 층)으로 재현: A(인코더 없음) · B(단일 스케일 transformer를 레벨별 공유) · C(멀티스케일 transformer, 공통 크기 패딩 후 concat) · D(intra/cross 분리 + PAN) · D_S5(intra를 S5만) · E(RT 하이브리드 + fusion block의 RepBlock).
- 각 변형의 **실측 지연·파라미터**를 논문 Table 3과 나란히 비교.

### 실측
```
변형     ours ms   ours M   paper ms   paper AP
A          0.00     0.00       7.2       43.0
B          0.90     0.05      11.1       44.9
C          2.55     0.05      13.3       45.6
D          1.03     0.083     12.2       46.4
D_S5       0.27     0.083      7.9       46.8
E          0.52     0.093      9.3       47.9
→ 지연 순서(ours) : A < D_S5 < E < B < D < C
→ 지연 순서(paper): A < D_S5 < E < B < D < C
```

![인코더 변형별 지연](https://github.com/todd-jang/paper-diary/blob/main/day08_rtdetr/runs/chart_encoder_variants.png)

### 읽는 법
- **지연 순서가 논문과 완전히 일치**했다. 특히 (i) C(멀티스케일 attention)가 **가장 비싸다**, (ii) **D_S5가 transformer 계열 중 가장 빠르다**, (iii) E는 D_S5보다 약간 느리지만 논문에서 **가장 정확하다**(47.9).
- **절대값은 비교 불가**다(하드웨어·입력 크기·구현이 다르므로 ours ms는 A가 0.00이고 논문은 7.2ms). **비교 가능한 것은 순서와 상대 비용뿐**이다.
- D_S5가 빠른 이유는 attention이 **S5의 400 토큰에만** 걸리기 때문이다(검증 B). E가 D_S5보다 느린 이유는 CCFF의 fusion block(RepBlock 포함)이 **합성곱 비용**을 추가하기 때문이다 — 즉 **"정확도 +0.4 AP의 대가가 무엇인지"**가 여기서 구조적으로 드러난다.
- **경계**: 이 미니 구현은 feature 해상도·채널·attention 층 수가 논문과 달라 **절대 지연은 재현 대상이 아니다**. 순서 일치는 "설계 논리가 일관된 방향으로 구현된다"는 것까지만 증명한다.

---

## 4. 검증 D — uncertainty 항의 퇴화 방향 (§4.3 식 2·3)

### 방법
- 분류 점수 $s$와 위치 품질 $q$를 **logistic 파라미터화**(항상 [0,1])로 두고, ① **U만** 최소화($U=\text{relu}(s-q)$, 분류 손실 없음) ② **U + 분류 NLL**($-\log s$) 최소화. 초기값 $s=0.90$, $q=0.20$.

### 실측
```
U만 최소화(분류 손실 없음) → 80스텝 후 score s=0.690, quality q=0.693
   (U=0으로 가는 해는 '두 값이 만나는 지점' — 확신 0.90→0.690, 품질 0.20→0.693)
분류 손실까지 포함하면    → score s=0.794, quality q=0.793
   (의도된 메커니즘: 확신은 상대적으로 유지되고 '위치 품질이 확신에 따라 올라간다')
```

![uncertainty 퇴화 방향](https://github.com/todd-jang/paper-diary/blob/main/day08_rtdetr/runs/chart_uncertainty_degeneracy.png)

### 읽는 법
- **U 단독 최적화의 퇴화 해**: $\arg\min\,\text{relu}(s-q)$는 $s=q$인 **어떤 점**이 최적이므로, 품질을 실제로 개선하는 대신 **확신을 낮춰(0.90 → 0.690) U를 0으로 만드는** 경로가 최적해가 된다. 실측이 이 퇴화 해를 정량으로 보여준다.
- **분류 손실을 병행하면** 실측 $s=0.794$, $q=0.793$로 **품질이 확신에 따라 올라간다**. 이것이 논문 식 (3)의 의도된 메커니즘이다.
- **왜 이 검증이 필요한가**: Day 07 실습 [D]에서 네가 관측한 "uncertainty 런에서 Prop cls가 1.000 → 0.127로 붕괴"가 **정확히 이 퇴화**였다. 즉 **U 항을 단독으로 손질하면 점수가 낮아져 U가 0이 된다**. RT-DETR은 분류·위치 손실과 함께 최적화하므로(식 3) 억제된다.
- **방법론 결론**: 논문의 $U=\|\mathcal P-\mathcal C\|$는 **분포 간 거리**라 이 퇴화가 구조적으로 덜 쉽게 열려 있고, 1차원 프록시로 만든 ablation은 **반드시 이 점검을 해야 한다**.

---

## 5. 코드 ↔ 검증 대응표

| 검증 | 함수 | 원문 근거 | 확인한 것 |
|---|---|---|---|
| A | `run_a()` | §3.1 · Table 1 | conf↑/IoU↑ 에 따른 시간·정확도 방향 (NMS 2임계값 딜emma) |
| B | `run_b()` | §4.2 · DETR 부록 0.A.3 | attention 비용의 $M^2$ 스케일링, 441배 토큰 차, conv의 파레토 우위 |
| C | `run_c()` | §5.2 · Table 3 | 변형 A~E의 **비용 순서 재현**(논문과 동일) |
| D | `run_d()` | §4.3 식 (2)(3) | uncertainty 항의 퇴화 해와, 분류 손실 병행 시의 의도된 메커니즘 |

## 6. 이 실습이 증명하는 것 / 하지 못하는 것

**증명함**
- NMS의 두 임계값은 정확도와 시간이 **반대 방향**으로 움직인다(논문 Table 1과 같은 방향).
- attention 비용은 토큰 수의 **제곱**에 비례하고, conv는 그렇지 않다 → AIFI(S5 한정)+CCFF(conv) 조합의 비용 논리가 성립한다.
- RT-DETR의 인코더 변형별 **상대 비용 순서**가 논문과 일치한다.
- uncertainty 항을 단독으로 쓰면 점수를 낮추는 퇴화 해가 존재한다 → 분류 손실 병행이 필요하다.

**증명하지 못함(한계 명시)**
- **정확도(AP)는 재현하지 않았다.** 합성 데이터·미니 모델·학습 없음이라 논문의 43.0~47.9 AP를 비교할 수 없다. 지연 순서만 비교했다.
- 지연의 **절대값**은 하드웨어·입력·구현이 달라 비교 불가(ours A 0.00ms vs 논문 7.2ms).
- AP proxy는 COCO AP가 아닌 재현율 대용 지표다.
- uncertainty 검증은 2변수 toy이며, 실제 인코더의 다차원 feature 분포를 재현하지 않는다.
- **지연의 절대값은 재현되지 않는다**(벽시계 측정, 2회 실행에서 최대 2배 편차 — 위 재현성 주의). 보고서에서 지연 수치를 쓰려면 반드시 "단일 실행 측정"임을 표기하고, 결론은 **비율·순서**에 근거할 것.
