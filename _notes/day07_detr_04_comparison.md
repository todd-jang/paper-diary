---
layout: default
title: "Day 07 DETR — 04 · 비교표 — DETR 원문·RT-DETR 원문 숫자 총정리"
paper_id: detr
date: 2026-10-20
---

# 04 · 비교표 — DETR 원문·RT-DETR 원문 숫자 총정리

> 원칙(README): **ablation은 같은 시스템·도메인 안에서 변수 하나만 바꿀 때만 성립.** 서로 다른 논문·다른 스케줄·다른 입력 크기의 숫자를 한 표에 놓고 우열을 말하면 안 된다. 아래 표들은 **성격별로 분리**했고, 마지막에 "같이 놓을 수 없는 것"을 명시했다.
> 모든 수치는 `day07_detr/src/paper.txt`(DETR) · `day07_detr/src/rtdetr.txt`(RT-DETR)에서 대조했다.

---

## 1. DETR Table 1 — COCO val, Faster R-CNN 대조 (원문 §4.1)

| 모델 | GFLOPS/FPS | 파라미터 | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN-DC5 | 320/16 | 166M | 39.0 | 60.5 | 42.3 | 21.4 | 43.5 | 52.5 |
| Faster R-CNN-FPN | 180/26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0 |
| Faster R-CNN-R101-FPN | 246/20 | 60M | 42.0 | 62.5 | 45.9 | 25.2 | 45.6 | 54.6 |
| Faster R-CNN-DC5+ | 320/16 | 166M | 41.1 | 61.4 | 44.3 | 22.9 | 45.9 | 55.0 |
| Faster R-CNN-FPN+ | 180/26 | 42M | 42.0 | 62.1 | 45.5 | **26.6** | 45.4 | 53.4 |
| Faster R-CNN-R101-FPN+ | 246/20 | 60M | 44.0 | 63.9 | 47.8 | 27.2 | 48.1 | 56.0 |
| **DETR (R50)** | 86/28 | 41M | 42.0 | 62.4 | 44.2 | **20.5** | 45.8 | **61.1** |
| DETR-DC5 | 187/12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| DETR-R101 | 152/20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8 |
| DETR-DC5-R101 | 253/10 | 60M | 44.9 | 64.7 | 47.7 | 23.7 | 49.5 | 62.3 |

![DETR vs Faster R-CNN AP 분해(논문 Table 1)](https://github.com/todd-jang/paper-diary/blob/main/day07_detr/runs/chart_detr_ap_breakdown.png)

**같이 놓을 수 있는 쌍:** DETR(41M, 500ep) vs Faster R-CNN-FPN+(42M, 9× 스케줄) — **파라미터가 사실상 같고(41M vs 42M)** AP 42.0 = 42.0. 이 쌍에서 읽을 것은 두 가지다: **AP_S 20.5 vs 26.6(−5.5)**, **AP_L 61.1 vs 53.4(+7.8)**. 원문도 *"achieving 42 AP on the COCO val subset"*로 정리한다.
- 파라미터 분해: ResNet-50 23.5M + transformer 17.8M = 41.3M.
- "+"가 붙은 Faster R-CNN은 GIoU + random crop + 9× 스케줄(109ep)로 보강된 버전이다. **보강된 베이스라인과 비교해야 DETR의 "동등"이 성립한다** — 보강 전(40.2) 대비라면 DETR가 1.8 AP 앞선다.
- DC5: 마지막 stage dilation → 해상도 2배, 인코더 self-attention 16배 비용, 전체 2배(86 → 187 GFLOPs). AP +1.3, FPS 28 → 12.

---

## 2. DETR Table 2 — 인코더 층 수 (같은 시스템 안, R50 · 6+6 · 300ep)

| 인코더 층 | GFLOPS/FPS | 파라미터 | AP | AP50 | APS | APM | APL |
|---|---|---|---|---|---|---|---|
| 0 | 76/28 | 33.4M | 36.7 | 57.4 | 16.8 | 39.6 | 54.2 |
| 3 | 81/25 | 37.4M | 40.1 | 60.6 | 18.5 | 43.8 | 58.6 |
| 6 | 86/23 | 41.3M | 40.6 | 61.6 | 19.9 | 44.3 | 60.2 |
| 12 | 95/20 | 49.2M | 41.6 | 62.1 | 19.8 | 44.9 | 61.9 |

- **읽는 법**: 인코더를 빼면 AP −3.9, **AP_L −6.0**(54.2 → 60.2). 손해는 큰 물체 쪽에 크다. 논문의 해석: 전역 attention이 **인스턴스 분리**를 돕는다(Fig 3).
- 6 → 12층은 AP +1.0인데 파라미터 +7.9M, FPS 23 → 20. 기울기가 줄어든 구간.

---

## 3. DETR Table 3 — 위치 인코딩 (같은 시스템 안, 300ep)

| 공간 enc — 인코더 | 공간 enc — 디코더 | 출력 enc — 디코더 | AP | Δ | AP50 | Δ |
|---|---|---|---|---|---|---|
| none | none | learned at input | 32.8 | −7.8 | 55.2 | −6.5 |
| sine at input | sine at input | learned at input | 39.2 | −1.4 | 60.0 | −1.6 |
| learned at attn. | learned at attn. | learned at attn. | 39.6 | −1.0 | 60.7 | −0.9 |
| none | sine at attn. | learned at attn. | 39.3 | −1.3 | 60.3 | −1.4 |
| sine at attn. | sine at attn. | learned at attn. | **40.6** | — | **61.6** | — |

- **기본(마지막 행)**: 공간 pos enc를 **매 attention 층**에 더한다. 논문 Fig 10·부록 0.A.3과 일치.
- **"sine at input"**(원래 transformer 방식, 1회 추가) → 39.2. **"at attn."(매 층) → 40.6. 차이 1.4 AP.**
- **인코더에서만 빼면** 39.3(−1.3) — 즉 **디코더에 공간 위치가 있으면 그나마 버틴다**는 뜻이고, 동시에 "인코더의 self-attention은 내용 중심, 디코더는 위치 중심"이라는 해석을 지지한다.
- 출력(object query) pos enc는 **제거 불가능** — 논문: *"Output positional encodings are required and cannot be removed."*

---

## 4. DETR Table 4 — 손실 구성 (같은 시스템 안, 300ep · R50)

| class | ℓ1 | GIoU | AP | Δ | AP50 | Δ | APS | APM | APL |
|---|---|---|---|---|---|---|---|---|---|
| ✓ | ✓ | | 35.8 | −4.8 | 57.3 | −4.4 | 13.7 | 39.8 | 57.9 |
| ✓ | | ✓ | 39.9 | −0.7 | 61.6 | 0 | 19.9 | 43.2 | 57.9 |
| ✓ | ✓ | ✓ | **40.6** | — | 61.6 | — | 19.9 | 44.3 | 60.2 |

- L1만: APS 13.7(대형 위주 학습), GIoU만: −0.7로 거의 따라잡되 **APL 57.9로 동일**, 둘 다: APL 60.2.
- 가중치는 고정(5 / 2)이고, 원문 스스로 *"other means of combining them may achieve different results"*라고 한계를 밝힌다.
- **이 표는 300ep ablation 점수**다. 논문 최종 42.0은 **500ep** Table 1 값 — 섞지 말 것.

## 5. DETR의 그 밖의 수치 (같은 시스템 안)

| 실험 | 결과 | 출처 |
|---|---|---|
| FFN 제거 | −2.3 AP, 파라미터 41.3M → 28.7M (transformer 10.8M) | §4.2 |
| 스케줄 300ep → 500ep | +1.5 AP (40.6 → 42.0) | §4 |
| random crop (p=0.5) | +1 AP | §4 |
| 추론 시 ∅ 슬롯 → 2위 클래스로 대체 | +2 AP (필터링보다) | §4 |
| 디코더 층 1 → 6 | +8.2 AP, AP50 +9.5 | Fig 4 |
| NMS 적용 시 | 1층에서는 **도움**(중복 제거), 마지막 층에서는 **해침**(TP 제거) | Fig 4 |
| 펜옵틱 PQ | DETR R50 43.4 / DC5 44.6 / R101 45.1 (PQth 50.5, PQst 37.0) | Table 5 |
| 펜옵틱 test set | 46 PQ | §4.4 |
| COCO 평균 객체 수 | 7개/이미지 (최대 63) | §4 |

---

## 6. RT-DETR Table 3 — 하이브리드 인코더 변형 (같은 시스템 안, 1× 설정, R50)

| 변형 | AP | 파라미터 | 지연(ms) | 직전 대비 해석 |
|---|---|---|---|---|
| A (멀티스케일 인코더 제거) | 43.0 | 31M | 7.2 | 기준선 |
| B (단일 스케일 transformer 추가) | 44.9 | 32M | 11.1 | +1.9 AP / 지연 +54% |
| C (+멀티스케일 transformer) | 45.6 | 32M | 13.3 | +0.7 AP / +20% |
| D (intra·cross 분리, PAN 방식) | 46.4 | 35M | 12.2 | +0.8 AP / **−8%** |
| D_S5 (intra를 S5에만) | 46.8 | 35M | 7.9 | +0.4 AP / **−35%** |
| E (RT-DETR 하이브리드) | 47.9 | 42M | 9.3 | +1.5 AP / **−24%** |

- **가장 중요한 두 걸음**: (1) D: 인코더/디코더 분리를 통해 **정확도↑ + 지연↓를 동시에** 얻음, (2) D_S5: 하위 레벨 self-attention을 빼고 **지연 35% 절감 + 정확도까지 +0.4**.
- E는 D 대비 파라미터 +7M이지만 지연은 24% 낮다(Conv가 attention보다 싸면서 융합 성능은 유지).
- 맥락 숫자: Deformable-DETR에서 **인코더가 GFLOPs의 49%, AP 기여는 11%** — 병목 인식이 설계 출발점.

![RT-DETR 하이브리드 인코더 변형 trade-off(논문 Table 3)](https://github.com/todd-jang/paper-diary/blob/main/day07_detr/runs/chart_rtdetr_encoder_tradeoff.png)

## 7. RT-DETR Table 4 — 쿼리 선택 (같은 시스템 안, RT-DETR-R50 1×)

| 쿼리 선택 | AP | Prop cls ↑ | Prop both ↑ |
|---|---|---|---|
| Vanilla (분류 점수) | 47.9 | 0.35% | 0.30% |
| Uncertainty-minimal (식 2·3) | **48.7** | **0.82%** | **0.67%** |

- +0.8 AP. `Prop cls` = 분류 점수 > 0.5 비율, `Prop both` = 분류·IoU **둘 다** > 0.5 비율.
- Fig 6: uncertainty(보라)가 우상단(분류·위치 모두 높음)에 몰림. 분포 우측에서 138% 더 많은 보라 점, 두 점수 모두 > 0.5인 영역에서 120% 더 많음.

## 8. RT-DETR Table 5 — 디코더 깊이 (같은 시스템 안, RT-DETR-R50 6×)

| 학습 층수 | 사용 층 1 | 2 | 3 | 4 | 5 | 6 | 7 | 지연(ms) |
|---|---|---|---|---|---|---|---|---|
| Det 7 | 48.3 | 50.6 | 51.5 | 52.1 | 52.5 | 52.6 | 52.6 | 9.6 |
| Det 6 | 49.1 | 51.3 | 52.4 | 52.7 | 53.0 | **53.1** | — | 9.3 |
| Det 5 | 48.8 | 51.3 | 52.3 | 52.7 | 52.9 | — | — | 8.8 |
| Det 4 | 49.6 | 51.6 | 52.4 | 52.7 | — | — | — | 8.3 |

- **Det 6 기준**: 6층 53.1 AP / 9.3ms → 5층에서 중단 53.0 AP / 8.8ms(−0.1 AP, −0.5ms) → 1층 49.1 AP / 7.0ms.
- **재학습 없이** 조절 가능(원문 DETR의 aux loss가 모든 층에서 예측을 학습시키기 때문). 2층에서 4층으로의 AP 이득이 1.4이지만 5→6층은 0.1 — **위쪽 층의 기여가 급감**한다.
- 주의: Det 4/5/7 행은 **학습한 층수가 다른 별도 모델**이므로, 행 간 비교는 "학습 설정"이 다르고 열 내 비교는 "사용 층"만 다르다. 이 둘을 섞지 말 것.

## 9. RT-DETR Table 1 — NMS 임계값 민감도 (YOLOv8, T4 · TensorRT FP16)

| IoU 임계값 (conf=0.001) | AP | NMS(ms) | | conf 임계값 (IoU=0.7) | AP | NMS(ms) |
|---|---|---|---|---|---|---|
| 0.5 | 52.1 | 2.24 | | 0.001 | 52.9 | 2.36 |
| 0.6 | 52.6 | 2.29 | | 0.01 | 52.4 | 1.73 |
| 0.8 | 52.8 | 2.46 | | 0.05 | 51.2 | 1.06 |

- NMS는 **두 임계값**에 종속되고, 임계값을 올리면 정확도(AP)는 오르고 시간은 늘어난다(0.05 conf: −1.7 AP, 1.30ms 절약). 즉 "빠르게 하려면 정확도를 내줘야 한다"는 딜emma가 NMS에 내장돼 있다.
- anchor-based는 anchor-free보다 **3배 많은 박스**를 내므로 NMS 시간이 더 걸린다(원문 §3.2).
- **NMS-free(DETR 계열)의 실질적 이득**: 하이퍼파라미터 두 개가 사라진다.

---

## 10. 같이 놓으면 안 되는 것 (경계 명시)

| 비교 | 왜 같은 줄이 아닌가 |
|---|---|
| DETR 42.0 AP(R50, 500ep, 800×1333) vs RT-DETR-R50 53.1 AP(72ep, 640×640) | **입력 크기·스케줄·백본 변형(ResNet50-vd)·학습 데이터가 전부 다르다.** 같은 "R50"이라는 문자열만 공유. 서술하면: "RT-DETR은 다른 시스템이다"가 아니라 "두 논문의 Table 값은 동일 조건 비교가 아니다"로 쓴다. |
| DETR 42.0(500ep) vs DETR 40.6(300ep) | 같은 시스템 안의 **스케줄 변수** → 성립. 1.5 AP 차이는 학습 길이 때문이다. |
| DETR Table 4(35.8/39.9/40.6) vs Table 1(42.0) | 300ep ablation vs 500ep 최종. §4.2 절이 "the model has 40.6 and 42.0 AP on short and long schedules"로 둘을 함께 적어준다. |
| DETR vs RT-DETR의 "AP_L 61.1 vs 70.0" | 백본 변형(R50 vs R50-vd), 입력 크기, 에폭이 모두 다름. **성질 비교**(AP_L advantage)만 가능, 크기 비교는 불가. |
| DETR의 matcher 비용(-p̂) vs RT-DETR의 VarFocal 비용 | **구현이 다르다.** RT-DETR는 classification이 VFL이고 매칭 비용도 focal 형태(`matcher.py:89-91`), cost_class 가중치도 2(_DETR는 1). 01_코드워크스루 §6.3 참조. |
| 실습 [D]의 2.59x vs RT Table 4의 2.2x | **시뮬레이션 vs 실측.** 방향성만 비교 가능. |

---

## 11. 대조 완료 기록

`src/paper.txt` / `src/rtdetr.txt`에서 grep 대조한 항목:
DETR — Table 1의 10행 전체 · Table 2의 4행 · Table 3의 5행 · Table 4의 3행 · Table 5(펜옵틱) 4행 · Fig 4의 +8.2/+9.5 · FFN −2.3(41.3M→28.7M) · 300/500ep 스케줄(+1.5) · random crop(+1) · ∅ 슬롯 대체(+2) · λ_L1=5·λ_iou=2 · ∅ 하향 10 · N=100 · 평균 7객체 · DC5 16x/2x.
RT-DETR — Table 1(6행) · Table 2의 RT-DETR 2행(53.1/108FPS, 54.3/74FPS, 42M/136GFLOPs, 76M/259GFLOPs) · Table 3(A~E 6행) · Table 4(2행) · Table 5(4행) · 인코더 49%/11% · K=300 · DINO-Deformable-R50 50.9 AP / 5 FPS · Objects365 프리트레인 55.3/56.2.
