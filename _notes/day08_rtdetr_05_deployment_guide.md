---
layout: default
title: "Day 08 RT-DETR — 05 · 보충: 실전·배포 가이드 — RT-DETR를 실제로 돌리는 3가지 경로"
paper_id: rt-detr
date: 2026-10-21
---

# 05 · 보충: 실전·배포 가이드 — RT-DETR를 실제로 돌리는 3가지 경로

> 목표: 논문(2304.08069)을 읽은 뒤 **"내 손에서 무엇을 돌려야 하고, 무엇을 주의해야 하는가"**를 정리. 모든 코드 사실은 로컬 설치본을 직접 읽어 확인했다.
> 검증 대상: `lyuwenyu/rt-detr`(공식, Apache-2.0) · `ultralytics 8.4.183`(.venv에 설치됨) · 사용자 보유 `~/rtdetr-l.pt`(66.5MB)

---

## 1. 경로 ① 공식 저장소 (Apache-2.0) — 논문 재현용

| 트랙 | 경로 | 용도 |
|---|---|---|
| v1 PyTorch | `rtdetr_pytorch/` | 학습·검증(논문 Table 2의 6× 설정 재현) |
| v1 Paddle | `rtdetr_paddle/` | **논문 실험의 원본 구현**(weight·config가 여기서 나옴) |
| v2 PyTorch | `rtdetrv2_pytorch/` | deformable attention + denoising, 배포 예시 포함 |
| v2 Paddle | `rtdetrv2_paddle/` | v2의 paddle 구현 |
| benchmark | `benchmark/` | §3.2의 **end-to-end speed benchmark**(NMS 포함 속도) |

- README가 가리키는 배포 discussion: **onnxruntime / TensorRT / OpenVINO** 지원(2023-10-23 추가). v2는 `references/deploy/`에 torch/onnxruntime/openvino/tensorrt **4종 스크립트**가 들어 있다.
- **v1 PyTorch는 "paddle 버전에서 변환한 weight"**를 쓴다(README 2023-09-19 항목) — 즉 **정확한 재현은 paddle 쪽**, pytorch는 실전 편의용이다.

## 2. 경로 ② Ultralytics RT-DETR (사용자 보유 `rtdetr-l.pt`의 정체)

- 설치된 버전: **ultralytics 8.4.183**. 모델 스펙은 `cfg/models/rt-detr/rtdetr-l.yaml`에 있다.
- **백본이 ResNet이 아니다.** `rtdetr-l.yaml`은 **HGNetV2**(HGStem + HGBlock 4 stage, P3/8~P5/32)다. 즉 논문 §4.4의 "scaled RT-DETR"(백본을 바꿔 스케일 변형) 쪽 구현이며, **공식 R50/R101 체크포인트와 다른 계열**이다.
- **헤드는 논문 구조 그대로**: `input_proj(1×1 Conv)` → **AIFI** → `lateral_convs` → `fpn_blocks(RepC3)` → `downsample_convs` → `pan_blocks(RepC3)` → `RTDETRDecoder(P3,P4,P5)`.
- **쿼리 선택도 분류 점수 기준** — 논문·공식 구현과 **세 번째 코드 경로가 같은 결론**을 낸다:

```python
# ultralytics/nn/modules/head.py:1830
topk_ind = Detect._grouped_topk(enc_outputs_scores.max(-1).values, k, groups)[1].view(-1)
```

- **NMS가 없다**: `models/rtdetr/model.py`·`predict.py`에 `nms` 참조가 없고, 후처리는 **신뢰도 필터 + grouped top-k**뿐이다.
  ```python
  # models/rtdetr/predict.py:61
  idx = score.squeeze(-1) > self.args.conf
  ```
- **사용법(요약)**
  ```python
  from ultralytics import RTDETR
  model = RTDETR("rtdetr-l.pt")
  res = model.predict(source="img.jpg", conf=0.25)   # conf 임계값은 남는 유일한 손잡이
  ```
- **라이선스 주의**: Ultralytics는 **AGPL-3.0**(상업·폐쇄 제품 embedding 시오픈소스 disclose 의무 발생), 공식 `lyuwenyu/rt-detr`는 **Apache-2.0**. 실전에 넣기 전 라이선스부터 확인.

## 3. 경로 ③ 배포 최적화 (TensorRT/ONNX)

- **가장 큰 이득**: NMS가 없으므로 **후처리 파이프라인이 단순**하고, 임계값이 하나 줄었다(conf만). YOLO와 달리 IoU 임계값이 없어서 "NMS 튜닝"이라는 실전 비용이 사라진다.
- **내보내기 시 주의 3가지**
  1. **입력 크기**: 학습·벤치마크는 **640×640**. 임의 해상도에서 돌리면 letterbox/stretches가 어긋나 성능이 떨어진다 → 배포 타깃 해상도를 640으로 고정하거나, v2처럼 다중 스케일 대응을 확인한다.
  2. **max_det**: ONNX/TensorRT export 시 `max_det`로 출력 개수를 제한한다(헤드의 `postprocess`가 `k = min(num_queries, max_det)`로 자른다). 300쿼리 전체를 내보내지 않으면 그래프가 가벼워진다.
  3. **FP16**: 논문 FPS는 **T4 · TensorRT FP16 · bs=1** 기준이다. CPU나 FP32로 비교하면 논문과 다른 값이 나온다.
- **v2를 쓸지 v1을 쓸지**: v2는 deformable attention + denoising으로 정확도가 높지만, **deploy 스크립트가 v2 기준**(`references/deploy/rtdetrv2_tensorrt.py` 등)이고 논문 v1의 FPS 수치는 v1 기준이다 — 혼용하지 말 것.

## 4. 실전 체크리스트 (이번 논문에서 바로 가져갈 것)

| 항목 | 내용 | 근거 |
|---|---|---|
| ① 후처리 파라미터 | **conf 하나만** 튜닝한다 (IoU 임계값 없음) | §3.1 Table 1 (NMS 2손잡이 제거) |
| ② 속도 조절 | **디코더 층을 잘라** 재학습 없이 FPS를 올린다 (6층 53.1/9.3ms → 5층 53.0/8.8ms) | §5.4 Table 5, `eval_idx` |
| ③ 정확도-속도 절충 | 지연 −35%·AP +0.4는 **인코더를 S5로 한정**(AIFI)해서 얻은 값 | §4.2 Table 3 (D_S5) |
| ④ 소형 객체 | **여전히 약점**(AP_S 34.8, YOLOv8-L보다 −0.5). 소형이 중요한 서비스라면 다른 선택지를 검토 | §6 Limitation |
| ⑤ 학습 데이터 | Objects365 사전학습 시 +2.2 AP (55.3) — 라벨 커버리지가 곧 성능 | §5.1 |
| ⑥ NMS 없는 후처리 | 라벨이 있는 프레임에서만 평가(Video Object Detection IoU)하면 NMS-free가 **불리**할 수 있음 | §3.1 Video Object Detection 언급 |

## 5. 3D로 확장할 때 (다음 트랙 연결)

- RT-DETR 헤드 자체는 2D 전용(정규화 cxcywh + sigmoid)이라 3D로 바로 못 쓴다. 2026년 실사 예로 **PillarDETR**(2606.01757, YOLO 백본 + RT-DETR 헤드 → 실시간 3D)가 있고, 3D DETR 계열은 `3D_OBJECT_DETECTION_지도.md` 참고.
- 3D로 갈 때 RT-DETR의 설계가 그대로 재사용되는 부분: **하이브리드 인코더(멀티스케일 + 의미집중 attention)**, **NMS-free 세트 예측**, **쿼리 기반 초기화**. 바뀌는 부분: 깊이 추정, 3D 좌표/회전(yaw) 파라미터화, BEV 표현, 멀티뷰.

## 6. 오늘 실습(03_실습)과의 연결

- 실습 [A]가 재현한 "conf↑ → 시간↓·정확도↓ / IoU↑ → 시간↑·정확도↑"가 **위 체크리스트 ①의 근거**이고, 실측 NMS 시간 57.03 → 21.29ms(−63%)는 "NMS를 없애면 줄어드는 시간"의 **모형**이다(실제론 T4 GPU 기준 2.36 → 1.06ms 스케일이지만 **방향과 손잡이 수**가 같다는 점이 핵심).
- 실습 [C]의 **지연 순서 재현**은 ②·③의 근거가 된다: AIFI는 S5 한 곳(400토큰)만 보므로 attention 비용이 $M^2$로 지배될 때 가장 싼 선택이다.
- 실습 [D]의 퇴화 검증은 **uncertainty 항을 단독 손실로 쓰면 안 된다**는 실전 단서이며, 공식 구현이 분류 손실과 함께 최적화하므로 문제가 없다는 점을 확인해 준다.
