# 🗺️ 3D 객체 탐지 로드맵 (논문일기 보조 트랙)

> **성격:** 3D 객체 탐지(DETR 3D)를 **별도 트랙**으로 잡기 위한 지도(landscape). Day 07(2D DETR)과는 별개이며, 커리큘럼 어디에 끼울지는 아래 §6에서 제안한다.
> **ID 검증 원칙:** 아래 arXiv ID는 **직접 조회해 제목을 확인한 것만** 적었다. 확인하지 못한 고전 논문(SECOND · LSSD · CM3D 등)은 **ID를 비워 두고 제목+연도만** 적었다(추측 ID는 쓰지 않는다).
> **확인 일자:** 2026-10-20 (arXiv API·abs 페이지 조회 기준)
> **진행 10/22:** 계보 선두 3편(DETR3D·PETR·PETRv2·Sparse4D v1/v2)의 조사 초안을 작성했다 → `day09_3d_fsd/01_초안.md` + 합본 포스터 `poster/poster_3d_space.html`. §3 표의 ID·수치는 라이브러리니안 팩트시트(abs·본문 표·공식 README 교차검증)로 재확인 — 오기 가능성 있던 ID 4건 교정, 미검증 수치 5종(DETR3D "46.2/53.5"·"GPU-center"·Sparse4D "15.9 FPS"·"no NMS" 명시·"depth sort")은 배제 확정. StreamPETR(2303.11926) 이후 계보는 차기 세션 분량.

---

## 1. 왜 이 트랙인가 — 2D DETR에서 3D로 뛸 때 바뀌는 것

| 축 | 2D (DETR) | 3D (DETR3D·PETR·StreamPETR) |
|---|---|---|
| 입력 | 단일 이미지 | **다중 카메라(서라운드 뷰 6~7대)** + LiDAR |
| 좌표계 | 이미지 평면 (x,y,w,h) | **3D 박스 (x,y,z,w,h,l,yaw)** — yaw 회전이 필수 |
| 표현 | 1개 스케일(또는 2~3) | **BEV(Bird's-Eye-View)** 로 올리는 게 표준 |
| 깊이 | 암묵적으로 결정 | **깊이 추정 문제가 그대로 남음** (이는 3D DETR의 최대 약점) |
| 쿼리 | 학습된 슬롯 100개 | **3D-to-2D 쿼리**(카메라별 reference point) 또는 **3D 위치 임베딩** |
| 매칭 | 헝가리아(2D) | 헝가리아(3D) — 이건 그대로 물려받음 |
| NMS | 없음 | 없음(계속) |
| 시간 | 없음 | **시간 정보(이전 프레임 쿼리 전파)** 가 핵심 축으로 추가됨 |

**핵심 인사이트:** 3D DETR은 "2D DETR을 3D로 옮긴" 게 아니라, **"쿼리를 3D 공간에 정의하는 방법"**을 바꾼 계열이다. DETR3D→PETR→StreamPETR→Sparse4D는 이 축의 진화다.

## 2. LiDAR 계열 — 점/복셀/필러 → 중심점 → 트랜스포머

| 논문 | arXiv | 계보 | 핵심 |
|---|---|---|---|
| VoxelNet | [1711.06396](https://arxiv.org/abs/1711.06396) | 2017 | 복셀 CNN + VFE. 점군→/learned features |
| SECOND | (ID 미검증) | 2018 | **Sparse 3D conv**로 속도, anchor-based head |
| PointRCNN | [1812.04244](https://arxiv.org/abs/1812.04244) | 2018 | 3D proposal + ROIs |
| PointPillars | [1812.05784](https://arxiv.org/abs/1812.05784) | 2019 | **Pillar Feature Net** — BEV pseudo-image. 실전 표준 |
| PV-RCNN | [1912.13192](https://arxiv.org/abs/1912.13192) | 2019 | Voxel Set Abstraction + keypoint ROI |
| PV-RCNN++ | [2102.00463](https://arxiv.org/abs/2102.00463) | 2021 | semantic feature interaction |
| Deformable PV-RCNN | [2008.08766](https://arxiv.org/abs/2008.08766) | 2020 | deformable ROI |
| CenterPoint | [2006.11275](https://arxiv.org/abs/2006.11275) | 2020 | **center-based** 1-stage + 트래킹까지 확장. nuScenes 기본값 |
| TransFusion | [2203.11496](https://arxiv.org/abs/2203.11496) | 2023 | LiDAR+Camera **transformer 융합** |
| CenterFormer | [2209.05588](https://arxiv.org/abs/2209.05588) | 2022 | center-based **transformer** |
| VoxelNeXt | [2303.11301](https://arxiv.org/abs/2303.11301) | 2023 | fully-sparse, voxel 기반 재설계 |

## 3. 카메라 / BEV 계열 — DETR이 실제로 상륙하는 곳 ★

| 논문 | arXiv | 쿼리 정의 | 시간 |
|---|---|---|---|
| BEVDet | [2112.11790](https://arxiv.org/abs/2112.11790) | (트랜스포머 이전) BEV lift | – |
| **DETR3D** | [2110.06922](https://arxiv.org/abs/2110.06922) | **3D-to-2D 쿼리**: 3D reference point를 카메라 평면에 투영해 feature를 뽑음 | – |
| 3D-MAN | [2103.16054](https://arxiv.org/abs/2103.16054) | multi-frame attention | O |
| **PETR** | [2203.05625](https://arxiv.org/abs/2203.05625) | **Position Embedding Transformation**: 3D 좌표를 임베딩으로 | – |
| MUTR3D | [2205.00613](https://arxiv.org/abs/2205.00613) | 3D-to-2D query + **트래킹 쿼리 전파** | O |
| **BEVFormer** | [2203.17270](https://arxiv.org/abs/2203.17270) | BEV space 쿼리 + **spatiotemporal attention** | O |
| **StreamPETR** | [2303.11926](https://arxiv.org/abs/2303.11926) | **object-centric temporal modeling** — 쿼리를 프레임 간 전파 | O |
| **Sparse4D** | [2211.10581](https://arxiv.org/abs/2211.10581) | **fully sparse**: 쿼리마다 4D anchor + instance pooling | O |
| **Far3D** | [2308.09616](https://arxiv.org/abs/2308.09616) | **surround-view 장거리** 쿼리 확장 | O |
| Uni3DETR | [2310.05699](https://arxiv.org/abs/2310.05699) | 통합 3D detection transformer | – |
| MapTRv2 | [2308.05736](https://arxiv.org/abs/2308.05736) | 벡터화 HD map + 3D detection 동시 | – |
| VAD | [2303.12077](https://arxiv.org/abs/2303.12077) | 벡터 씬 표현으로 planning까지 | O |
| UniAD | [2212.10156](https://arxiv.org/abs/2212.10156) | planning-oriented 통합 (detection→planning) | O |

**3D DETR 4세대 요약**
1. **DETR3D(2021)** — 쿼리를 3D point로 정의하고 카메라에 투영. 깊이 추정 문제를 **쿼리 공간에 명시적으로 주입**한 최초의 구조.
2. **PETR(2022)** — 투영 대신 **3D 좌표를 임베딩으로 변환**(position embedding transformation). 카메라 개수와 해상도에 덜 예민.
3. **StreamPETR(2023)** — 시간 축. 쿼리를 **다음 프레임으로 전파**해 "한 객체 = 한 쿼리"의 정체성을 시간에 걸어 유지.
4. **Sparse4D / Far3D(2023)** — 쿼리를 **anchor 기반 sparse 집합**으로. BEV 그리드 대신 쿼리만 계산해 계산량 절감 + **거리 확장**(Far3D).

## 4. 점유(Occupancy) · 멀티모달 융합 · 2026 최신

| 논문 | arXiv | 축 |
|---|---|---|
| SurroundOcc | [2303.09551](https://arxiv.org/abs/2303.09551) | 멀티카메라 **3D semantic occupancy** |
| GaussianFormer | [2405.17429](https://arxiv.org/abs/2405.17429) | 장면을 **가우시안**으로 표현해 occupancy |
| GaussianFormer-2 | [2412.04384](https://arxiv.org/abs/2412.04384) | 확률적 가우시안 중첩, 효율 |
| BEVFusion | [2205.13790](https://arxiv.org/abs/2205.13790) | LiDAR+Camera → **통합 BEV** 융합 |
| FUTR3D | [2203.10642](https://arxiv.org/abs/2203.10642) | 통합 센서 융합 트랜스포머 |
| PillarDETR | [2606.01757](https://arxiv.org/abs/2606.01757) | **YOLO 백본 + RT-DETR 헤드** → 실시간 3D (오늘 D7과 직결) |
| Learned NMS for 3D | [2606.03568](https://arxiv.org/abs/2606.03568) | 3D에서도 NMS 학습 대체 시도 |
| MonoMIP | [2610.19071](https://arxiv.org/abs/2610.19071) | 모노쿨러 3D, 높이 기반 IPM |
| TADP | [2608.27282](https://arxiv.org/abs/2608.27282) | **deformable prediction** 싱글스테이지 3D |
| Mamba 기반 경량 3D | [2608.03490](https://arxiv.org/abs/2608.03490) | 경량화(실전) |
| Sparse4D-Radar | [2607.04098](https://arxiv.org/abs/2607.04098) | 4D 레이더 결합 |
| Open-vocab 3D 검출 | [2609.19358](https://arxiv.org/abs/2609.19358) · [2608.19973](https://arxiv.org/abs/2608.19973) | 프롬프트 가능 open-vocabulary |
| Map-Det3D | [2608.12179](https://arxiv.org/abs/2608.12179) | 맵 prior를 3D recon prior로 |
| PointLAM | [2609.21780](https://arxiv.org/abs/2609.21780) | point-based Mamba |
| PQR3D | [2609.32163](https://arxiv.org/abs/2609.32163) | 점진적 쿼리 정련(시간축) |
| Sparse2comm | [2610.08573](https://arxiv.org/abs/2610.08573) | V2X 협력 3D |

> **주의:** 2026년 항목은 대부분 arXiv preprint 동시 제출이라 **피어 리뷰 전**이다. 성능 수치를 인용하기 전에 원문 표를 직접 확인할 것.

## 5. 서베이(지도 읽기용)

| 논문 | arXiv | 범위 |
|---|---|---|
| 3D Object Detection for Autonomous Driving: A Comprehensive Survey | [2206.09474](https://arxiv.org/abs/2206.09474) | LiDAR/카메라/멀티모달 전부, IJCV 게재 |
| Multi-Modal 3D Object Detection: a Survey | [2106.12735](https://arxiv.org/abs/2106.12735) | **융합 시점·입력·granularity** 축 |
| A Survey of Robust 3D Object Detection Methods in Point Clouds | [2204.00106](https://arxiv.org/abs/2204.00106) | 강건성(야간·비·안개·공격) |
| A Comprehensive Review of 3D Object Detection (2024) | [2408.16530](https://arxiv.org/abs/2408.16530) | 최신 흐름 + E2E/occupancy/V2X까지 |

## 6. 오늘(Day 07)과의 연결 — 학습 순서 제안

```
D7  DETR(2D)  ← 오늘. 세트 예측·헝가리아·쿼리·NMS-free의 원리
D8  Deformable DETR ← 멀티스케일 attention(3D의 멀티뷰와 동일한 문제)
D9  DAB / DN-DETR  ← 쿼리를 anchor로, denoising으로
D10 DINO / RT-DETR ← 실시간·쿼리 최적화 (오늘 3D 실전 동향과 직결)
─────────────────────────────────────────────
3D 트랙(제안) ① DETR3D ② PETR ③ StreamPETR ④ Sparse4D ⑤ Occupancy(SurroundOcc)
```

**왜 이 순서인가:** 3D DETR의 핵심 난제는 **깊이**다. DETR3D는 "3D reference point를 카메라에 투영"해 깊이를 쿼리에 실었다 → PETR은 "3D 좌표를 임베딩으로" → StreamPETR은 "시간까지" → Sparse4D는 "쿼리 자체를 sparse anchor로". 이 축이 곧 DETR 2D의 진화 축(고정 쿼리 → 인코더 피처 쿼리 → anchor 쿼리)과 **동형**이다. 그러니 D7~D10을 마친 뒤에 붙이면 개념 이동이 없다.

**실전 가치 기준(학생용):**
- **실전 배포** → PillarDETR(2606.01757), Sparse4D, CenterPoint 계열
- **논문 이해(개념)** → DETR3D → PETR → StreamPETR 순이 필수
- **특수 분야** → Occupancy(SurroundOcc/GaussianFormer), V2X, open-vocabulary

## 7. 데이터셋·지표(짧게)

| 데이터셋 | 특징 | 지표 |
|---|---|---|
| KITTI | LiDAR 단일, 소규모, 오래됨 | AP3D(R11) / BEV AP, IoU 0.7(car)·0.5(ped/cyc) |
| nuScenes | 6카메라 + LiDAR + radar, 32-beam | **NDS**(종합), mAP, CVPRillow |
| Waymo Open | 5카메라(4화각)+LiDAR, 대규모 | mAP, 4D |
| Argoverse 2 | 7카메라 + 2×LiDAR, 센서 동기화 | mAP, 3D tracking |

- **3D mAP의 함정:** IoU 임계값(0.5 vs 0.7), BEV vs 3D IoU, yaw 각도 오차 처리(회전 기반 IoU) 때문에 **수치만 놓고 다른 논문끼리 비교하면 안 된다**(Diary README의 "도메인 교차 비교 금지"와 같은 원칙). nuScenes NDS는 AP·mIOU·CENTER 거리·오버랩을 합성한 단일 지표라 논문 간 비교에 그나마 편하다.

## 8. 미해결(|ID 미검증이라 제목만)

SECOND(Sparsely Embedded Convolutional Detection) · LSSD(Learning to Sample 3D Object Proposals) · CM3D(Camera Aided 3D Object Detection) · DSVT(Dynamic Sparse Voxel Transformer) · 3D-MAN은 위 표에 있음(2103.16054).
→ 이 다섯 편을 읽을 때는 arXiv에서 **제목으로 재검색**해 ID를 채울 것.
