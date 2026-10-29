---
layout: default
title: "Day 09 — 카메라 3D 탐지 조사: DETR3D → PETR → Sparse4D"
date: 2026-10-22
---

# Day 09 — 자율 세션: 카메라 3D 탐지 조사 초안 + 합본 포스터 (DETR3D → PETR → Sparse4D)

> **2026-10-22 (자율 세션)** · 주제: 카메라 전용 멀티뷰 3D 물체 탐지(nuScenes)의 3D→2D 투영 계보
> 대상: DETR3D (2110.06922, CoRL 2021) → PETR (2203.05625, ECCV 2022) → PETRv2 (2206.01256, ICCV 2023) → Sparse4D (2211.10581) → Sparse4D v2 (2305.14018)
> 진행 방식: **자율 세션** — Quest 형식 없음. 이 노트는 세션 기록을 2026-10-23에 소급 정리한 요약이다.
> 가설: **"Lift를 어떻게 하느냐"가 이 계열의 분수령** — 명시적 투영 인덱싱 → 사전 연산 3D PE → 앵커 중심 희소 샘플링 + 순환 전파.
> 이전 Day 연결: **Day 08(RT-DETR)로 DETR 계열 트랙 종료** → D9 3D 확장(개념 지도: [`3D_OBJECT_DETECTION_지도.md`](./3D_OBJECT_DETECTION_지도.md)) → 다음은 D10 GLIP/SAM(10/23) · D11 자율(10/24) · D12 CLIP(10/23 당김) · D13 LLaVA(10/26).

## 🧪 산출물
- 📜 [`day09_3d_fsd/01_초안.md`](./day09_3d_fsd/01_초안.md) — 조사 초안. **인용 수치 전용 원칙**: 모든 숫자는 원 논문 표·초록 또는 공식 README에서 직접 확인 + **조건(분할·백본·해상도·GPU) 명시**. 우리 환경 재측정 수치 0건.
- 🗺️ [`3D_OBJECT_DETECTION_지도.md`](./3D_OBJECT_DETECTION_지도.md) — 3D 트랙 개념 지도(10/20 작성 · 링크 44편 arXiv ID 검증 완료).
- 🧾 **포스터 키트 `poster/`** — `poster_3d_space.html`(9카드) · A0/A1 PDF 빌더(`build_poster.py`) · 문안 · 슬라이드 3장 · 레이아웃 겹침 검사(`qa_poster.py`).
- 📄 **A1L/A0L PDF+PNG 4종** — `poster/runs/poster_3d_space_A1L.pdf`/`.png` · `poster_3d_space_A0L.pdf`/`.png` · QA PASS.
- 근거 선: 라이브러리니안 팩트시트 — **오기 arXiv ID 4건 교정(abs 페이지 재검증) · 미검증 수치 배제**.

## 📊 계보 한눈에 보기 (nuScenes test · 최고 구성)

> 다섯 모델 모두 대형 백본(V2-99/Vov-99)+외부데이터 포함 최고 구성 — **절대 비교가 아니라 조건부 경향만**. 상세 표(분할·백본·조건 전체)는 초안 §3~§4.

| | DETR3D | PETR | PETRv2 | Sparse4D | Sparse4D v2 |
|---|---|---|---|---|---|
| Lift 방식 | 3D 쿼리 → 변환행렬로 2D 피처 인덱싱 | 3D 좌표 → MLP → **3D PE를 피처에 합산** | 3D PE 확장으로 **시간 정렬** | 앵커 **4D 키포인트** 희소 샘플링 | 앵커 **순환 전파**(ego motion 재투영) |
| 디코더 | 레이어마다 재투영 | 전역 어텐션 **원샷** | PETR + 작업별 쿼리 | cascade 6 모듈 + 시간 스택 | 시간 융합 **재귀형** |
| 시간 팔 | 1 | 1 | 2 | T (스택) | T (순환, O(T)→O(1)) |
| test mAP | 41.2 | 44.1 | 49.0 | 51.1 | **55.7** |
| test NDS | 47.9 | 50.4 | 58.2 | 59.5 | **63.8** |

## 🧭 서사 (포스터 논지 — 초안 §5)
1. **DETR3D** — 쿼리 기반 set prediction을 3D로 들고 온 첫 걸음. 투영은 명시적(변환행렬), 디코더는 NMS-free("We do not use any post-processing such as NMS", §4.1). 깊이 애매성을 본인 인정(§4.5).
2. **PETR** — 재투영을 버리는 전략: 3D 좌표를 한 번 PE로 바꿔 피처에 합산하고 전역 어텐션 원샷. 비용이 디코더에서 임베딩 전처리로 이동.
3. **PETRv2** — PE가 프레임 간 시간 정렬까지 흡수(2 프레임).
4. **Sparse4D** — PE 노선의 반대편: 앵커 기준 4D 키포인트 희소 샘플링 + instance-level depth reweight(투영 ill-posed 완화). 시간은 스택 — T 늘리면 FPS·메모리가 동선형으로 비싸짐(v2 본문 Table 1: 21.5→6.1 FPS, 424→1149 MB, RTX 3090·704×256·R50).
5. **Sparse4D v2** — 스택 압박을 순환으로 해소: 인스턴스 상태만 프레임 간 전달, O(T)→O(1), test NDS 63.8.

## 🔬 검증과 배제 (초안 §6 — 미검증 명시 원칙)

| 기억했던 수치/표현 | 실제 |
|---|---|
| DETR3D test "46.2 mAP / 53.5 NDS" | **배제** — 논문·README·후속 비교표(0.412/0.479) 어디에도 없음 |
| DETR3D "GPU-center" 변형 | **배제** — 논문 용어가 아님("8 RTX 3090 GPUs"만 존재) |
| Sparse4D "15.9 FPS" | **배제** — 실제 인접값: v1 15.3(T=2) · v2 20.3 |
| Sparse4D "no NMS" 명시 | **배제** — v1/v2에 표현 없음. NMS-free 명시는 DETR3D가 유일 |
| Sparse4D "depth sort" | **배제** — 실제: depth reweight(v1) · dense depth supervision(v2) |
| DETR3D "균일 깊이 가정" | **귀속 오류** — LID 깊이축 64점 샘플링은 PETR §4.2 |
| arXiv ID | 오기 4/5건 → abs 페이지에서 검증 후 교정 |

## 🔜 다음 연결
- 차트/그림 일체 미제작(초안 §7 Scope: "표·문만으로 이미 3D 공간을 압축 — 그림은 다음 세션").
- StreamPETR(2303.11926) 이후 3D/BEV 후속은 별도 트랙 큐로 소진(`3D_OBJECT_DETECTION_지도.md` 순서 제안 참고).
- 11/04(수) 자율 세션(LLM 사후강화학습) 포스터는 `poster/` 키트 재사용.
