# 05 · 보충: RT-DETRv2 Explained in 8 Illustrations (arXiv:2509.01241)

> **2026-10-20 보충** · Chua & Tan (Singapore General Hospital), 2025-09-01 · 5쪽 · 그림 8개 · CC-BY
> 원문: [RT-DETRv2 Explained in 8 Illustrations](https://arxiv.org/abs/2509.01241) · PDF: `day07_detr/src/rtdetrv2.pdf`(로컬 캐시·커밋 제외) → 재취득: `curl -L https://arxiv.org/pdf/2509.01241 -o day07_detr/src/rtdetrv2.pdf`
> **이 자료의 성격:** 새 방법이 아니라 **그림 해설형 정리(explainer)**. 오늘 다루는 RT-DETR(v1)과 이어서, **실전에서 돌아가는 RT-DETRv2의 텐서 흐름을 머릿속에 그리기 위한 시각적 참고서**다.

## 🎯 왜 지금 붙이는가

- 오늘 본 RT-DETR(2304.08069)은 **v1**이고, 실전에서 쓰는 `rtdetr-l.pt` 계열은 **v2**(RT-DETRv2, 2024)다. v2는 v1의 하이브리드 인코더 위에 **deformable attention + denoising(DN) 학습**을 얹은 버전이다.
- 이 해설 논문은 그 v2의 **텐서 모양까지 그려서** 설명한다. Day 08(Deformable DETR)의 사전지식과 Day 10(RT-DETR 본문·DINO-DETR)의 시각적 문맥을 한 번에 잡아준다.
- **주의:** 이 논문은 *v2*를 다룬다. v1 논문의 "uncertainty-minimal 쿼리 선택"(Q5-b)과 v2 해설 그림의 "분류 브랜치 TopK" 표현은 **문헌이 다르다** — 아래 §4에서 구분한다.

## 📐 기본 설정(논문 명시)

| 항목 | 값 |
|---|---|
| 백본 | ResNet-50 |
| 입력 | 640×640 |
| 쿼리 수 | 300 (object queries) |
| denoising 쿼리 | 100 (실제 d는 배치마다 다름) |
| 디코더 블록 | 6개 |
| 채널 | 256 (embed dim) |
| attention head | 8 |
| batch | B로 표기 |

**숫자 하나 메모:** 인코더 출력은 3개 스케일(80×80, 40×40, 20×20)을 이어 붙인 `(B, 8400, 256)` 텐서다. `80²+40²+20² = 6400+1600+400 = 8400`. 이 8400이 deformable attention의 "샘플링 대상 위치 수"가 되고, 쿼리 300개를 고르는 대상 풀(pool)이다.

## 🖼️ 그림 8장 해설

### Figure 1 — 전체 파이프라인 (denoising off, 즉 추론 시)
```
input (B,3,640,640)
  └ ResNet embedding + Stage0~3
      (B,64,160,160) → (B,256,160,160) → (B,512,80,80) → (B,1024,40,40) → (B,2048,20,20)
  └ Conv2D+BN로 256ch 투영
      (B,256,80,80) · (B,256,40,40) · (B,256,20,20)
  └ Hybrid Encoder
      (B,256,20,20) · (B,256,40,40) · (B,256,80,80)
  └ Concat + Reshape → (B,8400,256)   ← 6400+1600+400
      ├ Valid Mask (1,8400,1) 원소곱 → 무효 공간 위치 제거
      ├ Linear → (B,8400,256)
      ├ 분류 브랜치: Linear → (B,8400,80) → TopK → (B,300,80)
      └ 위치 브랜치: Linear → (B,8400,4) → Gather → (B,300,4)  (anchors)
  └ 인코더 임베딩 Gather → (B,300,256) (query embeddings, detach)
  └ Decoder → 클래스/박스 예측
```
핵심 흐름: **8400개 후보 위치에서 300개만 골라 디코더에 넣는다.** 즉 쿼리 수는 고정(300)이고, "무엇을 물어볼지"만 인코더 출력이 결정한다.

### Figure 2 — 하이브리드 인코더 (v1의 AIFI+CCFF를 그림으로)
- **Encoder 부분**: self-attention은 **해상도가 가장 낮고 의미가 가장 richest한 20×20 맵 단 하나에만** 수행된다. 그림의 텐서: `(B,C,H,W)` → Conv2D+BN+SiLU → Reshape `(B,400,256)`(=20×20) → 위치 임베딩 `(1,400,256)` 더하기 → MHA → Dropout/LayerNorm → FFN(1024, GELU) → 다시 `(B,256,20,20)`.
- **융합 경로 2갈래**:
  - **Top-down(FPN 방향)**: `(B,256,20,20)` → Interpolate → 40×40 → ResNet Stage2의 `(B,256,40,40)`과 Concat → 512ch → CSP Net → `(B,256,40,40)` → 같은 절차로 80×80.
  - **Bottom-up(PAN 방향)**: 80×80 → Downsample → 40×40 → 앞선 40×40 텐서와 Concat → 512 → CSP → 256 → 반복해서 20×20까지.
- 즉 **의미는 위에서 내려오고, 공간 해상도는 아래에서 올라온다.** 세 스케일 모두 256ch로 정렬된 채 디코더로 넘어간다.

### Figure 3 — Encoder 내부 & CSP 블록
- **Encoder**: Conv2D+BN+SiLU 블록(3개) → Reshape → 위치 임베딩 더하기 → Multi-Head Attention → Dropout/LayerNorm → Linear(1024) → GELU → Dropout → Linear(256) → Dropout/LayerNorm.
- **CSP 블록**: Conv2D+BN+SiLU 반복 + **RepVGG 블록** 삽입. RepVGG는 train-time에는 여러 브랜치, inference 때는 단일 3×3 conv로 합쳐져 **속도 + 수렴 안정성**을 함께 챙긴다(RT-DETRv2의 bag-of-freebies 축).

### Figure 4 — denoising task가 켜진 전체 구조 (학습 시)
- GT 박스/라벨을 **의도적으로 조금 깨뜨려** denoising queries를 만든다 → `(B,d,4)` 박스와 `(B,d,80)` 클래스.
- 이 쿼리는 GT와 **직접 대응**하므로 헝가리아 매칭이 필요 없다. 본 쿼리(300)와 denoising 쿼리(d)를 concat해 `(300+d)`개로 디코더에 넣고, attention mask `(300+d, 300+d)`로 **두 그룹이 서로를 보지 못하게** 가린다.
- **손실은 두 그룹 따로 계산**한다(DN-DETR 계열, Day 09에서 다시 만난다).

### Figure 5 — Decoder 전체
- 6개 decoder block이 쌓이고, **각 블록마다 분류/박스 loss 스택**이 있다.
- 블록 사이 `detach` 표시: 박스 예측은 블록을 관통하는 역전파가 아니라 **각 블록 출력이 독립적으로 손실을 계산**한다(= 원문 DETR의 auxiliary decoding loss, RT-DETR v2에서의 재활용).

### Figure 6 — Decoder Block (= query positioning head + bbox head)
- **Query positioning head**: query 박스 `(B,300+d,4)` → Reshape `(B,300+d,1,4)` → Inverse Sigmoid → Linear+ReLU(512) → Linear(256) → 위치 임베딩으로 쓰인다.
- **Bbox head**: `(B,300+d,256)` → Linear+ReLU → Linear+ReLU → Linear → `(B,300+d,4)`.
- **inverse sigmoid의 의미**: 좌표를 `[0,1]` 구간에서 **무한대 영역으로 빼내** 디코더가 자유롭게 조정하게 하고, 마지막에 다시 sigmoid로 `[0,1]`에 눌러 넣는다. 그래서 bbox head는 좌표 자체가 아니라 **기준 박스에 대한 오프셋**을 출력한다.

### Figure 7 — Decoder Layer 한 층
```
query positioning head (B,300+d,256) ─┐
query embeddings        (B,300+d,256) ─┤
attention mask (300+d,300+d)          │
   └ Multi-Head Attention → Dropout → LayerNorm → (B,300+d,256)
   └ Multi-Scale Deformable Attention
        (encoder embeds (B,8400,256), reference points (B,300+d,1,4)) → (B,300+d,256)
   └ 두 출력 element-wise add (residual) → Dropout/LayerNorm
   └ FFN: Linear+ReLU(1024) → Linear → Dropout/LayerNorm
```
핵심: **쿼리끼리의 self-attention**으로 서로 다른 쿼리가 자기 역할을 조정하고, **deformable attention**으로 이미지 쪽에서 96개 지점만 콕 찌른다.

### Figure 8 — Multi-Scale Deformable Attention
- **샘플링 지점 수**: head 8개 × 스케일 3개 × head당 4지점 = **96**. 그래서 attention weight를 만드는 Linear는 `(B,300+d,256) → (B,300+d,96)`.
- **오프셋**: 별도 Linear가 `(B,300+d,192)`(=96×2, x/y) 를 만들고, **기준점(쿼리 박스 중심)에 더해** 실제 샘플링 위치를 만든다. 기준점 생성에 쿼리 박스의 **너비·높이**가 쓰인다(`[:, :, None, :, 2:]`).
- **정규화**: 샘플링 좌표는 `[0,1]` → `[-1,1]`로 재정규화(grid_sample 요구사항).
- **값 추출**: 인코더 `(B,8400,256)`를 스케일별로 잘라 `(Bx8,32,80×80)`, `(Bx8,32,40×40)`, `(Bx8,32,20×20)` → 각 스케일에서 grid_sample → 샘플 값들을 attention weight로 가중합 → `(B,300+d,256)`.
- **원리**: 전역 self-attention은 모든 키와 유사도를 계산해야 하지만, deformable attention은 **주변 몇 점만** 본다. Day 08(Deformable DETR)이 이 원문의 출발점이다.

## 🔍 v1 논문 vs v2 해설 — 표현 차이 감별

| 항목 | RT-DETR v1 (2304.08069, 오늘 본 논문) | RT-DETRv2 해설 (2509.01241) |
|---|---|---|
| 인코더 | AIFI(S5만 self-attn) + CCFF(conv 융합) | **동일 구조**를 그림으로 표현(Fig 2·3) |
| 쿼리 선택 | **uncertainty-minimal** — 분류·위치 불확실성을 명시 최적화(식 2·3), Table 4(0.30% → 0.67%) | 그림은 **분류 브랜치 TopK**로 그림(Fig 1) |
| 어텐션 | 표준 attention 기반 | **deformable attention** 도입(DN + bag-of-freebies) |
| denoising | 없음 | **있음**(Fig 4, d≈100) |
| 디코더 | 6층, aux loss | 6블록 + 블록별 loss 스택(Fig 5) |

**해석 지침:** v1의 "uncertainty"는 쿼리 **선택 기준과 그 학습 신호**에 대한 이야기이고, v2 해설 그림은 선택 파이프라인(분류 logits → TopK → Gather 위치박스)을 데이터 흐름 중심으로 그린다. **둘은 모순이 아니라 관점의 차이**이며, Q5-b 답변(오늘 과제)에서는 v1 논문 식(2)(3)과 Table 4를 근거로 쓰고, v2 그림은 "실제 파이프라인이 이렇게 생겼구나"의 시각 참고로만 쓴다.

## 🔗 다음 Day로 이어지는 다리

- **Day 08 (Deformable DETR)**: Figure 8의 deformable attention이 출발점. "모든 키를 보지 않고 몇 점만 본다"는 발상이 어디서 왔는지.
- **Day 09 (DAB / DN-DETR)**: Figure 4의 denoising task(DN-DETR)와, "쿼리는 처음에 무엇이야?"라는 DAB의 질문.
- **Day 10 (DINO-DETR / RT-DETR 본문)**: RT-DETRv2가 아니라 v1 본문 + DINO. v2는 오늘의 05 자료로 미리 한 번 훑은 상태.

## 📌 오늘 읽기 순서에 넣는 방법

1. 원문 DETR §3.1–3.2 (읽기 20분) → 세트 예측이 왜 가능한지
2. RT-DETR §4.2·§4.3·Table 3·4·5 (Q5 근거) → 실시간으로 바꾼 두 손잡이
3. **이 자료 Figure 1·2·4** (10분) → 오늘 다룬 구조가 실제 텐서로 어떻게 생겼는지 그림으로 고정
4. Figure 8은 Day 08 예습으로 한 번 더
