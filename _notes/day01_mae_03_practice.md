---
layout: default
title: "Day 01 MAE — 03. 실습 — 작은 데이터셋으로 MAE 사전학습 돌려보기"
paper_id: mae
date: 2026-10-15
---

# 03. 실습 — 작은 데이터셋으로 MAE 사전학습 돌려보기

> 스크립트: [`mae_pretrain_demo.py`](https://github.com/todd-jang/paper-diary/blob/main/day01_mae/mae_pretrain_demo.py)
> 환경: Python 3.14 / torch 2.14 / MPS (Apple Silicon) — 별도 설치 불필요

---

## 0. 즉시 실행 (의존성 0)

```bash
cd ~/paper-diary/day01_mae
python mae_pretrain_demo.py --data synthetic --epochs 3
```

`synthetic`은 `torchvision.datasets.FakeData`(무작위 이미지 256장)로, **다운로드 없이** 파이프라인 동작을 확인한다.

### 검증된 실제 출력 (스모크 테스트)
```
[device] mps
[model] total params = 2.94M | patches = 64 | keep = 16 | mask_ratio = 0.75
[epoch 1/2] mask-MSE loss = 0.2872
[epoch 2/2] mask-MSE loss = 0.1028      ← 감소 = 학습됨
```

---

## 1. 진짜 이미지 (CIFAR-10)

```bash
python mae_pretrain_demo.py --data cifar --epochs 20
```

- 최초 실행 시 `./data/`에 CIFAR-10을 자동 다운로드.
- 32×32 이미지, 패치 4 → `N=64`, 75% 마스킹 → 16개만 보임.

## 1.1 ImageNet-1k

ImageNet-1k는 별도로 내려받아 압축을 풀어야 한다. `--data-dir`에는 `n01440764/` 같은 클래스 폴더가 들어 있는 `train/` 디렉터리를 지정한다.

```bash
python mae_pretrain_demo.py --data imagenet1k \
    --data-dir /path/to/imagenet/train \
    --img-size 224 --patch-size 16 --epochs 1 --batch-size 16
```

이 설정은 224×224 이미지와 16×16 패치(`N=196`)를 쓴다. 이 코드는 작은 MAE 데모이며 논문 수준의 ImageNet 사전학습 재현은 아니다.

## 1.2 영상·오디오 대조학습 (VGGSound)

VGGSound 클립을 별도로 준비한 뒤 WavLM 오디오 인코더와 ImageNet 사전학습 ResNet-18 영상 인코더를 대조학습한다. FFmpeg가 필요하며 첫 학습 실행에서 두 인코더의 가중치를 내려받는다.

```bash
brew install ffmpeg
source .venv/bin/activate
python -m pip install -r day01_mae/requirements-av.txt
```

메타데이터 CSV는 `video_path` 열(비디오 루트 기준 상대 경로 또는 절대 경로)을 포함하거나, VGGSound 형식의 `youtube_id,start_seconds,label,split` 열을 포함해야 한다. 후자의 경우 루트 아래에서 `{youtube_id}_{start_seconds}.mp4` 파일을 찾는다.

```bash
python day01_mae/vggsound_av_contrastive.py \
    --metadata-csv /path/to/vggsound.csv \
    --video-root /path/to/vggsound/videos \
    --split train --limit 256 --epochs 1 --batch-size 4
```

가중치를 내려받기 전에 영상·오디오 디코딩만 확인하려면 같은 명령에 `--inspect-data`를 추가한다. 대조학습은 배치 안의 오디오-영상 쌍을 positive로 두고 나머지 쌍을 negative로 쓰므로 배치 크기는 최소 2다. 현재 환경에서 MPS는 비활성이라 `--device auto`는 CPU를 선택한다.

---

## 2. 관찰 포인트 (노트에 기록)

1. **손실이 줄어드는가?** → 가려진 픽셀 복원을 학습하고 있다는 증거.
2. **`mask_ratio`를 올리면?** 아래 §3 참고.
3. **재구성 이미지를 눈으로 보고 싶다면** §4의 코드 스니펫 추가.

---

## 3. 실험 과제 (Experiments)

### 실험 A — 마스킹 비율 스윕
```bash
for m in 0.5 0.75 0.9; do
  python mae_pretrain_demo.py --data synthetic --epochs 1 --mask-ratio $m
done
```
**검증된 1 epoch 결과 (synthetic):**
| mask_ratio | keep | epoch1 loss |
|---|---|---|
| 0.50 | 32 | 0.4191 |
| 0.75 | 16 | 0.4584 |
| 0.90 | 6 | 0.4497 |

**해석:** 마스킹이 높을수록 초기 loss가 **높게** 시작(과제가 어려움). 단, synthetic은 무작위 이미지라 정보 구조가 없어 0.9가 0.75보다 낮게 나올 수 있음 — **실제 CIFAR-10에서는 0.75 부근이 최적**이 되는지 확인해 볼 것. (논문 Fig.5: 75%가 최적, 95%는 급격히 나빠짐)

### 실험 B — "인코더는 보이는 패치만"이 정말 빠른가?
- `mask_ratio`를 0.75로 고정하고, 인코더에 마스크 토큰을 **억지로** 넣는 변형을 만들어 학습 시간/메모리를 비교.
- 기대: 대칭 버전이 더 느리고 메모리도 큼.

### 실험 C — 비대칭 깊이
- 디코더를 매우 깊게(예: 8블록, `dec_dim=384`) 바꿔 다시 학습 → 선형 프로빙 정확도(아래 §5)가 **오히려 나빠지는지** 확인.
- 논문 결과: 디코더가 얕아도(~1블록) 선형 프로빙 성능이 유지됨.

### 실험 D — 패치 크기
- `--patch-size 8` → `N=16`, 보이는 패치 4개. 더 어려운 과제. 표현 품질 비교.

---

## 4. (선택) 재구성 이미지 시각화

학습 후 아래를 추가하면 원본/마스킹/복원을 나란히 볼 수 있다:

```python
import torch, matplotlib.pyplot as plt
model.eval()
imgs, _ = next(iter(loader))
with torch.no_grad():
    loss, pred, mask = model(imgs.to(device))
# pred: (B,3,H,W)  torchvision.models.mae.unpatchify 형태
def show(img):
    return img.detach().cpu().permute(1,2,0).clip(0,1)
fig, ax = plt.subplots(3, 4, figsize=(8,6))
for i in range(4):
    ax[0,i].imshow(show(imgs[i])); ax[0,i].set_title("original")
    ax[1,i].imshow(show(imgs[i])); ax[1,i].set_title("masked 75%")
    ax[2,i].imshow(show(pred[i])); ax[2,i].set_title("recon")
    for r in range(3): ax[r,i].axis("off")
plt.tight_layout(); plt.savefig("recon.png", dpi=120)
```

> 참고: 공식 `torchvision.models.mae`는 ImageNet용(224, patch16). 여기 데모는 CIFAR용 축소판.

---

## 5. 핵심 평가 지표: 선형 프로빙(Linear Probing)

MAE의 진짜 가치는 복원 품질이 아니라 **표현의 질**이다. 검증법:
1. 사전학습한 **인코더를 freeze**.
2. 인코더 출력에 **선형 분류기만** 붙여 CIFAR-10 분류 학습.
3. 선형 분류 정확도 ↑ → 표현이 선형 분리 가능(좋은 표현).

> 실습 과제: `forward_encoder`를 freeze하고 linear head를 붙여 정확도를 재보면, MAE가 "복원"이 아닌 "표현"을 배웠음을 직접 확인할 수 있다. (디코더는 버림 — Q5의 실증)

---

## 6. 논문과의 대응

| 데모 설정 | 논문 설정 | 비고 |
|---|---|---|
| 32×32, P=4, N=64 | 224×224, P=16, N=196 | 축소판 |
| enc D=192, 6블록, 2.94M | ViT-Base 86M ~ ViT-H 632M | 스케일 |
| mask 0.75 | mask 0.75 | **동일 (핵심)** |
| dec D=96, 2블록 | dec D=512, 8블록 | 가볍게 |
| MSE(패치 정규화 없음) | 정규화 후 MSE | 단순화 |

**핵심 설계(비대칭 + 75%)는 축소판에서도 그대로 유지**한다.
