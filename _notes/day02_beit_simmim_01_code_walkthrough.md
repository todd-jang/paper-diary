---
layout: default
title: "Day 02 BEiT & SimMIM — 01. 코드 워크스루 — BEiT(dVAE+MIM) / SimMIM 라인바이런 해설"
paper_id: beit-simmim
date: 2026-10-16
---

# 01. 코드 워크스루 — BEiT(dVAE+MIM) / SimMIM 라인바이런 해설

> 대상: BEiT 공식 구현 [`unilm/beit`](https://github.com/microsoft/unilm/tree/master/beit) · SimMIM 공식 구현 [`microsoft/SimMIM`](https://github.com/microsoft/SimMIM)
> 두 코드를 **데이터 흐름** 기준으로 위→아래로 따라가며 "왜 이렇게 짰나"를 본다.

---

## PART A. BEiT — 2단계 학습

### A.0 큰 그림

```
[Stage 1: 이미지 토크나이저 (dVAE)]
이미지 x (B,3,224,224)
  │ dVAE Encoder (DALL-E, 다운샘플 8×)
  ▼
연속 latent → 코드북(8192) 양자화 → z (B,14,14)  ← "시각 단어 ID"
  │
  ▼  저장: 토큰 ID 그리드 z  (MIM의 정답 라벨이 된다)

[Stage 2: Masked Image Modeling]
패치 임베딩 Z (B,196,768)  (+ [CLS], 상대 위치편향)
  │ blockwise masking (~40%) → 마스크 자리에 [MASK] 토큰
  ▼
Transformer Encoder (ViT-B, 12층)
  ▼
MIM Head (Linear → |V|=8192 softmax)   ← 가린 자리의 토큰 분류
  ▼
CE loss = CrossEntropy(pred_tokens[masked], z[masked])
```

**핵심:** Stage 2는 **분류 문제**다. 타깃은 픽셀이 아니라 **dVAE가 만든 이산 토큰 ID**(Stage 1 산출물).

---

### A.1 Stage 1 — dVAE 토크나이저

```python
# openai/DALL-E 의 dVAE 를 그대로 사용 (BEiT repo: beit/dall_e/)
z = image_tokenizer.get_codebook_indices(img)   # (B, H/p, W/p) int64, 값 ∈ [0, 8192)
# 224 / 16 = 14 → z: (B, 14, 14)  ← 196개 토큰
```

- **코드북 크기 |V| = 8192**, 다운샘플 팩터 8 (DALL-E 표준).
- 이미지를 16×16 패치 격자(14×14)에 맞춰 **시각 단어 ID**로 변환.
- 이것이 BEiT의 "라벨"이 된다 — **사전학습된 토크나이저가 필요**하다는 게 BEiT의 비용이자 특징.
- dVAE는 EMA로 안정 학습되며, BEiT는 이 단계를 **한 번만** 수행해 토큰을 캐시한다.

> **왜 이산 토큰?** 논문: "pixel-level auto-encoding은 모델을 **단거리 의존·고주파 디테일**에 집중시킨다." 이산 의미 토큰을 예측하게 하면 **추상적·의미적** 표현을 강제한다.

### A.2 Stage 2 — 마스킹 (blockwise, ~40%)

```python
# BEiT: 무작위 사각형 블록 단위로 마스킹 (Algorithm 1)
mask = random_blockwise_mask(grid=(14,14), mask_ratio=0.40,
                             num_blocks_min=1, num_blocks_max=5, aspect=0.3..3.3)
# → 마스크된 위치 집합 M  (blockwise = 이웃에 힌트가 남지 않게)
```

- **blockwise(rectangular block)**: n-gram 마스킹처럼 인접 패치를 **블록째** 가린다.
- 목적: 가린 패치의 이웃에서 "공짜 힌트"를 못 얻어 **과제를 진짜 어렵게** 만든다.
- 기본 마스킹 비율 ≈ **40%** (MAE의 75%보다 낮음 — 타깃이 이산이라 과제 난이도가 다름).

### A.3 Stage 2 — 모델 forward

```python
x = patch_embed(x) + rel_pos_bias          # (B,196,768)  BEiT는 상대위치편향(shared)
x = torch.cat([cls_token, x], dim=1)       # [CLS] 추가 → (B,197,768)
x = mask_tokens(x, mask)                    # 마스크 자리를 학습된 [MASK] 임베딩으로 교체
x = encoder(x)                              # ViT-B 12층 (마스크 토큰도 함께 통과)
logits = mim_head(x[:, 1:])                 # (B,196,8192)  가린 자리만 쓴다
loss = F.cross_entropy(logits[mask], z[mask])   # 토큰 분류(CE)
```

- **인코더가 마스크 토큰을 받는다** (BERT식). MAE와의 결정적 차이.
- **MIM head = 선형 + softmax(8192)** → 각 가린 패치가 어느 시각 단어인지 분류.
- 손실은 **가린 위치에서만** CE.
- BEiT는 **상대 위치 편향**(T5식)을 사용 — 사전학습 땐 self-attention 층 간 **공유**, 미세조정 시 각 층이 공유 편향에서 초기화.

### A.4 미세조정

```python
# 사전학습 후: MIM head 는 버리고(분류 head 교체), encoder 위에 task layer 부착
model = BEiT(pretrained=ckpt); model.head = nn.Linear(768, num_classes)
# BEiT-B: IN-1K 83.2% (from-scratch DeiT 81.8% 상회), BEiT-L: 86.3%
```

---

## PART B. SimMIM — 단일 인코더 + 경량 head

### B.0 큰 그림

```
이미지 x (B,3,192,192)
  │ PatchEmbed
  ▼
패치 토큰 Z (B, N, D)
  │ random mask (마스크 패치 32×32, 비율 0.6)
  │ → 마스크 자리를 "학습된 마스크 토큰"으로 교체 (★ 삭제 아님)
  ▼
Encoder (Swin / ViT)   ← 보이는 패치 + 마스크 토큰을 "함께" 처리
  ▼
Prediction Head (Linear + LayerNorm → 업샘플 192²)
  ▼
픽셀 예측  ── ℓ1 loss (가린 픽셀에서만)
```

**MAE와의 결정적 차이:** SimMIM은 **마스크 토큰을 인코더에 넣는다** (MAE는 안 넣음). 디코더도 없다 — **단일 인코더 + 선형 head**.

### B.1 마스킹

```python
# random masking, 큰 마스크 패치(32×32), 비율 0.6
mask = random_patch_mask(num_patches, mask_ratio=0.6, mask_patch_size=32)
# 192² 이미지: 마스크 패치 32 → 6×6=36 블록 중 60% 가림
```

- **larger masked patch size(32)**: 가린 픽셀 주변에 보이는 이웃이 남을 확률이 줄어 **과제가 어려워진다**.
- 큰 패치(32)면 **10%~70%** 넓은 비율에서 경쟁력, 작은 패치(8)면 비율을 **80%**까지 올려야 함.

### B.2 모델 forward

```python
x = patch_embed(x)                                 # (B,N,D)
x = torch.where(mask_token_expand(mask),           # 마스크 자리 교체
                self.mask_token.expand_as(x), x)    # ★ 삭제가 아니라 대체 (인코더가 본다)
feat = encoder(x)                                   # Swin/ViT (가린 것도 통과)
pred = head(feat)                                   # head = LayerNorm + Linear (+업샘플)
loss = F.l1_loss(pred[mask], target[mask])          # ℓ1, 가린 픽셀만
```

- **head**: 논문은 **1-층 선형층**으로도 무거운 head와 성능이 같음을 보였다(LayerNorm + Linear, 192² 업샘플).
- **target**: 원시 RGB 픽셀. 다운샘플(192²)한 값에 회귀. `ℓ1` 기본(ℓ2·smooth-ℓ1과 유사).
- 마스크 자리에 **학습된 공유 mask token** 하나를 쓴다.

### B.3 head 최소화가 "기여"인 이유

```python
# 저자들의 주장: "prediction head can be as light as a linear layer"
self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch_size**2*3))
```

- MAE처럼 **비대칭 디코더**를 두지 않고, 인코더 출력에서 **바로** 픽셀을 예측.
- 즉 표현 학습의 부담이 **인코더 자체**에 실린다 → 다운스트림 전이에 유리.

---

## 한눈에 보는 3자 대비 (구조 관점)

| | **MAE** | **BEiT** | **SimMIM** |
|---|---|---|---|
| 인코더 입력 | 보이는 패치만 | 보이는 + **[MASK]** | 보이는 + **마스크 토큰** |
| 타깃 | 원시 픽셀 | **dVAE 이산 토큰** | 원시 픽셀 |
| 예측 방식 | MSE 회귀 | **CE 분류(8192)** | **ℓ1 회귀** |
| 마스킹 | random 75% | **blockwise ~40%** | **random, 패치32, 60%** |
| 별도 모듈 | 비대칭 **디코더**(폐기) | dVAE **토크나이저**(사전학습) | **경량 선형 head** |
| 백본 | ViT | ViT | **Swin/ViT/Conv** |

**핵심 문장:** BEiT는 시각을 **"단어 분류"** 로, SimMIM은 **"픽셀 회귀"** 로, MAE는 **"보이는 것만 보고 픽셀 복원"** 으로 MIM을 푼다.
