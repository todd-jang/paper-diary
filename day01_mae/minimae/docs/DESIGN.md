# DESIGN — mini-MAE 설계 문서

## 1. 아키텍처 (데이터 흐름)
```
이미지 (B,3,224,224)
   │  patchify: Conv2d(k=16,s=16)
   ▼
패치 임베딩 (B,196,D) + 고정 sin-cos 2D pos emb
   │  random_masking(mask_ratio=0.75)
   ▼
보이는 25% 토큰만 (B,49,D)         ← 인코더 입력 (마스크 토큰 없음)
   │  Encoder: 6 × [LN→MHA→res, LN→MLP→res] → LN
   ▼
latent (B,49,D)
   │  dec_embed(Linear D→256)
   │  + 공유 mask_token(147개) + ids_restore(원래 순서)
   │  + 전체 196 위치 임베딩
   ▼
디코더 토큰 (B,196,256)
   │  Decoder: 2 × Block → LN → Linear(256→p·p·C)
   ▼
픽셀 복원 (B,3,224,224)
   │  per-patch 정규화 MSE (마스킹된 패치만)
   ▼
loss (scalar)
```
- 인코더가 무겁고(6블록, 192-d) 디코더가 가볍다(2블록, 128-d→Linear) = **비대칭**.

## 2. 논문 대응표
| MAE 설계 (논문) | 이 구현 | 위치 |
|---|---|---|
| 비대칭 인코더-디코더 | ✅ enc 6블록/192d, dec 2블록/128d | `model.MaskedAutoencoderViT` |
| 인코더에 마스크 토큰 없음(보이는 패치만) | ✅ `forward_encoder`가 `x_vis`만 통과 | `model.forward_encoder` |
| 높은 마스킹 비율(0.75) | ✅ `mask_ratio` | `model.random_masking` |
| **per-patch 정규화 MSE** | ✅ `norm_pix_loss` (mean/var 정규화 후 MSE) | `model.forward_loss` |
| **고정 2D sin-cos 위치 임베딩**(학습 X) | ✅ `register_buffer` | `get_2d_sincos_pos_embed`, `_init_weights` |
| class token 없음 | ✅ | — |
| 공유 mask token + 전체 위치 임베딩 | ✅ `mask_token`, `dec_pos_embed` | `forward_decoder` |
| AdamW(β=0.9,0.95), wd=0.05 | ✅ | `train` |
| warmup + cosine, layer-wise lr decay(0.75), grad clip 1.0 | ✅ | `param_groups_lrd`, `cosine_schedule` |
| RandomResizedCrop + HFlip, **정규화 없음** | ✅ | `data.build_transform` |
| 선형 프로빙: 인코더 freeze + 특징 mean/std 정규화 | ✅ | `eval_linear` |

## 3. 손실 (per-patch 정규화 MSE)
각 패치(16×16×3=768차원)를 **개별적으로 zero-mean/unit-variance 정규화**한 뒤 MSE.
```
target = patchify(img)                      # (B, N, p·p·C)
mean   = target.mean(-1, keepdim=True)
var    = target.var(-1, keepdim=True)
target = (target - mean) / sqrt(var + 1e-6) # ← 정규화 (예측값은 정규화 X)

loss_patch = mean( (pred - target)^2 , dim=-1 )   # (B, N)
loss       = sum(loss_patch * mask) / sum(mask)    # 마스킹된 패치만 평균
```
- **왜 정규화?** 패치별 밝기/대비가 달라도 모델이 국소 구조에 집중하게 함(고주파 둔감화). 논문 핵심 디테일.
- `norm_pix_loss=False`로 끄면 원시 픽셀 MSE.

## 4. 마스킹 메커니즘
```
noise = rand(B, N)
ids_shuffle = argsort(noise)        # 무작위 순열
ids_restore = argsort(ids_shuffle)  # 역순열
ids_keep    = ids_shuffle[:, :49]   # 앞 25%만 남김
x_vis       = gather(x, ids_keep)
mask        = ones(B,N); mask[:, :49]=0; mask = gather(mask, ids_restore)  # 1=가림
```
- 디코더에서 `gather(x, ids_restore)`로 원래 패치 순서를 복원.

## 5. 학습 레시피
- Optimizer: AdamW(lr=1.5e-4, betas=(0.9,0.95), wd=0.05).
- LR: `warmup_epochs` 선형 워밍업 후 cosine 감쇠(min_lr).
- **layer-wise lr decay 0.75**: 인코더/디코더 블록이 깊을수록 lr 작게(깊이 `d`에서 `0.75^(L-d)`). bias·norm·pos_embed·mask_token은 weight decay 제외.
- grad-norm clip 1.0.

## 6. 선형 프로빙 프로토콜
1. 인코더 freeze, `mask_ratio=0`(전체 196 패치)로 특징 추출.
2. `enc` 출력 토큰을 **평균 풀링** → (B, D). (class token 없으므로 평균 사용)
3. 특징을 train 통계로 **mean/std 정규화**(MAE 논문 프로토콜).
4. 그 위에 `Linear(D → nclass)`만 학습(CE), val top-1 측정.

## 7. 미니 스케일 vs 논문 스케일
| 축 | mini(이 저장소) | 논문 ViT-B | 논문 ViT-H |
|---|---|---|---|
| 데이터 | 14 클래스 / 16k | ImageNet-1K 1.28M | 1.28M |
| 모델 | ViT-Tiny 3.34M | ViT-B 86M | ViT-H 632M |
| epoch | 20 | 800 | 800 |
| 해상도 | 224 | 224 | 224 |
| 결과 | loss↓, probe 28%+ | IN-1K FT 83.6% | IN-1K FT 87.8% |
> 목표는 절대 성능이 아니라 **논문 설계를 충실히 구현·검증**하는 것.

## 8. 참고
- 원논문: *Masked Autoencoders Are Scalable Vision Learners*, He et al., CVPR 2022, arXiv:2111.06377.
- 공식 구현: github.com/facebookresearch/mae.
