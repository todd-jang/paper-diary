# HANDOFF — mini-MAE (FAIR MAE 충실 재현)

> 이 문서는 처음 보는 엔지니어가 **추가 질문 없이** 이 저장소를 실행·이해·확장할 수 있도록 쓴 이관 가이드다.

## 0. 한 줄 소개 + 현재 상태
`mrm8488/ImageNet1K-train`(실제 ImageNet-1K, 비게이트) 서브셋으로 돌아가는 **축소판 Masked Autoencoder**(He et al., CVPR 2022, arXiv:2111.06377) 재현.

| 항목 | 값 |
|---|---|
| 모델 | ViT-Tiny MAE, **3.34M** 파라미터, 196 패치, mask 0.75 → 49개만 인코딩 |
| 데이터 | ImageNet-1K **14 클래스 / train 15,956 / val 840** |
| 검증(2 epoch) | 사전학습 loss **0.9860 → 0.8293**, 선형프로빙 **28.1%** (14-way, 랜덤≈7%) |
| 풀 학습 | **20 epoch 진행 중** (MPS, ~3.8분/epoch, 총 ~75분) |
| 테스트 | `pytest -q tests` → **29 passed** |

## 1. 실행 환경
- 인터프리터: **`/opt/homebrew/bin/python3.14`** (torch 2.14, torchvision 0.29, numpy 2.5, pyarrow 26, pyyaml 6, pytest 9.1).
  - ⚠️ 맨 `python3`는 `/usr/bin/python3`(3.9, torchvision 없음)로 잡힐 수 있음 → 반드시 풀패스 사용.
- 디바이스: Apple **MPS** 자동 선택(`device: auto`). CPU/CUDA 폴백 지원.
- 설치: `pip install -r requirements.txt`

## 2. 데이터 출처(provenance)
- 원본 ImageNet-1K는 **게이트/수동 다운로드**(HF `datasets`는 토큰+동의 필요) → 자동화 불가.
- 대안: HuggingFace **`mrm8488/ImageNet1K-train`** (비게이트, 실제 ImageNet-1K 이미지, 296 shard).
- shard URL 패턴:
  `https://huggingface.co/datasets/mrm8488/ImageNet1K-train/resolve/main/data/train-{NNNNN}-of-00296.parquet`
- shard schema: `image` = struct<bytes, path>, `label` = int64. 라벨은 표준 ImageNet 순서(0=tench).
- 이 저장소는 **4개 shard(00000~00003, ~2GB)** → 14 클래스. 클래스명은 `data/subset/class_map.json`.

## 3. 실행법
### 원샷
```bash
PY=/opt/homebrew/bin/python3.14 ./run.sh          # subset(없으면) → pretrain → linear probe
# 또는
make subset && make train && make eval && make test
```
### 개별
```bash
# 1) 서브셋 구성 (parquet → ImageFolder)
$PY src/prepare_subset.py --shards "data/train-0000*.parquet" --out data/subset \
    --per-class 1200 --val-per-class 60

# 2) MAE 사전학습
$PY src/train.py --config configs/tiny.yaml
$PY src/train.py --config configs/tiny.yaml --epochs 3 --batch-size 32   # CLI 오버라이드

# 3) 선형 프로빙 (표현 품질 측정)
$PY src/eval_linear.py --ckpt runs/mini_mae_tiny.pt --config configs/eval.yaml

# (선택) loss 곡선
$PY src/plot_loss.py --metrics runs/metrics_tiny.csv --out runs/loss_curve.png

# 테스트
$PY -m pytest -q tests
```

## 4. 설정 레퍼런스
우선순위: **CLI 인자 > `--config` 파일 > 코드 내 DEFAULTS**. 불리언은 `--norm-pix-loss / --no-norm-pix-loss`.

| 키 | 기본 | 의미 |
|---|---|---|
| `arch` | `vit_tiny` | `vit_tiny` / `vit_small` |
| `img_size` / `patch_size` | 224 / 16 | 입력 해상도 / 패치 크기(→ 196 패치) |
| `mask_ratio` | 0.75 | 가리는 패치 비율(논문값) |
| `epochs` / `batch_size` | 20 / 64 | 학습 길이 / 배치 |
| `lr` / `min_lr` | 1.5e-4 / 1e-6 | AdamW base / cosine 최저 |
| `weight_decay` | 0.05 | bias·norm 제외 |
| `warmup_epochs` | 5 | 선형 워밍업 |
| `layer_decay` | 0.75 | layer-wise lr decay 계수 |
| `clip_grad` | 1.0 | grad-norm 클리핑 |
| `norm_pix_loss` | true | **per-patch 정규화 MSE**(논문 핵심) |
| `num_workers` | 4 | DataLoader 워커 |
| `seed` | 0 | 재현성 |
| `data` | data/subset/train | 학습 폴더 |
| `out` / `metrics` / `logfile` | runs/… | 체크포인트 / CSV / 로그 |
| `device` | auto | mps/cuda/cpu |

`configs/eval.yaml`은 선형 프로빙용(`epochs, batch_size, lr, wd, train, val, out`).

## 5. 산출물 위치
| 파일 | 내용 |
|---|---|
| `runs/mini_mae_tiny.pt` | 매 epoch 저장 체크포인트(`model/args/epoch/optimizer`) |
| `runs/metrics_tiny.csv` | `epoch,loss,lr,sec` |
| `runs/train_tiny.log` | 학습 로그 |
| `runs/linear_results.json` | 선형 프로빙 결과(`best_val_acc` 등) |

## 6. 파일 맵 (src/)
| 파일 | 역할 |
|---|---|
| `model.py` | MAE 본체: `PatchEmbed`/`Block`/`MaskedAutoencoderViT` + `mae_vit_tiny/small` |
| `data.py` | MAE augmentation(RandomResizedCrop+HFlip, 무정규화) + 로더 |
| `train.py` | 사전학습 루프(옵티마이저/스케줄/LRD/clip) |
| `eval_linear.py` | 인코더 freeze + 선형 프로빙 |
| `prepare_subset.py` | parquet → ImageFolder + `class_map.json` |
| `utils.py` | 시드/디바이스/로거/체크포인트/메트릭/config 머지 |
| `plot_loss.py` | loss 곡선 이미지(선택) |

## 7. 확장 방법
- **데이터 ↑**: shard를 더 받아 `--shards`에 병합, `--per-class`·`--max-classes` 상향 (296 shard 전체 ≈ 150GB, 1000 클래스).
- **모델 ↑**: `--arch vit_small` (또는 model.py에 ViT-B 추가).
- **성능 ↑**: `--epochs 100+` (논문은 800).
- 권장 비율: 데이터/모델 키울수록 epoch·lr 튜닝 필요.

## 8. 알려진 한계 / 주의
- 클래스 폴더명은 **정수 라벨 인덱스**(표준 ImageNet 순서 가정). synset(LOC) 매핑은 미포함, 이름은 `class_map.json` 참고.
- 원본 ImageNet-1K 게이트 데이터셋과 **정확히 동일 분할은 아님**(미러 서브셋).
- MPS는 CUDA보다 느림(~3.8분/epoch). 대량 학습은 CUDA 권장.
- 라이선스: ImageNet 이미지 사용은 원 저작권자의 이용약관을 따름(연구 목적).

## 9. 트러블슈팅
| 증상 | 원인/해결 |
|---|---|
| `ModuleNotFoundError: torchvision` | 맨 `python3` 사용 → `/opt/homebrew/bin/python3.14` 사용 |
| MPS out of memory | `--batch-size` 축소, `--num-workers` 조정 |
| DataLoader 멈춤 | macOS에서 `--num-workers 0` 시도 |
| 재현 안 됨 | `seed` 고정 확인(train/eval 동일 시드) |
