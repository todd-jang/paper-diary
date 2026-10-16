# mini-MAE — FAIR MAE (He et al., CVPR 2022) 충실 재현

ImageNet-1K 서브셋으로 돌아가는 **축소판 Masked Autoencoder**. 논문의 핵심 설계를
빠뜨리지 않고 재현하고, 실제로 학습·평가까지 검증했다.

> 원논문: *Masked Autoencoders Are Scalable Vision Learners*, arXiv:2111.06377

---

## ✅ 검증된 결과 (이 머신, Apple MPS)

| 항목 | 값 |
|---|---|
| 데이터 | ImageNet-1K 실제 이미지 **14 클래스 / train 15,956 / val 840** |
| 모델 | ViT-Tiny MAE (3.34M 파라미터, 196 패치, mask 0.75 → 49개만 인코딩) |
| 사전학습 loss (20 epoch) | **1.0289 → 0.5387** ✅ 단조 감소·수렴 |
| 선형 프로빙 (20ep ckpt) | val_acc **39.76%** (14-way, 랜덤≈7%) — 2 epoch 때 28.1%에서 상승 |
| 테스트 | `pytest -q tests` → **29 passed** |
| 속도 | ~190–336s/epoch (ViT-Tiny, batch64, 224px, Apple MPS) |

📉 loss 곡선: `runs/loss_curve.png` (생성: `python src/plot_loss.py`)

---

## 🧬 논문 충실도 체크리스트

| MAE 설계 | 이 구현 | 비고 |
|---|---|---|
| 비대칭 인코더-디코더 | ✅ | 인코더는 보이는 패치만(49개), 디코더는 전체(196) 처리 |
| 인코더에 마스크 토큰 없음 | ✅ | `forward_encoder`가 `x_vis`만 통과 |
| 높은 마스킹 비율 0.75 | ✅ | `--mask-ratio` |
| **per-patch 정규화 MSE** | ✅ | `norm_pix_loss` — 공식 구현 핵심 디테일 |
| **고정 2D sin-cos 위치 임베딩** | ✅ | 학습 파라미터 아님 |
| class token 없음 | ✅ | |
| 공유 마스크 토큰 + 전체 위치 임베딩 | ✅ | `dec_pos_embed` |
| RandomResizedCrop + HFlip, 정규화 없음 | ✅ | `data.py` |
| AdamW(0.9,0.95) + warmup/cosine + layer-wise lr decay(0.75) + grad clip | ✅ | `train.py` |
| 선형 프로빙: 인코더 freeze + 특징 mean/std 정규화 | ✅ | `eval_linear.py` |

> ℹ️ 이전 `../../mae_pretrain_demo.py`(CIFAR 축소판)와의 차이: **sin-cos pos + patch-norm loss** 2가지 논문 디테일을 여기서 추가.

---

## 📁 구조

```
minimae/
├── data/
│   ├── train-0000{0..3}.parquet           # ImageNet-1K 실제 shard (~2GB)
│   ├── subset/{train,val}/<class>/*.jpg   # 서브셋 (391MB) + class_map.json
│   └── imagenet_classes.txt               # 표준 1000 클래스 이름
├── src/
│   ├── prepare_subset.py   # parquet → ImageFolder (+ class_map.json)
│   ├── model.py            # 충실 MAE (ViT-Tiny/Small 프리셋)
│   ├── data.py             # MAE augmentation + 로더
│   ├── train.py            # 사전학습 (config 지원)
│   ├── eval_linear.py      # 선형 프로빙
│   ├── utils.py            # 시드/로거/체크포인트/메트릭/config
│   └── plot_loss.py        # loss 곡선
├── configs/                # tiny / small / eval (YAML)
├── tests/                  # pytest 단위 테스트 (29 passed)
├── docs/                   # HANDOFF.md / DESIGN.md
├── run.sh · Makefile       # 원샷 실행
├── requirements.txt
└── runs/                   # ckpt · metrics · log · loss_curve.png
```

---

## 🚀 실행

```bash
cd paper-diary/day01_mae/minimae
PY=/opt/homebrew/bin/python3.14   # torch 2.14 + torchvision 0.29 환경

# 0) 데이터 다운로드 (이미 있으면 생략)
#    출처: huggingface.co/datasets/mrm8488/ImageNet1K-train (비게이트, 실제 ImageNet-1K)
curl -sL -o data/train-00000.parquet \
  https://huggingface.co/datasets/mrm8488/ImageNet1K-train/resolve/main/data/train-00000-of-00296.parquet

# 1) 서브셋 구성 → 2) 사전학습 → 3) 선형 프로빙 (원샷)
PY=$PY ./run.sh                       # 또는: make all

# 개별 실행 (config 기반; 우선순위 CLI > config > 기본값)
$PY src/prepare_subset.py --shards "data/train-0000*.parquet" --out data/subset --per-class 1200 --val-per-class 60
$PY src/train.py --config configs/tiny.yaml
$PY src/eval_linear.py --ckpt runs/mini_mae_tiny.pt --config configs/eval.yaml
$PY -m pytest -q tests
```
> 상세: [`docs/HANDOFF.md`](./docs/HANDOFF.md)(실행·이관) · [`docs/DESIGN.md`](./docs/DESIGN.md)(설계·논문 대응)

---

## 📈 스케일 업 (논문에 더 가깝게)

| 축 | mini (현재) | 논문 (ViT-B) | 늘리는 법 |
|---|---|---|---|
| 데이터 | 14 클래스 / ~16k | 1000 클래스 / 1.28M | shard 더 받기(296개 전체 ≈ 150GB) + `--per-class`↑ |
| 모델 | ViT-Tiny 3.34M | ViT-B 86M / ViT-H 632M | `--arch vit_small` / model.py에 base 추가 |
| 해상도 | 224 | 224 | 동일 |
| epoch | 20 (완료) | 800 | `--epochs 800` |

> ImageNet-1K 전체(150GB)는 HF `datasets`(게이트, 토큰+동의 필요) 또는 공식 사이트
> (image-net.org, 등록 필요)에서 받아 `--shards`로 서브셋 구성하면 된다.

---

## ⚠️ 참고
- 원본 ImageNet-1K는 게이트/수동 다운로드라 자동화 불가 → 여기선 **비게이트 미러 `mrm8488/ImageNet1K-train`(실제 ImageNet-1K 이미지)** 을 사용.
- 클래스 폴더명은 정수 라벨(0–13) 인덱스. (synset 이름 매핑은 미포함)
