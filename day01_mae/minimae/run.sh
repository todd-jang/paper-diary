#!/usr/bin/env bash
# run.sh — mini-MAE 전체 파이프라인 한 방 실행 (개발이관용)
#   subset 구성(없으면) → 사전학습 → 선형 프로빙
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/opt/homebrew/bin/python3.14}"     # torch+torchvision 있는 인터프리터
SHARDS="${SHARDS:-data/train-0000*.parquet}"
CONFIG="${CONFIG:-configs/tiny.yaml}"

echo "== [1/3] subset 구성 (이미 있으면 skip) =="
if [ ! -d data/subset/train ]; then
  "$PY" src/prepare_subset.py --shards "$SHARDS" --out data/subset \
        --per-class 1200 --val-per-class 60
else
  echo "data/subset/train 이미 존재 → skip"
fi

echo "== [2/3] MAE 사전학습 =="
"$PY" src/train.py --config "$CONFIG"

echo "== [3/3] 선형 프로빙 =="
CKPT="$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['out'])")"
"$PY" src/eval_linear.py --ckpt "$CKPT" --config configs/eval.yaml

echo "== 완료 =="
