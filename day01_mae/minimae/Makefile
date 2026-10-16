# Makefile — mini-MAE 개발이관용 단축 명령
PY ?= /opt/homebrew/bin/python3.14
SHARDS ?= data/train-0000*.parquet
CONFIG ?= configs/tiny.yaml

.PHONY: help subset train eval test clean

help:
	@echo "make subset  - parquet → ImageFolder 서브셋"
	@echo "make train   - MAE 사전학습 (CONFIG=configs/tiny.yaml)"
	@echo "make eval    - 선형 프로빙"
	@echo "make test    - 단위 테스트 (pytest)"
	@echo "make all     - subset + train + eval"

subset:
	$(PY) src/prepare_subset.py --shards "$(SHARDS)" --out data/subset --per-class 1200 --val-per-class 60

train:
	$(PY) src/train.py --config $(CONFIG)

eval:
	$(PY) src/eval_linear.py --ckpt runs/mini_mae_tiny.pt --config configs/eval.yaml

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf runs/*.pt runs/*.csv runs/*.log src/__pycache__ tests/__pycache__ .pytest_cache

all: subset train eval
