"""utils.py — 공통 유틸 (재현성·로깅·체크포인트·메트릭). 개발이관 품질."""
import os
import csv
import json
import time
import random
import logging

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 재현성
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
def get_logger(name: str = "minimae", logfile: str | None = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        fh = logging.FileHandler(logfile); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# 메트릭 CSV (epoch별 append)
# ---------------------------------------------------------------------------
class MetricWriter:
    def __init__(self, path: str, fields: list[str]):
        self.path = path
        self.fields = fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def log(self, **kw):
        row = [kw.get(k, "") for k in self.fields]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model: torch.nn.Module, args: dict, epoch: int,
                    optimizer=None, best: float | None = None, extra: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "args": args,
        "epoch": epoch,
        "best": best,
        "saved_at": time.time(),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# config(JSON/YAML) 병합: CLI > config 파일 > 기본값
# ---------------------------------------------------------------------------
def load_config(path: str | None) -> dict:
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f) or {}
        if ext == ".json":
            return json.load(f)
    raise ValueError(f"지원하지 않는 config 형식: {ext}")


def merge_config(defaults: dict, config: dict, cli: dict) -> dict:
    """우선순위: CLI(None 아님) > config > defaults."""
    out = dict(defaults)
    out.update({k: v for k, v in config.items() if k in out})
    out.update({k: v for k, v in cli.items() if v is not None})
    return out
