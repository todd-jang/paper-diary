"""
eval_linear.py — 선형 프로빙 (인코더 freeze + linear head)

MAE의 진짜 가치는 복원 품질이 아니라 인코더 표현 → 인코더를 freeze하고
평균 풀링 특징에 선형 분류기만 학습해 top-1 정확도를 측정한다.

  py eval_linear.py --ckpt runs/mini_mae_tiny.pt --config configs/eval.yaml
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from model import mae_vit_tiny, mae_vit_small
from data import build_train_loader, build_eval_loader
from utils import seed_everything, get_device, get_logger, load_checkpoint, load_config, merge_config

DEFAULTS = dict(
    img_size=224, patch_size=16, epochs=30, batch_size=128, lr=1.0e-3, wd=0.0,
    num_workers=4, seed=0, train="data/subset/train", val="data/subset/val",
    device="auto", out="runs/linear_results.json",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        ap.add_argument(f"--{k.replace('_','-')}", dest=k, type=t, default=None)
    a = ap.parse_args()
    cfg = merge_config(DEFAULTS, load_config(a.config), vars(a))
    cfg.pop("config", None)
    return cfg, a.ckpt


def build_encoder(ckpt, device):
    a = ckpt["args"]
    model = {"vit_tiny": mae_vit_tiny, "vit_small": mae_vit_small}[a["arch"]](
        img_size=a["img_size"], patch_size=a["patch_size"],
        mask_ratio=0.0, norm_pix_loss=bool(a.get("norm_pix_loss", True)))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    feats, labels = [], []
    for imgs, y in loader:
        x = model.patch_embed(imgs.to(device)) + model.enc_pos_embed
        for blk in model.enc_blocks:
            x = blk(x)
        x = model.enc_norm(x)
        feats.append(x.mean(dim=1).cpu()); labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    cfg, ckpt_path = parse_args()
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    log = get_logger("minimae.linear")
    ckpt = load_checkpoint(ckpt_path, "cpu")
    model = build_encoder(ckpt, device)
    log.info(f"device={device} arch={ckpt['args']['arch']} pretrain_epoch={ckpt.get('epoch')}")

    tr_loader, tr_ds = build_train_loader(cfg["train"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"])
    va_loader, va_ds = build_eval_loader(cfg["val"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"])
    nclass = len(tr_ds.classes)
    log.info(f"train={len(tr_ds)} val={len(va_ds)} classes={nclass}")

    Xtr, ytr = extract_features(model, tr_loader, device)
    Xva, yva = extract_features(model, va_loader, device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6   # MAE linear probe: 특징 정규화
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Xtr, ytr, Xva, yva = Xtr.to(device), ytr.to(device), Xva.to(device), yva.to(device)

    clf = nn.Linear(Xtr.shape[1], nclass).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.CrossEntropyLoss()
    bs = cfg["batch_size"]
    best = 0.0
    for ep in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            loss = lossf(clf(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == cfg["epochs"]:
            with torch.no_grad():
                acc = (clf(Xva).argmax(1) == yva).float().mean().item()
            best = max(best, acc)
            log.info(f"[linear ep {ep}] loss={tot/max(1,len(Xtr)//bs):.4f} val_acc={acc*100:.2f}%")

    res = {"ckpt": ckpt_path, "arch": ckpt["args"]["arch"], "pretrain_epoch": ckpt.get("epoch"),
           "nclass": nclass, "train": len(tr_ds), "val": len(va_ds),
           "best_val_acc": round(best * 100, 2)}
    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    with open(cfg["out"], "w") as f:
        json.dump(res, f, indent=2)
    log.info(f"done. best_val_acc={best*100:.2f}% -> {cfg['out']}")


if __name__ == "__main__":
    main()
