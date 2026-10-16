"""
train.py — mini MAE 사전학습 (FAIR MAE 하이퍼파라미터 충실)

우선순위: CLI 인자 > --config 파일 > DEFAULTS
  py train.py --config configs/tiny.yaml
  py train.py --config configs/tiny.yaml --epochs 5 --batch-size 32   # 오버라이드
"""
import argparse
import math
import time

import torch

from model import mae_vit_tiny, mae_vit_small
from data import build_train_loader
from utils import (seed_everything, get_device, get_logger, MetricWriter,
                   save_checkpoint, load_config, merge_config)

DEFAULTS = dict(
    arch="vit_tiny", img_size=224, patch_size=16, mask_ratio=0.75,
    epochs=20, batch_size=64, lr=1.5e-4, weight_decay=0.05, warmup_epochs=5,
    layer_decay=0.75, min_lr=1.0e-6, clip_grad=1.0, norm_pix_loss=True,
    num_workers=4, seed=0, log_every=50,
    data="data/subset/train", out="runs/mini_mae.pt",
    metrics="runs/metrics.csv", logfile=None, device="auto",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML/JSON 설정 파일")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_','-')}", dest=k, action="store_true", default=None)
            ap.add_argument(f"--no-{k.replace('_','-')}", dest=k, action="store_false")
        else:
            t = type(v) if v is not None else str
            ap.add_argument(f"--{k.replace('_','-')}", dest=k, type=t, default=None)
    a = ap.parse_args()
    cfg = merge_config(DEFAULTS, load_config(a.config), vars(a))
    cfg.pop("config", None)
    return cfg


def param_groups_lrd(model, weight_decay, layer_decay):
    """layer-wise lr decay (MAE official): 깊은 블록일수록 lr↓, bias/norm은 wd 제외."""
    no_decay = {"bias", "norm", "pos_embed", "mask_token"}
    num_layers = len(model.enc_blocks) + len(model.dec_blocks) + 1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers)]

    def get_layer_id(name):
        if name.startswith("enc_blocks"):
            return int(name.split(".")[1]) + 1
        if name.startswith("dec_blocks"):
            return len(model.enc_blocks) + int(name.split(".")[1]) + 1
        return 0

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = get_layer_id(name)
        scale = layer_scales[lid] if lid < len(layer_scales) else 1.0
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        groups.setdefault((round(scale, 8), wd), []).append(p)
    return [{"params": v, "lr_scale": k[0], "weight_decay": k[1]} for k, v in groups.items()]


def cosine_schedule(base_lr, min_lr, epochs, warmup_epochs, steps_per_epoch):
    total = epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch

    def fn(step):
        if step < warm:
            return base_lr * step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))

    return fn


def main():
    cfg = parse_args()
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    log = get_logger("minimae.train", cfg.get("logfile"))
    log.info(f"device={device} seed={cfg['seed']}")
    log.info(f"config={cfg}")

    model = {"vit_tiny": mae_vit_tiny, "vit_small": mae_vit_small}[cfg["arch"]](
        img_size=cfg["img_size"], patch_size=cfg["patch_size"],
        mask_ratio=cfg["mask_ratio"], norm_pix_loss=bool(cfg["norm_pix_loss"]),
    ).to(device)
    n = sum(p.numel() for p in model.parameters())
    log.info(f"model={cfg['arch']} params={n/1e6:.2f}M patches={model.patch_embed.num_patches} "
             f"keep={int(model.patch_embed.num_patches*(1-cfg['mask_ratio']))}")

    loader, ds = build_train_loader(cfg["data"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"])
    log.info(f"data={len(ds)} imgs, {len(ds.classes)} classes, {len(loader)} steps/epoch")

    opt = torch.optim.AdamW(param_groups_lrd(model, cfg["weight_decay"], cfg["layer_decay"]),
                            lr=cfg["lr"], betas=(0.9, 0.95))
    sched_fn = cosine_schedule(cfg["lr"], cfg["min_lr"], cfg["epochs"], cfg["warmup_epochs"], len(loader))
    mw = MetricWriter(cfg["metrics"], ["epoch", "loss", "lr", "sec"])

    step = 0
    model.train()
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        run = 0.0
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            lr = sched_fn(step)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)
            loss, _, _ = model(imgs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            opt.step()
            run += loss.item(); step += 1
            if (i + 1) % cfg["log_every"] == 0:
                log.info(f"ep{ep} step{i+1}/{len(loader)} loss={run/(i+1):.4f} lr={lr:.2e}")
        avg = run / len(loader); sec = time.time() - t0
        log.info(f"[epoch {ep}/{cfg['epochs']}] mask-MSE={avg:.4f} ({sec:.1f}s)")
        mw.log(epoch=ep, loss=f"{avg:.6f}", lr=f"{lr:.3e}", sec=f"{sec:.1f}")
        save_checkpoint(cfg["out"], model, cfg, ep, optimizer=opt)
    log.info(f"done. ckpt -> {cfg['out']}")


if __name__ == "__main__":
    main()
