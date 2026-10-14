"""Time-Contrastive: 같은 클립 프레임은 가깝게, 다른 클립은 멀게."""
import sys, json
from pathlib import Path

MAE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAE_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ViTMAEConfig

from data.dataset import FrameMAEDataset
from data.transforms import val_tf

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    # ViTMAE
    config = ViTMAEConfig(
        image_size=224, patch_size=16,
        hidden_size=384, num_hidden_layers=6,
        num_attention_heads=6, intermediate_size=1536,
        mask_ratio=0.0,
    )
    model = ViTMAEForPreTraining(config).to(DEVICE).eval()

    # Feature 추출
    feats_all = []
    with torch.no_grad():
        for imgs in dl:
            out = model.vit(pixel_values=imgs.to(DEVICE),
                            interpolate_pos_encoding=True)
            feats_all.append(out.last_hidden_state[:, 0, :].cpu())
    feats = torch.cat(feats_all, dim=0)   # [80, 384]

    # 클립 라벨 (0=C01, 1=C02, 2=C04, 3=T09)
    clips = ds.df["clip"].unique().tolist()
    clip2idx = {c: i for i, c in enumerate(clips)}
    labels = torch.tensor([clip2idx[c] for c in ds.df["clip"]])

    print(f"[clips] {clips}")
    print(f"[feats] {feats.shape}")

    # Linear probing: 클립 분류
    head = nn.Linear(384, len(clips)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    # Train/Test split
    is_train = (ds.df["split"] == "train").values.copy()
    is_test  = (ds.df["split"] == "test").values.copy()
    X_tr = feats[is_train].to(DEVICE).float()
    y_tr = labels[is_train].to(DEVICE)
    X_te = feats[is_test].to(DEVICE).float()
    y_te = labels[is_test].to(DEVICE)
    print(f"[split] train={X_tr.shape} test={X_te.shape}")

    for step in range(200):
        logits = head(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 40 == 0:
            with torch.no_grad():
                tr_acc = (head(X_tr).argmax(-1) == y_tr).float().mean()
                te_acc = (head(X_te).argmax(-1) == y_te).float().mean()
            print(f"[step {step:03d}] loss={loss.item():.4f} "
                  f"train_acc={tr_acc.item():.3f} test_acc={te_acc.item():.3f}")


if __name__ == "__main__":
    main()
    