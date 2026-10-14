"""Mahalanobis: C01을 정상으로, C02/C04를 이상으로."""
import sys
from pathlib import Path

MAE_ROOT = Path("/Users/robotjang/paper-diary/day01_mae")
sys.path.insert(0, str(MAE_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining
from data.dataset import FrameMAEDataset
from data.transforms import val_tf

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

NORMAL_CLIP = "C01"
ABNORMAL_CLIPS = ["C02", "C04"]


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE).eval()

    feats_all = []
    with torch.no_grad():
        for imgs in dl:
            out = model.vit(pixel_values=imgs.to(DEVICE),
                            interpolate_pos_encoding=True)
            feats_all.append(out.last_hidden_state[:, 0, :].cpu())
    feats = torch.cat(feats_all, dim=0).float()
    print(f"[feat] {feats.shape}")

    # 정상/이상 분리
    is_normal = (ds.df["clip"] == NORMAL_CLIP).values
    is_abnormal = ds.df["clip"].isin(ABNORMAL_CLIPS).values
    print(f"[split] normal={is_normal.sum()} abnormal={is_abnormal.sum()}")

    X_normal = feats[is_normal]
    X_abnormal = feats[is_abnormal]

    # 정상 분포
    mu = X_normal.mean(dim=0)
    X_centered = X_normal - mu
    cov = (X_centered.T @ X_centered) / (len(X_normal) - 1)
    cov_inv = torch.linalg.pinv(cov)
    print(f"[stats] mu.norm={mu.norm():.3f}, cov.trace={cov.trace():.2f}")

    def mahalanobis(X):
        diff = X - mu
        d2 = torch.einsum("nd,de,ne->n", diff, cov_inv, diff)
        return torch.sqrt(torch.clamp(d2, min=0))

    d_normal = mahalanobis(X_normal)
    d_abnormal = mahalanobis(X_abnormal)

    print(f"\n[result] Mahalanobis distance:")
    print(f"  normal   ({NORMAL_CLIP}): mean={d_normal.mean():.2f}, std={d_normal.std():.2f}")
    print(f"  abnormal ({ABNORMAL_CLIPS}): mean={d_abnormal.mean():.2f}, std={d_abnormal.std():.2f}")

    all_d = torch.cat([d_normal, d_abnormal])
    labels = torch.cat([torch.zeros_like(d_normal), torch.ones_like(d_abnormal)])
    order = torch.argsort(all_d)
    labels_sorted = labels[order]
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    rank_sum = torch.where(labels_sorted == 1)[0].float().sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print(f"\n[result] AUC = {auc.item():.3f}")

    print(f"\n[sample scores]")
    for clip in ["C01", "C02", "C04", "T09"]:
        idx = (ds.df["clip"] == clip).values
        d_clip = mahalanobis(feats[idx])
        print(f"  {clip}: mean={d_clip.mean():.2f}, std={d_clip.std():.2f}, max={d_clip.max():.2f}")


if __name__ == "__main__":
    main()