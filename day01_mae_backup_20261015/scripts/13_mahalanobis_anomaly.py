"""Mahalanobis 거리 기반 이상 탐지.
정상 = T09 (사이렌 없는 도메인).
이상 = C01/C02/C04 (사이렌 있는 도메인).
"""
import sys, json
from pathlib import Path

MAE_ROOT = Path("/Users/robotjang/paper-diary/day01_mae")
sys.path.insert(0, str(MAE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining

from data.dataset import FrameMAEDataset
from data.transforms import val_tf

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    # 1) 사전학습 ViTMAE
    print("[model] loading vit-mae-base...")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE).eval()

    # 2) feature 추출
    feats_all = []
    with torch.no_grad():
        for imgs in dl:
            out = model.vit(pixel_values=imgs.to(DEVICE),
                            interpolate_pos_encoding=True)
            feats_all.append(out.last_hidden_state[:, 0, :].cpu())
    feats = torch.cat(feats_all, dim=0).float()   # [80, 768]
    print(f"[feat] {feats.shape}")

    # 3) 정상 분포 (T09)
    is_normal = (ds.df["clip"] == "T09").values
    is_abnormal = ~is_normal
    print(f"[split] normal={is_normal.sum()} abnormal={is_abnormal.sum()}")

    X_normal = feats[is_normal]   # [20, 768]
    X_abnormal = feats[is_abnormal]  # [60, 768]

    # 정상 분포 통계
    mu = X_normal.mean(dim=0)               # [768]
    X_centered = X_normal - mu
    cov = (X_centered.T @ X_centered) / (len(X_normal) - 1)   # [768, 768]
    cov_inv = torch.linalg.pinv(cov)        # [768, 768]
    print(f"[stats] mu.norm={mu.norm():.3f}, cov.trace={cov.trace():.1f}")

    # 4) Mahalanobis 거리
    def mahalanobis(X):
        diff = X - mu                                # [N, 768]
        # d² = diff Σ⁻¹ diffᵀ
        d2 = torch.einsum("nd,de,ne->n", diff, cov_inv, diff)
        return torch.sqrt(torch.clamp(d2, min=0))

    d_normal = mahalanobis(X_normal)
    d_abnormal = mahalanobis(X_abnormal)

    print()
    print("[result] Mahalanobis distance:")
    print(f"  normal   (T09): mean={d_normal.mean():.2f}, std={d_normal.std():.2f}")
    print(f"  abnormal (C*):  mean={d_abnormal.mean():.2f}, std={d_abnormal.std():.2f}")

    # AUC-like 분리도
    all_d = torch.cat([d_normal, d_abnormal])
    labels = torch.cat([torch.zeros_like(d_normal), torch.ones_like(d_abnormal)])
    # 간단 AUC
    order = torch.argsort(all_d)
    labels_sorted = labels[order]
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        rank_sum = torch.where(labels_sorted == 1)[0].float().sum()
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        print(f"\n[result] AUC = {auc.item():.3f}  (0.5=random, 1.0=perfect)")

    # 5) 프레임별 점수 미리보기
    print("\n[sample scores]")
    for clip in ["T09", "C01", "C02", "C04"]:
        idx = (ds.df["clip"] == clip).values
        d_clip = mahalanobis(feats[idx])
        print(f"  {clip}: mean={d_clip.mean():.2f}, max={d_clip.max():.2f}")


if __name__ == "__main__":
    main()