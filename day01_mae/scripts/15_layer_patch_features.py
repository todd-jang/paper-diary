"""ViTMAE의 여러 레이어 + patch tokens feature로 이상 탐지."""
import sys
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


def extract_features(model, dl, mode="cls", layer=-1):
    """feature 추출 (CLS / patch mean / 특정 layer CLS)."""
    feats = []
    with torch.no_grad():
        for imgs in dl:
            out = model.vit(
                pixel_values=imgs.to(DEVICE),
                interpolate_pos_encoding=True,
                output_hidden_states=True,
            )
            if mode == "cls":
                h = out.hidden_states[layer][:, 0, :]
            elif mode == "patch_mean":
                h = out.hidden_states[layer][:, 1:, :].mean(dim=1)
            elif mode == "cls_patch_concat":
                cls = out.hidden_states[layer][:, 0, :]
                patch = out.hidden_states[layer][:, 1:, :].mean(dim=1)
                h = torch.cat([cls, patch], dim=-1)
            feats.append(h.cpu())
    return torch.cat(feats, dim=0).float()


def mahalanobis_auc(X_normal, X_abnormal):
    """Mahalanobis AUC 계산."""
    mu = X_normal.mean(dim=0)
    Xc = X_normal - mu
    cov = (Xc.T @ Xc) / (len(X_normal) - 1)
    cov_inv = torch.linalg.pinv(cov + 1e-4 * torch.eye(cov.shape[0]))

    def d(X):
        diff = X - mu
        d2 = torch.einsum("nd,de,ne->n", diff, cov_inv, diff)
        return torch.sqrt(torch.clamp(d2, min=0))

    d_n = d(X_normal)
    d_a = d(X_abnormal)
    all_d = torch.cat([d_n, d_a])
    labels = torch.cat([torch.zeros_like(d_n), torch.ones_like(d_a)])
    order = torch.argsort(all_d)
    labels_sorted = labels[order]
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    rank_sum = torch.where(labels_sorted == 1)[0].float().sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc.item(), d_n, d_a


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE).eval()

    is_normal = (ds.df["clip"] == "T09").values
    is_abnormal = (ds.df["clip"] != "T09").values

    configs = [
        ("CLS layer 12 (last)", "cls", -1),
        ("CLS layer 6", "cls", 6),
        ("CLS layer 9", "cls", 9),
        ("patch_mean layer 12", "patch_mean", -1),
        ("patch_mean layer 6", "patch_mean", 6),
        ("cls+patch layer 12", "cls_patch_concat", -1),
    ]

    print(f"{'config':30s} {'AUC':>6s} {'norm_mean':>10s} {'abn_mean':>10s}")
    print("-" * 62)

    for name, mode, layer in configs:
        try:
            feats = extract_features(model, dl, mode=mode, layer=layer)
            X_n = feats[is_normal]
            X_a = feats[is_abnormal]
            auc, d_n, d_a = mahalanobis_auc(X_n, X_a)
            print(f"{name:30s} {auc:6.3f} {d_n.mean():10.2f} {d_a.mean():10.2f}")
        except Exception as e:
            print(f"{name:30s} ERROR: {e}")


if __name__ == "__main__":
    main()