"""08과 동일하지만, T09를 학습에서 제외하고 평가."""
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
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
GT_KEY = "scenario"


def load_clip_metas():
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        metas[d["clip_id"]] = d
    return metas


def build_labels(df, metas, key):
    clip_to_label = {}
    for cid, m in metas.items():
        v = m.get(key, None)
        clip_to_label[cid] = str(v) if v not in (None, "") else "unknown"
    classes = sorted(set(clip_to_label.values()))
    cls2idx = {c: i for i, c in enumerate(classes)}
    labels = [cls2idx[clip_to_label[row["clip"]]] for _, row in df.iterrows()]
    return torch.tensor(labels, dtype=torch.long), classes


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    metas = load_clip_metas()
    labels, classes = build_labels(ds.df, metas, GT_KEY)
    print(f"[meta] classes = {classes}")
    print(f"[meta] clips   = {ds.df['clip'].unique().tolist()}")
    print(f"[meta] splits  = {ds.df['split'].unique().tolist()}")

    # Train/Test 마스크
    is_train = (ds.df["split"] == "train").values
    is_test  = (ds.df["split"] == "test").values
    print(f"[split] train frames = {is_train.sum()}, test frames = {is_test.sum()}")

    # ViTMAE
    config = ViTMAEConfig(
        image_size=224, patch_size=16,
        hidden_size=384, num_hidden_layers=6,
        num_attention_heads=6, intermediate_size=1536,
        mask_ratio=0.0,
    )
    model = ViTMAEForPreTraining(config).to(DEVICE).eval()

    # Feature 추출
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    feats_all = []
    with torch.no_grad():
        for imgs in dl:
            out = model.vit(pixel_values=imgs.to(DEVICE),
                            interpolate_pos_encoding=True)
            feats_all.append(out.last_hidden_state[:, 0, :].cpu())
    feats = torch.cat(feats_all, dim=0)   # [80, 384]
    print(f"[feature] shape = {feats.shape}")

    # Train으로만 head 학습
    X_tr = feats[is_train].to(DEVICE)
    y_tr = labels[is_train].to(DEVICE)
    X_te = feats[is_test].to(DEVICE)
    y_te = labels[is_test].to(DEVICE)

    head = nn.Linear(384, len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

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

    print("[done] hold-out probe finished.")


if __name__ == "__main__":
    main()