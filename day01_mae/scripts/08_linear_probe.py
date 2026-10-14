"""07 feature → linear probing (여러 GT 태스크 지원)."""
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

# ── 사용할 GT 키 (필요시 변경) ──────────────────────
GT_KEY = "scenario"   # ← 여기만 바꾸면 다른 태스크
# GT_KEY = "siren_track"
# GT_KEY = "condition"
# GT_KEY = "view"


def load_clip_metas():
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        metas[d["clip_id"]] = d
    return metas


def build_labels(df, metas, key):
    """None 안전 + 문자열 강제."""
    clip_to_label = {}
    for clip_id, meta in metas.items():
        v = meta.get(key, None)
        if v is None or v == "":
            v = "unknown"
        clip_to_label[clip_id] = str(v)

    classes = sorted(set(clip_to_label.values()))
    cls2idx = {c: i for i, c in enumerate(classes)}

    labels = []
    for _, row in df.iterrows():
        clip = row["clip"]
        labels.append(cls2idx.get(clip_to_label.get(clip, "unknown"), 0))

    return torch.tensor(labels, dtype=torch.long), classes


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    metas = load_clip_metas()
    print(f"[meta] {len(metas)} clips: {list(metas.keys())}")

    # clip별 GT 출력 (디버깅)
    print(f"[meta] GT key = {GT_KEY!r}")
    for cid, m in sorted(metas.items()):
        v = m.get(GT_KEY, None)
        print(f"  {cid}: {v!r}")

    # ViTMAE
    config = ViTMAEConfig(
        image_size=224, patch_size=16,
        hidden_size=384, num_hidden_layers=6,
        num_attention_heads=6, intermediate_size=1536,
        mask_ratio=0.0,
    )
    model = ViTMAEForPreTraining(config).to(DEVICE).eval()

    # Feature
    all_feats = []
    with torch.no_grad():
        for imgs in dl:
            out = model.vit(pixel_values=imgs.to(DEVICE),
                            interpolate_pos_encoding=True)
            all_feats.append(out.last_hidden_state[:, 0, :].cpu())
    feats = torch.cat(all_feats, dim=0)

    # Labels
    labels, classes = build_labels(ds.df, metas, key=GT_KEY)
    print(f"[labels] classes = {classes}")
    print(f"[labels] counts  = {torch.bincount(labels).tolist()}")

    # Sanity check
    if len(classes) < 2:
        print(f"[warn] classes < 2 ({len(classes)}). GT_KEY='{GT_KEY}'로는 "
              f"분류 태스크 불가. 다른 키를 시도하세요.")
        return

    # Linear probing
    head = nn.Linear(384, len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    for step in range(100):
        logits = head(feats.to(DEVICE))
        loss = F.cross_entropy(logits, labels.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            acc = (logits.argmax(-1) == labels.to(DEVICE)).float().mean()
            print(f"[step {step:03d}] loss={loss.item():.4f} acc={acc.item():.3f}")

    print("[done] linear probing finished.")


if __name__ == "__main__":
    main()