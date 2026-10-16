"""Frame-level InfoNCE: 60 프레임 각각에 대해 대조."""
import sys, json
from pathlib import Path

MAE_ROOT = Path("/Users/robotjang/paper-diary/day01_mae")
sys.path.insert(0, str(MAE_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ASTModel

from data.dataset import FrameMAEDataset
from data.transforms import val_tf
if __package__:
    from ._ast_utils import load_logmel
else:
    from _ast_utils import load_logmel

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

    # ── Vision features ──
    vit = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").vit.to(DEVICE).eval()
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    feats_v = []
    with torch.no_grad():
        for imgs in dl:
            out = vit(pixel_values=imgs.to(DEVICE), interpolate_pos_encoding=True)
            feats_v.append(out.last_hidden_state[:, 0, :].cpu())
    feats_v = torch.cat(feats_v, dim=0).float().to(DEVICE)   # [80, 768]

    # ── Audio features (클립별 with_siren 1개) ──
    ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(DEVICE).eval()
    clip_audio = {}
    for clip, meta in metas.items():
        rel = meta.get("audio", {}).get("with_siren", None)
        if not isinstance(rel, str):
            continue
        p = Path(meta["_meta_path"]).parent / rel
        if not p.exists():
            continue
        log_mel = load_logmel(str(p)).unsqueeze(0)   # [1, 128, T]
        x = log_mel.transpose(1, 2)                    # [1, T, 128]
        x = (x - x.mean()) / (x.std() + 1e-6)
        with torch.no_grad():
            out = ast(x.to(DEVICE))
        clip_audio[clip] = out.pooler_output.squeeze(0).to(DEVICE)   # [768]

    # ── Frame-level InfoNCE ──
    # 각 프레임의 vision feature ↔ 그 클립의 audio feature
    valid_clips = [c for c in ["C01", "C02", "C04"] if c in clip_audio]
    frame_idx = [i for i, row in ds.df.iterrows() if row["clip"] in valid_clips]
    clip_of_frame = [ds.df.iloc[i]["clip"] for i in frame_idx]

    V = F.normalize(feats_v[frame_idx], dim=-1)   # [60, 768]
    A = F.normalize(torch.stack([clip_audio[c] for c in valid_clips]), dim=-1)   # [3, 768]

    # 60 x 3 logits
    logits = V @ A.T / 0.07   # [60, 3]

    # label: 각 프레임이 어느 클립인지
    label_of_clip = {c: i for i, c in enumerate(valid_clips)}
    labels = torch.tensor([label_of_clip[c] for c in clip_of_frame], device=DEVICE)

    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(-1) == labels).float().mean()

    print(f"[frame-level] V={V.shape}, A={A.shape}, logits={logits.shape}")
    print(f"[frame-level] loss = {loss.item():.4f}")
    print(f"[frame-level] acc = {acc.item():.3f}")

    # 클립별 정확도
    for c in valid_clips:
        mask = torch.tensor([cf == c for cf in clip_of_frame], device=DEVICE)
        c_acc = (logits[mask].argmax(-1) == labels[mask]).float().mean()
        print(f"  {c}: acc = {c_acc.item():.3f}")


if __name__ == "__main__":
    main()