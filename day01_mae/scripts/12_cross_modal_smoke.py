"""Cross-Modal Contrastive (사전학습 ViTMAE + log-mel)."""
import sys, json
from pathlib import Path

MAE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAE_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining

from data.dataset import FrameMAEDataset
from data.transforms import val_tf

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[device] {DEVICE}")


def load_clip_metas():
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d
    return metas


def load_logmel(path, sr=16000, n_mels=128, n_fft=1024, hop=256):
    data, orig_sr = sf.read(str(path), always_2d=True)
    wav = torch.from_numpy(data.T).float()
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    mel = T.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop
    )(wav)
    return torch.log(mel + 1e-6)


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    metas = load_clip_metas()

    # ── 사전학습된 ViTMAE 로드 ─────────────────
    print("[model] loading pretrained vit-mae-base...")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    model = model.to(DEVICE).eval()
    print("[model] loaded.")

    # Vision feature
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    feats_v = []
    with torch.no_grad():
        for imgs in dl:
            imgs = imgs.to(DEVICE)
            out = model.vit(pixel_values=imgs, interpolate_pos_encoding=True)
            feats_v.append(out.last_hidden_state[:, 0, :].cpu())   # CPU로 수집
    feats_v = torch.cat(feats_v, dim=0).float()                    # [80, 768]
    print(f"[vision] feats = {feats_v.shape}")

    # Audio feature (CPU에서 계산)
    audio_feat = {}
    for clip, meta in metas.items():
        audio = meta.get("audio", {})
        clip_dir = Path(meta["_meta_path"]).parent
        for key in ["untouched", "with_siren"]:
            rel = audio.get(key, None)
            if isinstance(rel, str) and (clip_dir / rel).exists():
                mel = load_logmel(clip_dir / rel)
                v = mel.mean(dim=-1).flatten().float()
                audio_feat[(clip, key)] = v

    in_dim = next(iter(audio_feat.values())).shape[0]
    proj = nn.Linear(in_dim, feats_v.shape[1]).to(DEVICE)

    valid_clips = ["C01", "C02", "C04"]
    clip_to_frames = {}
    for i, row in ds.df.iterrows():
        if row["clip"] in valid_clips:
            clip_to_frames.setdefault(row["clip"], []).append(i)

    clip_v, clip_a = {}, {}
    for c in valid_clips:
        idxs = clip_to_frames[c]
        # ★ 여기서 명시적으로 DEVICE로 이동
        clip_v[c] = feats_v[idxs].mean(dim=0).to(DEVICE).float()
        a_in = audio_feat[(c, "with_siren")].to(DEVICE).float()
        clip_a[c] = proj(a_in)

    # ★ stack 후에도 명시적 DEVICE
    V = F.normalize(torch.stack([clip_v[c] for c in valid_clips]).to(DEVICE), dim=-1)
    A = F.normalize(torch.stack([clip_a[c] for c in valid_clips]).to(DEVICE), dim=-1)

    logits = V @ A.T / 0.07
    labels = torch.arange(len(valid_clips), device=DEVICE)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(-1) == labels).float().mean()

    print(f"[infonce] logits = \n{logits.detach().cpu()}")
    print(f"[infonce] loss = {loss.item():.4f}")
    print(f"[infonce] acc = {acc.item():.3f}")


if __name__ == "__main__":
    main()