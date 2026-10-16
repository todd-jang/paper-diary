"""Siren 감지 (binary): frame + audio → 0 or 1."""
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

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_logmel(path):
    import soundfile as sf
    import torchaudio
    import torchaudio.transforms as T
    data, sr = sf.read(str(path), always_2d=True)
    wav = torch.from_numpy(data.T).float()
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    mel = T.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024,
                           hop_length=256)(wav)
    lm = torch.log(mel + 1e-6).squeeze(0)   # [128, T]
    T_cur = lm.shape[-1]
    if T_cur < 1024:
        lm = F.pad(lm, (0, 1024 - T_cur))
    else:
        lm = lm[..., :1024]
    return lm


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

    # Vision features
    vit = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").vit.to(DEVICE).eval()
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    feats_v = []
    with torch.no_grad():
        for imgs in dl:
            out = vit(pixel_values=imgs.to(DEVICE), interpolate_pos_encoding=True)
            feats_v.append(out.last_hidden_state[:, 0, :].cpu())
    feats_v = torch.cat(feats_v, dim=0).float().to(DEVICE)   # [80, 768]

    # Audio features (클립별 2개)
    ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(DEVICE).eval()
    clip_audio = {}
    for clip, meta in metas.items():
        clip_audio[clip] = {}
        for key in ["untouched", "with_siren"]:
            rel = meta.get("audio", {}).get(key, None)
            if not isinstance(rel, str):
                continue
            p = Path(meta["_meta_path"]).parent / rel
            if not p.exists():
                continue
            lm = load_logmel(p).unsqueeze(0)   # [1, 128, 1024]
            x = lm.transpose(1, 2)              # [1, 1024, 128]
            x = (x - x.mean()) / (x.std() + 1e-6)
            with torch.no_grad():
                out = ast(x.to(DEVICE))
            clip_audio[clip][key] = out.pooler_output.squeeze(0)   # [768]

    print(f"[audio] clips = {list(clip_audio.keys())}")
    for c, d in clip_audio.items():
        print(f"  {c}: {list(d.keys())}")

    # Pairwise 유사도
    print("\n[similarity] vision vs audio:")
    for clip in ["C01", "C02", "C04"]:
        if clip not in clip_audio:
            continue
        frames_idx = [i for i, row in ds.df.iterrows() if row["clip"] == clip]
        v_mean = F.normalize(feats_v[frames_idx].mean(0, keepdim=True), dim=-1)
        for key in ["untouched", "with_siren"]:
            if key not in clip_audio[clip]:
                continue
            a = F.normalize(clip_audio[clip][key].unsqueeze(0), dim=-1)
            sim = (v_mean @ a.T).item()
            print(f"  {clip} + {key:12s}: cos = {sim:.4f}")

    # Binary classification
    # positive: (frame, audio_with_siren)
    # negative: (frame, audio_untouched)
    X, y = [], []
    for clip in ["C01", "C02", "C04"]:
        if clip not in clip_audio:
            continue
        frames_idx = [i for i, row in ds.df.iterrows() if row["clip"] == clip]
        v = feats_v[frames_idx].mean(0)   # [768]
        for key, label in [("untouched", 0), ("with_siren", 1)]:
            if key not in clip_audio[clip]:
                continue
            a = clip_audio[clip][key]
            pair = torch.cat([v, a])   # [1536]
            X.append(pair)
            y.append(label)

    if len(X) < 2:
        print("[warn] not enough pairs")
        return

    X = torch.stack(X)   # [N, 1536]
    y = torch.tensor(y, device=DEVICE)

    # Linear head
    head = nn.Linear(1536, 2).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    print(f"\n[binary] X={X.shape}, y={y.shape}")
    for step in range(100):
        logits = head(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            acc = (logits.argmax(-1) == y).float().mean()
            print(f"[step {step:03d}] loss={loss.item():.4f} acc={acc.item():.3f}")


if __name__ == "__main__":
    main()