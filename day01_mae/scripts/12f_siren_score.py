"""Siren score: cos(vision, untouched) - cos(vision, with_siren)."""
import sys, json
from pathlib import Path

MAE_ROOT = Path("/Users/robotjang/paper-diary/day01_mae")
sys.path.insert(0, str(MAE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ASTModel

from data.dataset import FrameMAEDataset
from data.transforms import val_tf

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

    # Vision features
    vit = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").vit.to(DEVICE).eval()
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    feats_v = []
    with torch.no_grad():
        for imgs in dl:
            out = vit(pixel_values=imgs.to(DEVICE), interpolate_pos_encoding=True)
            feats_v.append(out.last_hidden_state[:, 0, :].cpu())
    feats_v = torch.cat(feats_v, dim=0).float().to(DEVICE)

    # Audio features
    ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(DEVICE).eval()
    import soundfile as sf
    import torchaudio
    import torchaudio.transforms as T

    def load_logmel(path):
        data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(data.T).float()
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        mel = T.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024,
                               hop_length=256)(wav)
        lm = torch.log(mel + 1e-6).squeeze(0)
        Tc = lm.shape[-1]
        lm = F.pad(lm, (0, 1024 - Tc)) if Tc < 1024 else lm[..., :1024]
        return lm

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
            lm = load_logmel(str(p)).unsqueeze(0)
            x = lm.transpose(1, 2)
            x = (x - x.mean()) / (x.std() + 1e-6)
            with torch.no_grad():
                out = ast(x.to(DEVICE))
            clip_audio[clip][key] = out.pooler_output.squeeze(0)

    # Siren score 계산
    print("=== Siren Score ===")
    print(f"{'clip':>5s}  {'untouched':>10s}  {'with_siren':>10s}  {'score':>8s}")
    print("-" * 45)
    scores = {}
    for clip in ["C01", "C02", "C04"]:
        frames_idx = [i for i, row in ds.df.iterrows() if row["clip"] == clip]
        v = F.normalize(feats_v[frames_idx].mean(0, keepdim=True), dim=-1)

        if "untouched" in clip_audio[clip] and "with_siren" in clip_audio[clip]:
            a_u = F.normalize(clip_audio[clip]["untouched"].unsqueeze(0), dim=-1)
            a_s = F.normalize(clip_audio[clip]["with_siren"].unsqueeze(0), dim=-1)
            cos_u = (v @ a_u.T).item()
            cos_s = (v @ a_s.T).item()
            score = cos_u - cos_s
            scores[clip] = score
            print(f"{clip:>5s}  {cos_u:>10.4f}  {cos_s:>10.4f}  {score:>+8.4f}")

    # T09 (untouched만)
    print()
    clip = "T09"
    frames_idx = [i for i, row in ds.df.iterrows() if row["clip"] == clip]
    v = F.normalize(feats_v[frames_idx].mean(0, keepdim=True), dim=-1)
    if "untouched" in clip_audio.get(clip, {}):
        a_u = F.normalize(clip_audio[clip]["untouched"].unsqueeze(0), dim=-1)
        cos_u = (v @ a_u.T).item()
        print(f"{clip:>5s}  {cos_u:>10.4f}  (no with_siren)")

    # 판정
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n[avg siren score] = {avg:+.4f}")
        print(f"[std] = {(sum((s-avg)**2 for s in scores.values())/len(scores))**0.5:.4f}")


if __name__ == "__main__":
    main()