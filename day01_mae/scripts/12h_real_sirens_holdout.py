"""Score clip vision features by cosine similarity to real-siren audio features.

exports/real_sirens/*.wav 사용.
The reported siren score is cos(vision, real_siren), not a difference
against an untouched-road audio embedding.
"""
from __future__ import annotations

import sys
from pathlib import Path

MAE_ROOT = Path("/Users/robotjang/paper-diary/day01_mae")
sys.path.insert(0, str(MAE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ASTModel

from data.dataset import FrameMAEDataset
from data.transforms import val_tf

# ── 설정 ──────────────────────────────────────
CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
REAL_SIRENS = Path("/Users/robotjang/spatial-avla-pipeline/exports/real_sirens")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

AUDIO_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TARGET_LEN = 1024


# ── log-mel 로드 ─────────────────────────────
def load_logmel(path, sr=AUDIO_SR, n_mels=N_MELS, n_fft=N_FFT,
                hop=HOP, target_len=TARGET_LEN):
    data, orig_sr = sf.read(str(path), always_2d=True)
    wav = torch.from_numpy(data.T).float()
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    mel = T.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop
    )(wav)
    lm = torch.log(mel + 1e-6).squeeze(0)
    Tc = lm.shape[-1]
    if Tc < target_len:
        lm = F.pad(lm, (0, target_len - Tc))
    else:
        lm = lm[..., :target_len]
    return lm


def get_ast_feature(ast, path):
    lm = load_logmel(str(path)).unsqueeze(0)
    x = lm.transpose(1, 2)
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        out = ast(x.to(DEVICE))
    return out.pooler_output.squeeze(0)


def cos_sim(v, a):
    return F.cosine_similarity(v.unsqueeze(0), a.unsqueeze(0), dim=-1).item()


def main():
    print(f"[device] {DEVICE}")
    ds = FrameMAEDataset(CSV, transform=val_tf)

    # ── Vision features ─────────────────────
    print("[model] loading ViTMAE...")
    vit = ViTMAEForPreTraining.from_pretrained(
        "facebook/vit-mae-base"
    ).vit.to(DEVICE).eval()

    dl = DataLoader(ds, batch_size=8, shuffle=False)
    feats_v_all = []
    with torch.no_grad():
        for imgs in dl:
            out = vit(pixel_values=imgs.to(DEVICE),
                      interpolate_pos_encoding=True)
            feats_v_all.append(out.last_hidden_state[:, 0, :].cpu())
    feats_v_all = torch.cat(feats_v_all, dim=0).float().to(DEVICE)

    def clip_vision_mean(clip):
        idx = [i for i, row in ds.df.iterrows() if row["clip"] == clip]
        if not idx:
            raise ValueError(f"No frames found for clip {clip!r} in {CSV}")
        v = feats_v_all[idx].mean(dim=0)
        return F.normalize(v, dim=-1)

    # ── Audio features (AST) ────────────────
    print("[model] loading AST...")
    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    # real_sirens feature
    print()
    print("[real_sirens] AST features:")
    real_siren_files = sorted(REAL_SIRENS.glob("G44-*_16k.wav"))
    if not real_siren_files:
        raise FileNotFoundError(
            f"No real-siren WAV files matching G44-*_16k.wav in {REAL_SIRENS}"
        )
    real_siren_feats = {}
    for p in real_siren_files:
        name = p.stem.replace("_16k", "")
        print(f"  {name} ...", end=" ", flush=True)
        real_siren_feats[name] = F.normalize(get_ast_feature(ast, p), dim=-1)
        print("done")

    print(f"\n[real_sirens] {len(real_siren_feats)} files loaded")

    # ── Hold-out A: T09 vision + real_sirens ─
    print()
    print("=" * 70)
    print("HOLD-OUT A: T09 vision + real_sirens")
    print("  (해외 도로 + 실제 사이렌)")
    print("  score_siren = cos(T09 vision, real_siren audio)")
    print("=" * 70)
    v_t09 = clip_vision_mean("T09")
    print()
    print(f"{'real_siren':<35s}  {'score_siren':>12s}")
    print("-" * 52)

    t09_scores = {}
    for name, a_s in real_siren_feats.items():
        score_siren = cos_sim(v_t09, a_s)
        t09_scores[name] = score_siren
        print(f"{name:<35s}  {score_siren:>+12.4f}")

    # ── Hold-out B: 한국 클립 + real_sirens ─
    print()
    print("=" * 70)
    print("HOLD-OUT B: C01 vision + real_sirens")
    print("  (한국 도로 + 실제 사이렌)")
    print("  score_siren = cos(C01 vision, real_siren audio)")
    print("=" * 70)
    v_c01 = clip_vision_mean("C01")
    print()
    print(f"{'real_siren':<35s}  {'score_siren':>12s}")
    print("-" * 52)

    c01_scores = {}
    for name, a_s in real_siren_feats.items():
        score_siren = cos_sim(v_c01, a_s)
        c01_scores[name] = score_siren
        print(f"{name:<35s}  {score_siren:>+12.4f}")

    # ── 요약 ────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if t09_scores:
        avg_t = sum(t09_scores.values()) / len(t09_scores)
        best_name, best_score = max(t09_scores.items(), key=lambda item: item[1])
        print(f"[T09 + real_sirens] mean cosine = {avg_t:+.4f}")
        print(f"  highest similarity: {best_name} ({best_score:+.4f})")

    if c01_scores:
        avg_c = sum(c01_scores.values()) / len(c01_scores)
        best_name, best_score = max(c01_scores.items(), key=lambda item: item[1])
        print(f"[C01 + real_sirens] mean cosine = {avg_c:+.4f}")
        print(f"  highest similarity: {best_name} ({best_score:+.4f})")

    print()
    print("[interpretation] Higher score means higher cosine similarity to that siren.")
    print("[caveat] ViT-MAE and AST are independently pretrained; cosine is exploratory,")
    print("          not a calibrated cross-modal siren-presence probability.")


if __name__ == "__main__":
    main()
