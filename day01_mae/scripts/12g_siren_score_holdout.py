"""Siren score hold-out: T09(사이렌 없음) 검증.

3가지 시나리오:
  A. T09 + C01_with_siren (도메인 mismatch)
  B. T09 + synthetic siren  (합성)
  C. T09 untouched only     (baseline)

핵심: T09의 score가 0에 가까운지 확인.
"""
from __future__ import annotations

import sys
import json
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
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
SYNTH = Path("/Users/robotjang/spatial-avla-pipeline/exports/synthetic/siren")
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
    return out.pooler_output.squeeze(0)   # [768]


def cos_sim(v, a):
    return F.cosine_similarity(v.unsqueeze(0), a.unsqueeze(0), dim=-1).item()


def main():
    print(f"[device] {DEVICE}")
    ds = FrameMAEDataset(CSV, transform=val_tf)

    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

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

    # 클립별 평균
    def clip_vision_mean(clip):
        idx = [i for i, row in ds.df.iterrows() if row["clip"] == clip]
        v = feats_v_all[idx].mean(dim=0)
        return F.normalize(v, dim=-1)

    # ── Audio features (AST) ────────────────
    print("[model] loading AST...")
    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    # 각 클립의 audio 경로
    audio_paths = {}
    for clip, meta in metas.items():
        audio_paths[clip] = {}
        for key in ["untouched", "with_siren"]:
            rel = meta.get("audio", {}).get(key, None)
            if isinstance(rel, str):
                p = Path(meta["_meta_path"]).parent / rel
                if p.exists():
                    audio_paths[clip][key] = p

    # 오디오 feature 추출 (캐시)
    audio_feats = {}
    for clip, paths in audio_paths.items():
        audio_feats[clip] = {}
        for key, p in paths.items():
            print(f"  AST: {clip}/{key} ...", end=" ", flush=True)
            a = get_ast_feature(ast, p)
            audio_feats[clip][key] = F.normalize(a, dim=-1)
            print("done")

    # ── Siren score (베이스라인) ────────────
    print()
    print("=" * 65)
    print("BASELINE: 클립별 score (12f와 동일)")
    print("=" * 65)
    print(f"{'clip':>5s}  {'cos_u':>10s}  {'cos_s':>10s}  {'score':>10s}")
    print("-" * 45)

    baseline = {}
    for clip in ["C01", "C02", "C04"]:
        v = clip_vision_mean(clip)
        if "untouched" in audio_feats[clip] and "with_siren" in audio_feats[clip]:
            a_u = audio_feats[clip]["untouched"]
            a_s = audio_feats[clip]["with_siren"]
            cos_u = cos_sim(v, a_u)
            cos_s = cos_sim(v, a_s)
            score = cos_u - cos_s
            baseline[clip] = score
            print(f"{clip:>5s}  {cos_u:>10.4f}  {cos_s:>10.4f}  {score:>+10.4f}")

    # ── HOLD-OUT: T09 + C01_with_siren ──────
    print()
    print("=" * 65)
    print("HOLD-OUT A: T09 vision + C01/C02/C04 with_siren")
    print("  (도메인 mismatch: 해외 도로 vs 한국 도로+사이렌)")
    print("=" * 65)
    print(f"{'vision':>8s}  {'audio_siren':>14s}  {'cos_u':>10s}  {'cos_s':>10s}  {'score':>10s}")
    print("-" * 60)

    v_t09 = clip_vision_mean("T09")
    a_t09_u = audio_feats["T09"].get("untouched")

    for src_clip in ["C01", "C02", "C04"]:
        if "with_siren" not in audio_feats[src_clip]:
            continue
        a_s = audio_feats[src_clip]["with_siren"]
        cos_u = cos_sim(v_t09, a_t09_u) if a_t09_u is not None else 0.0
        cos_s = cos_sim(v_t09, a_s)
        score = cos_u - cos_s
        print(f"{'T09':>8s}  {src_clip:>14s}  {cos_u:>10.4f}  {cos_s:>10.4f}  {score:>+10.4f}")

    # ── HOLD-OUT B: T09 + synthetic siren ───
    print()
    print("=" * 65)
    print("HOLD-OUT B: T09 vision + synthetic siren")
    print("  (합성 siren wav로 검증)")
    print("=" * 65)

    syn_files = sorted(SYNTH.glob("siren_theta*.wav"))[:4]
    if syn_files:
        print(f"{'vision':>8s}  {'synthetic':>30s}  {'cos_u':>10s}  {'cos_s':>10s}  {'score':>10s}")
        print("-" * 75)
        for p in syn_files:
            a_s = F.normalize(get_ast_feature(ast, p), dim=-1)
            cos_u = cos_sim(v_t09, a_t09_u) if a_t09_u is not None else 0.0
            cos_s = cos_sim(v_t09, a_s)
            score = cos_u - cos_s
            print(f"{'T09':>8s}  {p.name[:30]:>30s}  {cos_u:>10.4f}  {cos_s:>10.4f}  {score:>+10.4f}")

    # ── HOLD-OUT C: T09 untouched only ──────
    print()
    print("=" * 65)
    print("HOLD-OUT C: T09 untouched only (baseline)")
    print("=" * 65)
    if a_t09_u is not None:
        cos_u = cos_sim(v_t09, a_t09_u)
        print(f"T09 + untouched: cos = {cos_u:.4f}  (비교용)")

    # ── 요약 ────────────────────────────────
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    if baseline:
        avg = sum(baseline.values()) / len(baseline)
        std = (sum((s - avg) ** 2 for s in baseline.values()) / len(baseline)) ** 0.5
        print(f"[baseline] C01/C02/C04: avg = {avg:+.4f}, std = {std:.4f}")
    print(f"[holdout] T09 + 다른 클립 with_siren: 위 표 참조")
    print(f"[판정] T09 score가 baseline과 유사하면 → 도메인 무관 siren 감지 ✅")
    print(f"[판정] T09 score가 0 근처면 → 도메인 차이가 큼 ❌")


if __name__ == "__main__":
    main()