"""Siren score 재정의: 사이렌 템플릿 vs 도로 소음 템플릿.

현재 문제:
  score = cos(v, untouched) - cos(v, with_siren)
  → "도로 소음 vs 도로+사이렌" 차이 = 도메인 의존적
  → real_sirens(순수 사이렌)는 부호 반전

새로운 정의:
  siren_template = mean(real_sirens)
  road_template  = mean(clip untouched)
  score = cos(input_audio, siren_template) - cos(input_audio, road_template)

이렇게 하면:
  - 입력이 "도로+사이렌" → siren_template에 가까움 → score +
  - 입력이 "순수 사이렌"  → siren_template에 매우 가까움 → score ++
  - 입력이 "도로 소음만"  → road_template에 가까움 → score -
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
REAL_SIRENS = Path("/Users/robotjang/spatial-avla-pipeline/exports/real_sirens")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

AUDIO_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TARGET_LEN = 1024


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

    def clip_vision_mean(clip):
        idx = [i for i, row in ds.df.iterrows() if row["clip"] == clip]
        v = feats_v_all[idx].mean(dim=0)
        return F.normalize(v, dim=-1)

    # ── AST 로드 ────────────────────────────
    print("[model] loading AST...")
    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    # ── 템플릿 구성 ─────────────────────────
    # 1. 사이렌 템플릿 (real_sirens 평균)
    print("\n[template] building siren template...")
    real_siren_files = sorted(REAL_SIRENS.glob("G44-*_16k.wav"))
    siren_feats = []
    for p in real_siren_files:
        f = F.normalize(get_ast_feature(ast, p), dim=-1)
        siren_feats.append(f)
        print(f"  {p.stem}: loaded")
    siren_template = F.normalize(torch.stack(siren_feats).mean(dim=0), dim=-1)
    print(f"[template] siren_template built from {len(siren_feats)} files")

    # 2. 도로 소음 템플릿 (클립 untouched 평균)
    print("\n[template] building road template...")
    road_feats = []
    for clip in ["C01", "C02", "C04", "T09"]:
        rel = metas[clip].get("audio", {}).get("untouched", None)
        if not isinstance(rel, str):
            continue
        p = Path(metas[clip]["_meta_path"]).parent / rel
        if not p.exists():
            continue
        f = F.normalize(get_ast_feature(ast, p), dim=-1)
        road_feats.append(f)
        print(f"  {clip}/untouched: loaded")
    road_template = F.normalize(torch.stack(road_feats).mean(dim=0), dim=-1)
    print(f"[template] road_template built from {len(road_feats)} files")

    # ── Score 재정의 ────────────────────────
    print()
    print("=" * 70)
    print("NEW SCORE: cos(audio, siren_template) - cos(audio, road_template)")
    print("=" * 70)

    def new_score(audio_feat):
        cos_siren = cos_sim(audio_feat, siren_template)
        cos_road = cos_sim(audio_feat, road_template)
        return cos_siren - cos_road, cos_siren, cos_road

    # 1. 클립 오디오 (untouched, with_siren)
    print()
    print("[1] 클립 오디오 분류")
    print(f"{'clip':>5s}  {'audio':>14s}  {'cos_siren':>10s}  {'cos_road':>10s}  {'score':>10s}")
    print("-" * 60)

    for clip in ["C01", "C02", "C04", "T09"]:
        for key in ["untouched", "with_siren"]:
            rel = metas[clip].get("audio", {}).get(key, None)
            if not isinstance(rel, str):
                continue
            p = Path(metas[clip]["_meta_path"]).parent / rel
            if not p.exists():
                continue
            f = F.normalize(get_ast_feature(ast, p), dim=-1)
            s, cs, cr = new_score(f)
            print(f"{clip:>5s}  {key:>14s}  {cs:>10.4f}  {cr:>10.4f}  {s:>+10.4f}")

    # 2. real_sirens 분류
    print()
    print("[2] real_sirens 분류")
    print(f"{'file':>35s}  {'cos_siren':>10s}  {'cos_road':>10s}  {'score':>10s}")
    print("-" * 70)
    for p in real_siren_files:
        f = F.normalize(get_ast_feature(ast, p), dim=-1)
        s, cs, cr = new_score(f)
        print(f"{p.stem[:35]:>35s}  {cs:>10.4f}  {cr:>10.4f}  {s:>+10.4f}")

    # 3. 합성 siren 분류
    SYNTH = Path("/Users/robotjang/spatial-avla-pipeline/exports/synthetic/siren")
    syn_files = sorted(SYNTH.glob("siren_theta*.wav"))[:5]
    if syn_files:
        print()
        print("[3] synthetic siren 분류")
        print(f"{'file':>35s}  {'cos_siren':>10s}  {'cos_road':>10s}  {'score':>10s}")
        print("-" * 70)
        for p in syn_files:
            f = F.normalize(get_ast_feature(ast, p), dim=-1)
            s, cs, cr = new_score(f)
            print(f"{p.stem[:35]:>35s}  {cs:>10.4f}  {cr:>10.4f}  {s:>+10.4f}")

    # ── 요약 ────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("판정 기준:")
    print("  - 'with_siren' score > 0  → 사이렌 감지 성공")
    print("  - 'untouched'  score < 0  → 정상 판정 성공")
    print("  - 'real_sirens' score > 0 → 사이렌 템플릿 유효")


if __name__ == "__main__":
    main()
