"""siren_template / road_template 오프라인 계산 + 저장.

한 번 실행 -> configs/siren_template.pt 생성
이후 추론 시 로드만.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

MAE_ROOT = Path("/Users/robotjang/paper-diary/day01_mae")
sys.path.insert(0, str(MAE_ROOT))

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from transformers import ASTModel

# -- 경로 --
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
REAL_SIRENS = Path("/Users/robotjang/spatial-avla-pipeline/exports/real_sirens")
OUT_DIR = Path("/Users/robotjang/spatial-avla-pipeline/configs")
OUT_PATH = OUT_DIR / "siren_template.pt"

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
    mel = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels,
                           n_fft=n_fft, hop_length=hop)(wav)
    lm = torch.log(mel + 1e-6).squeeze(0)
    Tc = lm.shape[-1]
    if Tc < target_len:
        lm = F.pad(lm, (0, target_len - Tc))
    else:
        lm = lm[..., :target_len]
    return lm


def get_ast_feature(ast, path):
    lm = load_logmel(path).unsqueeze(0)
    x = lm.transpose(1, 2)
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        out = ast(x.to(DEVICE))
    return F.normalize(out.pooler_output.squeeze(0), dim=-1)


def main():
    print(f"[device] {DEVICE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    # -- siren template --
    siren_sources = [
        "G44-01-Sirens",
        "G44-02-Air Raid Siren",
        "G44-03-Whoopee Whistle Siren",
    ]
    siren_feats = []
    print("\n[siren_template]")
    for name in siren_sources:
        p = REAL_SIRENS / f"{name}_16k.wav"
        if p.exists():
            f = get_ast_feature(ast, p)
            siren_feats.append(f)
            print(f"  {name}")
    siren_template = F.normalize(torch.stack(siren_feats).mean(0), dim=-1)

    # -- road template --
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

    road_sources = ["C01", "C02"]
    road_feats = []
    print("\n[road_template]")
    for clip in road_sources:
        rel = metas[clip]["audio"]["untouched"]
        p = Path(metas[clip]["_meta_path"]).parent / rel
        if p.exists():
            f = get_ast_feature(ast, p)
            road_feats.append(f)
            print(f"  {clip}/untouched")
    road_template = F.normalize(torch.stack(road_feats).mean(0), dim=-1)

    # -- 저장 --
    torch.save({
        "siren_template": siren_template.cpu(),
        "road_template": road_template.cpu(),
        "metadata": {
            "siren_sources": siren_sources,
            "road_sources": road_sources,
            "ast_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "audio_sr": AUDIO_SR,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop": HOP,
            "target_len": TARGET_LEN,
            "window_sec": 2.0,
            "hop_sec": 1.0,
            "threshold": -0.04,
        },
    }, OUT_PATH)

    print(f"\n[저장] {OUT_PATH}")
    print(f"  siren_template: {siren_template.shape}")
    print(f"  road_template:  {road_template.shape}")


if __name__ == "__main__":
    main()
