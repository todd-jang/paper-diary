"""Cross-Modal Contrastive: ViTMAE (vision) + AST (audio).

log-mel 통계 대신 AST로 오디오 feature 추출.
InfoNCE loss로 클립별 vision-audio 정렬.
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from functools import lru_cache
from typing import Optional

MAE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAE_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ASTModel

from data.dataset import FrameMAEDataset
from data.transforms import val_tf

# ── 경로 설정 ──────────────────────────────────
CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── 설정 ──────────────────────────────────────
VIT_MAE_NAME = "facebook/vit-mae-base"
AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AUDIO_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TARGET_LEN = 1024   # AST는 1024 프레임 (약 10.24초) 표준
TEMPERATURE = 0.07


# ══════════════════════════════════════════════
# 1. clip_meta 로드
# ══════════════════════════════════════════════
def load_clip_metas() -> dict:
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d
    return metas


# ══════════════════════════════════════════════
# 2. 오디오 → log-mel (AST 입력)
# ══════════════════════════════════════════════
def load_logmel(path: str,
                sr: int = AUDIO_SR,
                n_mels: int = N_MELS,
                n_fft: int = N_FFT,
                hop: int = HOP,
                target_len: int = TARGET_LEN) -> torch.Tensor:
    """wav → log-mel (n_mels, target_len)."""
    data, orig_sr = sf.read(str(path), always_2d=True)   # (T, C)
    wav = torch.from_numpy(data.T).float()                # (C, T)

    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    # stereo → mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    mel = T.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop
    )(wav)                                                # (1, n_mels, T')
    log_mel = torch.log(mel + 1e-6)                       # (1, n_mels, T')

    # target_len으로 패딩/자르기
    T_cur = log_mel.shape[-1]
    if T_cur < target_len:
        pad = target_len - T_cur
        log_mel = F.pad(log_mel, (0, pad))
    else:
        log_mel = log_mel[..., :target_len]

    return log_mel.squeeze(0)   # (n_mels, target_len)


@lru_cache(maxsize=64)
def _audio_feature_cached(path: str) -> tuple:
    """오디오 파일 1개의 AST feature (캐시)."""
    ast_model = _get_ast()
    log_mel = load_logmel(path).unsqueeze(0)   # (1, n_mels, T)

    # AST는 (batch, time, freq) or (batch, freq, time) 형태
    # transformers ASTModel은 (batch, max_length, num_mel_bins) 요구
    # log_mel: (1, n_mels=128, T=1024) → (1, T, n_mels)
    x = log_mel.transpose(1, 2)   # (1, 1024, 128)

    # 표준화 (AST 논문 권장)
    x = (x - x.mean()) / (x.std() + 1e-6)

    with torch.no_grad():
        out = ast_model(x.to(DEVICE))

    feat = out.pooler_output   # (1, 768) — AST CLS
    return tuple(feat.squeeze(0).cpu().tolist())


# ══════════════════════════════════════════════
# 3. 모델 lazy loader
# ══════════════════════════════════════════════
_VIT_MAE = None
_AST = None


def _get_vit_mae():
    global _VIT_MAE
    if _VIT_MAE is None:
        print(f"[model] loading {VIT_MAE_NAME}...")
        _VIT_MAE = (
            ViTMAEForPreTraining
            .from_pretrained(VIT_MAE_NAME)
            .vit.eval()
            .to(DEVICE)
        )
    return _VIT_MAE


def _get_ast():
    global _AST
    if _AST is None:
        print(f"[model] loading {AST_NAME}...")
        _AST = ASTModel.from_pretrained(AST_NAME).eval().to(DEVICE)
    return _AST


# ══════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════
def main():
    print(f"[device] {DEVICE}")

    ds = FrameMAEDataset(CSV, transform=val_tf)
    metas = load_clip_metas()
    print(f"[data] {len(ds)} frames from {len(metas)} clips")

    # ── 1. Vision feature (ViTMAE) ──
    vit = _get_vit_mae()
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    feats_v = []
    with torch.no_grad():
        for imgs in dl:
            out = vit(pixel_values=imgs.to(DEVICE),
                      interpolate_pos_encoding=True)
            feats_v.append(out.last_hidden_state[:, 0, :].cpu())
    feats_v = torch.cat(feats_v, dim=0).float()   # [80, 768]
    print(f"[vision] feats = {feats_v.shape}")

    # ── 2. Audio feature (AST) ──
    audio_feats = {}
    for clip, meta in metas.items():
        audio = meta.get("audio", {})
        clip_dir = Path(meta["_meta_path"]).parent
        for key in ["untouched", "with_siren"]:
            rel = audio.get(key, None)
            if not isinstance(rel, str):
                continue
            p = clip_dir / rel
            if not p.exists():
                continue
            try:
                feat = _audio_feature_cached(str(p))
                audio_feats[(clip, key)] = torch.tensor(feat, dtype=torch.float32)
            except Exception as e:
                print(f"[warn] AST failed for {p}: {e}")

    print(f"[audio] AST features = {len(audio_feats)}")
    for k in audio_feats:
        print(f"  {k}: dim={audio_feats[k].shape[0]}")

    # ── 3. InfoNCE (frame ↔ audio_with_siren) ──
    valid_clips = ["C01", "C02", "C04"]   # with_siren 있는 클립

    # 프레임별 feature를 클립 평균으로
    clip_to_frames = {}
    for i, row in ds.df.iterrows():
        if row["clip"] in valid_clips:
            clip_to_frames.setdefault(row["clip"], []).append(i)

    # 투영 (오디오 768 → vision 768 그대로 or proj)
    proj = nn.Identity()   # 둘 다 768이면 불필요

    V = torch.stack([
        feats_v[clip_to_frames[c]].mean(dim=0).to(DEVICE)
        for c in valid_clips
    ])
    A = torch.stack([
        proj(audio_feats[(c, "with_siren")].to(DEVICE))
        for c in valid_clips
    ])

    V = F.normalize(V, dim=-1)
    A = F.normalize(A, dim=-1)

    logits = V @ A.T / TEMPERATURE
    labels = torch.arange(len(valid_clips), device=DEVICE)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(-1) == labels).float().mean()

    print()
    print(f"[infonce] logits = \n{logits.detach().cpu()}")
    print(f"[infonce] loss = {loss.item():.4f}")
    print(f"[infonce] acc = {acc.item():.3f}")


if __name__ == "__main__":
    main()