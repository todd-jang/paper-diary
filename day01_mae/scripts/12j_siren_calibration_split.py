"""Siren score: 보정/검증/테스트 분할 검증."""
from __future__ import annotations

import sys, json
from pathlib import Path
from itertools import product

MAE_ROOT = Path("/Users/robotjang/paper-diary/day01_mae")
sys.path.insert(0, str(MAE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from transformers import ViTMAEForPreTraining, ASTModel

# ── 분할 정의 ─────────────────────────────────
SIREN_TRAIN = ["G44-01-Sirens", "G44-02-Air Raid Siren", "G44-03-Whoopee Whistle Siren"]
SIREN_VAL   = ["G44-04-Ambulance Siren"]
SIREN_TEST  = ["G44-05-Warbling Boat Siren", "G44-07-Emergency Siren"]

ROAD_TRAIN  = ["C01", "C02"]      # untouched
ROAD_VAL    = ["C04"]
ROAD_TEST   = ["T09"]

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
    mel = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels,
                           n_fft=n_fft, hop_length=hop)(wav)
    lm = torch.log(mel + 1e-6).squeeze(0)
    Tc = lm.shape[-1]
    lm = F.pad(lm, (0, target_len - Tc)) if Tc < target_len else lm[..., :target_len]
    return lm


def get_ast_feature(ast, path):
    lm = load_logmel(str(path)).unsqueeze(0)
    x = lm.transpose(1, 2)
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        out = ast(x.to(DEVICE))
    return F.normalize(out.pooler_output.squeeze(0), dim=-1)


def main():
    print(f"[device] {DEVICE}")
    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    # ── 템플릿 학습 (TRAIN) ────────────────────
    print("\n[TRAIN] siren template...")
    siren_train_feats = []
    for name in SIREN_TRAIN:
        # 파일 찾기
        for suffix in ["_16k.wav", "_stereo_16k.wav"]:
            p = REAL_SIRENS / f"{name}{suffix}"
            if p.exists():
                f = get_ast_feature(ast, p)
                siren_train_feats.append(f)
                print(f"  {p.stem}")
                break
    siren_template = F.normalize(torch.stack(siren_train_feats).mean(0), dim=-1)

    print("\n[TRAIN] road template...")
    road_train_feats = []
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

    for clip in ROAD_TRAIN:
        rel = metas[clip]["audio"].get("untouched")
        p = Path(metas[clip]["_meta_path"]).parent / rel
        f = get_ast_feature(ast, p)
        road_train_feats.append(f)
        print(f"  {clip}/untouched")
    road_template = F.normalize(torch.stack(road_train_feats).mean(0), dim=-1)

    # ── 테스트 함수 ────────────────────────────
    def score(path):
        f = get_ast_feature(ast, path)
        return ((f @ siren_template) - (f @ road_template)).item()

    # ── VAL 평가 ───────────────────────────────
    print("\n" + "=" * 65)
    print("[VAL] 검증: G44-04 (학습에 없는 siren)")
    print("=" * 65)
    for suffix in ["_16k.wav", "_stereo_16k.wav"]:
        p = REAL_SIRENS / f"G44-04-Ambulance Siren{suffix}"
        if p.exists():
            print(f"  {p.stem}: score = {score(p):+.4f}")

    print("\n[VAL] 검증: C04 (학습에 없는 road)")
    for key in ["untouched", "with_siren"]:
        rel = metas["C04"]["audio"].get(key)
        if isinstance(rel, str):
            p = Path(metas["C04"]["_meta_path"]).parent / rel
            print(f"  C04/{key}: score = {score(p):+.4f}")

    # ── TEST 평가 (독립) ───────────────────────
    print("\n" + "=" * 65)
    print("[TEST] 독립: G44-05, G44-07 (완전 새로운 siren)")
    print("=" * 65)
    for name in SIREN_TEST:
        for suffix in ["_16k.wav", "_stereo_16k.wav"]:
            p = REAL_SIRENS / f"{name}{suffix}"
            if p.exists():
                print(f"  {p.stem[:45]:45s}: {score(p):+.4f}")

    print("\n[TEST] 독립: T09 (완전 새로운 road)")
    for key in ["untouched", "with_siren"]:
        rel = metas["T09"]["audio"].get(key)
        if isinstance(rel, str):
            p = Path(metas["T09"]["_meta_path"]).parent / rel
            print(f"  T09/{key}: score = {score(p):+.4f}")

    # ── 합성 검증 (mix) ────────────────────────
    print("\n" + "=" * 65)
    print("[SYNTH] 합성 검증: T09 untouched + real_sirens")
    print("=" * 65)
    import numpy as np
    road_data, sr = sf.read(str(Path(metas["T09"]["_meta_path"]).parent /
                                metas["T09"]["audio"]["untouched"]))
    if road_data.ndim > 1:
        road_data = road_data.mean(axis=1)
    for name in SIREN_TEST:
        p = REAL_SIRENS / f"{name}_16k.wav"
        if not p.exists():
            continue
        siren_data, sr2 = sf.read(str(p))
        if siren_data.ndim > 1:
            siren_data = siren_data.mean(axis=1)
        # 길이 맞추기
        L = min(len(road_data), len(siren_data))
        mix = 0.7 * road_data[:L] + 0.3 * siren_data[:L]
        tmp = Path("/tmp/_mix.wav")
        sf.write(str(tmp), mix, sr)
        print(f"  T09 + {name}: score = {score(tmp):+.4f}")


if __name__ == "__main__":
    main()