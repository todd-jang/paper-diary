"""프레임 단위 siren score 계산.

클립 오디오를 프레임에 맞춰 슬라이싱하여
프레임별 siren score 시계열 생성.

목적:
  - spatial-avla 프레임 단위 추론 준비
  - 클립 단위 vs 프레임 단위 비교
  - 시계열 패턴 확인 (siren 접근/이탈)
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
from transformers import ASTModel

# ── 설정 ──────────────────────────────────────
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
REAL_SIRENS = Path("/Users/robotjang/spatial-avla-pipeline/exports/real_sirens")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

AUDIO_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TARGET_LEN = 1024

# ── 프레임 슬라이싱 파라미터 ─────────────────
FPS = 30                # 가정
WINDOW_SEC = 1.0        # 1초 윈도우
HOP_SEC = 0.5           # 0.5초 hop

# ── 템플릿 (12j/12k와 동일) ──────────────────
SIREN_TEMPLATE_SOURCES = [
    "G44-01-Sirens",
    "G44-02-Air Raid Siren",
    "G44-03-Whoopee Whistle Siren",
]
ROAD_TEMPLATE_SOURCES = ["C01", "C02"]


def load_logmel_from_array(wav: np.ndarray, sr: int,
                            n_mels=N_MELS, n_fft=N_FFT,
                            hop=HOP, target_len=TARGET_LEN):
    """np.array wav → log-mel."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)   # [1, T]
    if sr != AUDIO_SR:
        wav_t = torchaudio.functional.resample(wav_t, sr, AUDIO_SR)
    mel = T.MelSpectrogram(
        sample_rate=AUDIO_SR, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop
    )(wav_t)
    lm = torch.log(mel + 1e-6).squeeze(0)   # [n_mels, T']
    Tc = lm.shape[-1]
    if Tc < target_len:
        lm = F.pad(lm, (0, target_len - Tc))
    else:
        lm = lm[..., :target_len]
    return lm


def get_ast_feature_from_array(ast, wav: np.ndarray, sr: int):
    """np.array wav → AST CLS feature."""
    lm = load_logmel_from_array(wav, sr).unsqueeze(0)   # [1, n_mels, T]
    x = lm.transpose(1, 2)                               # [1, T, n_mels]
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        out = ast(x.to(DEVICE))
    return F.normalize(out.pooler_output.squeeze(0), dim=-1)


def get_ast_feature_from_path(ast, path):
    wav, sr = sf.read(str(path), always_2d=True)
    return get_ast_feature_from_array(ast, wav, sr)


def slice_frames(wav: np.ndarray, sr: int,
                 window_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    """wav → 프레임별 슬라이스 리스트."""
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    slices = []
    for start in range(0, len(wav) - win_len + 1, hop_len):
        slices.append(wav[start:start + win_len])
    return slices


def main():
    print(f"[device] {DEVICE}")
    print(f"[config] FPS={FPS}, window={WINDOW_SEC}s, hop={HOP_SEC}s")

    # ── 1. AST 로드 + 템플릿 ────────────────
    print("\n[1] AST 로드 + 템플릿 학습")
    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    siren_feats = []
    for name in SIREN_TEMPLATE_SOURCES:
        p = REAL_SIRENS / f"{name}_16k.wav"
        if p.exists():
            siren_feats.append(get_ast_feature_from_path(ast, p))
    siren_template = F.normalize(torch.stack(siren_feats).mean(0), dim=-1)
    print(f"  siren_template: {len(siren_feats)} files")

    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

    road_feats = []
    for clip in ROAD_TEMPLATE_SOURCES:
        rel = metas[clip]["audio"]["untouched"]
        p = Path(metas[clip]["_meta_path"]).parent / rel
        road_feats.append(get_ast_feature_from_path(ast, p))
    road_template = F.normalize(torch.stack(road_feats).mean(0), dim=-1)
    print(f"  road_template: {len(road_feats)} files")

    def score(feat):
        return ((feat @ siren_template) - (feat @ road_template)).item()

    # ── 2. 프레임 단위 score 계산 ────────────
    print("\n[2] 프레임 단위 score 계산")

    results = {}
    for clip in ["C01", "C02", "C04", "T09"]:
        for key in ["untouched", "with_siren"]:
            rel = metas[clip]["audio"].get(key)
            if not isinstance(rel, str):
                continue
            p = Path(metas[clip]["_meta_path"]).parent / rel
            if not p.exists():
                continue

            wav, sr = sf.read(str(p), always_2d=True)
            slices = slice_frames(wav, sr)
            if not slices:
                continue

            scores = []
            for i, s in enumerate(slices):
                feat = get_ast_feature_from_array(ast, s, sr)
                scores.append(score(feat))

            scores = np.array(scores)
            results[(clip, key)] = scores
            print(f"  {clip}/{key}: N={len(scores)}, "
                  f"mean={scores.mean():+.4f}, "
                  f"std={scores.std():.4f}, "
                  f"min={scores.min():+.4f}, "
                  f"max={scores.max():+.4f}")

    # ── 3. 클립 단위 vs 프레임 단위 비교 ─────
    print("\n" + "=" * 70)
    print("[3] 클립 단위 vs 프레임 단위 비교")
    print("=" * 70)
    print(f"{'clip/audio':>25s}  {'clip_score':>12s}  {'frame_mean':>12s}  "
          f"{'frame_std':>12s}")
    print("-" * 70)

    for clip in ["C01", "C02", "C04", "T09"]:
        for key in ["untouched", "with_siren"]:
            if (clip, key) not in results:
                continue
            scores = results[(clip, key)]

            # 클립 단위 (전체 오디오)
            rel = metas[clip]["audio"][key]
            p = Path(metas[clip]["_meta_path"]).parent / rel
            clip_feat = get_ast_feature_from_path(ast, p)
            clip_score = score(clip_feat)

            print(f"{clip+'/'+key:>25s}  {clip_score:>+12.4f}  "
                  f"{scores.mean():>+12.4f}  {scores.std():>12.4f}")

    # ── 4. 시계열 패턴 (with_siren) ──────────
    print("\n" + "=" * 70)
    print("[4] with_siren 시계열 (프레임별 score)")
    print("=" * 70)

    for clip in ["C01", "C02", "C04"]:
        if (clip, "with_siren") not in results:
            continue
        scores = results[(clip, "with_siren")]
        print(f"\n[{clip}/with_siren] N={len(scores)}")
        # 앞 10개, 뒤 10개
        print(f"  앞 10: {[f'{s:+.3f}' for s in scores[:10]]}")
        print(f"  뒤 10: {[f'{s:+.3f}' for s in scores[-10:]]}")

        # 임계값 판정
        n_detected = (scores > -0.20).sum()
        print(f"  detected (>-0.20): {n_detected}/{len(scores)} "
              f"({100*n_detected/len(scores):.1f}%)")

    # ── 5. untouched 시계열 ──────────────────
    print("\n" + "=" * 70)
    print("[5] untouched 시계열 (프레임별 score)")
    print("=" * 70)

    for clip in ["C01", "C02", "C04", "T09"]:
        if (clip, "untouched") not in results:
            continue
        scores = results[(clip, "untouched")]
        print(f"\n[{clip}/untouched] N={len(scores)}")
        print(f"  mean={scores.mean():+.4f}, std={scores.std():.4f}")

        n_detected = (scores > -0.20).sum()
        print(f"  detected (>-0.20): {n_detected}/{len(scores)} "
              f"({100*n_detected/len(scores):.1f}%)  "
              f"{'⚠️ 오탐' if n_detected > 0 else '✅ 정상'}")

    # ── 6. 요약 ──────────────────────────────
    print("\n" + "=" * 70)
    print("[6] 요약")
    print("=" * 70)

    print("\n[임계값 -0.20 기준]")
    print(f"{'clip/audio':>25s}  {'detected':>10s}  {'ratio':>8s}")
    print("-" * 50)

    for clip in ["C01", "C02", "C04", "T09"]:
        for key in ["untouched", "with_siren"]:
            if (clip, key) not in results:
                continue
            scores = results[(clip, key)]
            n = (scores > -0.20).sum()
            ratio = 100 * n / len(scores)
            print(f"{clip+'/'+key:>25s}  {n:>4d}/{len(scores):<5d}  {ratio:>7.1f}%")

    # ── 7. 결과 저장 ─────────────────────────
    out_dir = MAE_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # 요약 저장
    summary_path = out_dir / "12l_frame_level_siren.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("12l: 프레임 단위 siren score\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"FPS={FPS}, window={WINDOW_SEC}s, hop={HOP_SEC}s\n\n")

        for clip in ["C01", "C02", "C04", "T09"]:
            for key in ["untouched", "with_siren"]:
                if (clip, key) not in results:
                    continue
                scores = results[(clip, key)]
                n = (scores > -0.20).sum()
                f.write(f"[{clip}/{key}] N={len(scores)} "
                        f"mean={scores.mean():+.4f} "
                        f"std={scores.std():.4f} "
                        f"detected={n}/{len(scores)}\n")

    # 시계열 저장 (npy)
    ts_path = out_dir / "12l_frame_scores.npz"
    np.savez(ts_path, **{f"{c}_{k}": v for (c, k), v in results.items()})

    print(f"\n[저장] {summary_path}")
    print(f"[저장] {ts_path}")


if __name__ == "__main__":
    main()
