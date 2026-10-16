"""윈도우 크기별 프레임 단위 siren score 비교.

12l에서 발견한 문제:
  window=0.5초 슬라이스 → 96% 패딩 → feature 왜곡 → 오탐

12m:
  window = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]초
  → 각 윈도우별 프레임 score 분포
  → 최적 윈도우 결정

TARGET_LEN도 윈도우에 맞춰 조정:
  target_len = min(1024, 실제 log-mel 프레임 수)
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

# ── 실험 윈도우 크기 ─────────────────────────
WINDOWS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
HOP_RATIO = 0.5   # hop = window * 0.5

# ── 템플릿 ────────────────────────────────────
SIREN_TEMPLATE_SOURCES = [
    "G44-01-Sirens",
    "G44-02-Air Raid Siren",
    "G44-03-Whoopee Whistle Siren",
]
ROAD_TEMPLATE_SOURCES = ["C01", "C02"]


def load_logmel_from_array(wav: np.ndarray, sr: int,
                            n_mels=N_MELS, n_fft=N_FFT,
                            hop=HOP, target_len=None):
    """np.array wav → log-mel. target_len=None이면 실제 길이."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    if sr != AUDIO_SR:
        wav_t = torchaudio.functional.resample(wav_t, sr, AUDIO_SR)
    mel = T.MelSpectrogram(
        sample_rate=AUDIO_SR, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop
    )(wav_t)
    lm = torch.log(mel + 1e-6).squeeze(0)   # [n_mels, T']

    if target_len is not None:
        Tc = lm.shape[-1]
        if Tc < target_len:
            lm = F.pad(lm, (0, target_len - Tc))
        else:
            lm = lm[..., :target_len]
    return lm


def get_ast_feature_from_array(ast, wav: np.ndarray, sr: int, target_len=None):
    """np.array wav → AST CLS feature."""
    lm = load_logmel_from_array(wav, sr, target_len=target_len).unsqueeze(0)
    x = lm.transpose(1, 2)   # [1, T, n_mels]
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        out = ast(x.to(DEVICE))
    return F.normalize(out.pooler_output.squeeze(0), dim=-1)


def get_ast_feature_from_path(ast, path, target_len=None):
    wav, sr = sf.read(str(path), always_2d=True)
    return get_ast_feature_from_array(ast, wav, sr, target_len=target_len)


def slice_frames(wav: np.ndarray, sr: int, window_sec: float, hop_sec: float):
    """wav → 프레임 슬라이스 리스트."""
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    slices = []
    for start in range(0, len(wav) - win_len + 1, max(1, hop_len)):
        slices.append(wav[start:start + win_len])
    return slices


def main():
    print(f"[device] {DEVICE}")
    print(f"[config] windows = {WINDOWS}")
    print(f"[config] hop_ratio = {HOP_RATIO}")

    # ── 1. AST + 템플릿 ─────────────────────
    print("\n[1] AST 로드 + 템플릿 학습")
    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    # 템플릿은 전체 오디오 기준 (TARGET_LEN=1024)
    siren_feats = []
    for name in SIREN_TEMPLATE_SOURCES:
        p = REAL_SIRENS / f"{name}_16k.wav"
        if p.exists():
            siren_feats.append(get_ast_feature_from_path(ast, p, target_len=1024))
    siren_template = F.normalize(torch.stack(siren_feats).mean(0), dim=-1)

    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

    road_feats = []
    for clip in ROAD_TEMPLATE_SOURCES:
        rel = metas[clip]["audio"]["untouched"]
        p = Path(metas[clip]["_meta_path"]).parent / rel
        road_feats.append(get_ast_feature_from_path(ast, p, target_len=1024))
    road_template = F.normalize(torch.stack(road_feats).mean(0), dim=-1)
    print(f"  siren_template: {len(siren_feats)}, road_template: {len(road_feats)}")

    def score(feat):
        return ((feat @ siren_template) - (feat @ road_template)).item()

    # ── 2. 윈도우별 프레임 score ────────────
    print("\n[2] 윈도우별 프레임 score")
    all_results = {}   # {(window, clip, key): scores}

    for window_sec in WINDOWS:
        hop_sec = window_sec * HOP_RATIO
        # 윈도우 크기에 맞춰 target_len 결정
        target_len = 1024

        print(f"\n  [window={window_sec}s, hop={hop_sec}s, target_len={target_len}]")

        for clip in ["C01", "C02", "C04", "T09"]:
            for key in ["untouched", "with_siren"]:
                rel = metas[clip]["audio"].get(key)
                if not isinstance(rel, str):
                    continue
                p = Path(metas[clip]["_meta_path"]).parent / rel
                if not p.exists():
                    continue

                wav, sr = sf.read(str(p), always_2d=True)
                slices = slice_frames(wav, sr, window_sec, hop_sec)
                if not slices:
                    continue

                scores = []
                for s in slices:
                    feat = get_ast_feature_from_array(ast, s, sr, target_len=target_len)
                    scores.append(score(feat))

                scores = np.array(scores)
                all_results[(window_sec, clip, key)] = scores

                print(f"    {clip}/{key}: N={len(scores)}, "
                      f"mean={scores.mean():+.4f}, "
                      f"std={scores.std():.4f}, "
                      f"min={scores.min():+.4f}, "
                      f"max={scores.max():+.4f}")

    # ── 3. 윈도우별 분리도 비교 ─────────────
    print("\n" + "=" * 70)
    print("[3] 윈도우별 분리도")
    print("=" * 70)
    print(f"{'window':>8s}  {'pos_mean':>10s}  {'neg_mean':>10s}  "
          f"{'gap':>8s}  {'pos_min':>10s}  {'neg_max':>10s}  {'overlap':>10s}")
    print("-" * 80)

    for window_sec in WINDOWS:
        pos_scores = []
        neg_scores = []
        for clip in ["C01", "C02", "C04", "T09"]:
            for key in ["untouched", "with_siren"]:
                if (window_sec, clip, key) not in all_results:
                    continue
                s = all_results[(window_sec, clip, key)]
                if key == "with_siren":
                    pos_scores.extend(s)
                else:
                    neg_scores.extend(s)

        if not pos_scores or not neg_scores:
            continue

        pos_scores = np.array(pos_scores)
        neg_scores = np.array(neg_scores)

        pos_mean = pos_scores.mean()
        neg_mean = neg_scores.mean()
        gap = pos_mean - neg_mean
        pos_min = pos_scores.min()
        neg_max = neg_scores.max()
        overlap = pos_min - neg_max   # 양수면 겹침 없음

        print(f"{window_sec:>8.1f}  {pos_mean:>+10.4f}  {neg_mean:>+10.4f}  "
              f"{gap:>+8.4f}  {pos_min:>+10.4f}  {neg_max:>+10.4f}  "
              f"{overlap:>+10.4f}")

    # ── 4. 임계값별 성능 (각 윈도우) ────────
    print("\n" + "=" * 70)
    print("[4] 임계값별 성능")
    print("=" * 70)

    for window_sec in WINDOWS:
        pos_scores = []
        neg_scores = []
        for clip in ["C01", "C02", "C04", "T09"]:
            for key in ["untouched", "with_siren"]:
                if (window_sec, clip, key) not in all_results:
                    continue
                s = all_results[(window_sec, clip, key)]
                if key == "with_siren":
                    pos_scores.extend(s)
                else:
                    neg_scores.extend(s)

        if not pos_scores or not neg_scores:
            continue

        pos_scores = np.array(pos_scores)
        neg_scores = np.array(neg_scores)
        all_s = np.concatenate([pos_scores, neg_scores])
        all_y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

        # 최적 임계값 (Youden)
        thresholds = np.linspace(all_s.min(), all_s.max(), 50)
        best_J = -1
        best_thr = 0
        for thr in thresholds:
            pred = (all_s > thr).astype(int)
            tp = ((pred == 1) & (all_y == 1)).sum()
            fp = ((pred == 1) & (all_y == 0)).sum()
            fn = ((pred == 0) & (all_y == 1)).sum()
            tn = ((pred == 0) & (all_y == 0)).sum()
            tpr = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            J = tpr - fpr
            if J > best_J:
                best_J = J
                best_thr = thr

        # 최적 임계값에서 성능
        pred = (all_s > best_thr).astype(int)
        tp = ((pred == 1) & (all_y == 1)).sum()
        fp = ((pred == 1) & (all_y == 0)).sum()
        fn = ((pred == 0) & (all_y == 1)).sum()
        tn = ((pred == 0) & (all_y == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)

        # AUC (수동)
        order = np.argsort(all_s)
        y_sorted = all_y[order]
        n_pos = all_y.sum()
        n_neg = len(all_y) - n_pos
        rank_sum = np.where(y_sorted == 1)[0].sum()
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-9)

        print(f"\n[window={window_sec}s]")
        print(f"  최적 임계값 = {best_thr:+.4f}")
        print(f"  TPR = {tpr:.3f}, FPR = {fpr:.3f}")
        print(f"  AUC = {auc:.4f}")
        print(f"  N_pos = {n_pos}, N_neg = {n_neg}")

    # ── 5. 요약 저장 ────────────────────────
    out_dir = MAE_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # npz
    npz_data = {f"{w}_{c}_{k}": v for (w, c, k), v in all_results.items()}
    np.savez(out_dir / "12m_window_ablation.npz", **npz_data)

    # 요약 텍스트
    summary_path = out_dir / "12m_window_ablation.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("12m: 윈도우 크기별 프레임 단위 siren score\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"WINDOWS = {WINDOWS}\n")
        f.write(f"HOP_RATIO = {HOP_RATIO}\n\n")

        for window_sec in WINDOWS:
            f.write(f"\n[window={window_sec}s]\n")
            for clip in ["C01", "C02", "C04", "T09"]:
                for key in ["untouched", "with_siren"]:
                    if (window_sec, clip, key) not in all_results:
                        continue
                    s = all_results[(window_sec, clip, key)]
                    f.write(f"  {clip}/{key}: N={len(s)} "
                            f"mean={s.mean():+.4f} std={s.std():.4f}\n")

    print(f"\n[저장] {summary_path}")
    print(f"[저장] {out_dir / '12m_window_ablation.npz'}")

    # ── 6. 최적 윈도우 추천 ─────────────────
    print("\n" + "=" * 70)
    print("[5] 최적 윈도우 추천")
    print("=" * 70)
    print("기준: pos_min > neg_max (겹침 없음) + gap 최대")

    for window_sec in WINDOWS:
        pos_scores = []
        neg_scores = []
        for clip in ["C01", "C02", "C04", "T09"]:
            for key in ["untouched", "with_siren"]:
                if (window_sec, clip, key) not in all_results:
                    continue
                s = all_results[(window_sec, clip, key)]
                if key == "with_siren":
                    pos_scores.extend(s)
                else:
                    neg_scores.extend(s)

        if not pos_scores or not neg_scores:
            continue

        pos_min = np.min(pos_scores)
        neg_max = np.max(neg_scores)
        overlap = pos_min - neg_max
        gap = np.mean(pos_scores) - np.mean(neg_scores)

        verdict = "✅ 분리" if overlap > 0 else "⚠️ 겹침"
        print(f"  window={window_sec:>5.1f}s: gap={gap:+.4f}, "
              f"overlap={overlap:+.4f} {verdict}")


if __name__ == "__main__":
    main()
