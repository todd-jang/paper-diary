"""ROC 분석: siren score 임계값 결정.

데이터:
  양성 (siren=1):
    - real_sirens 6종 (12 파일)
    - 합성: T09_untouched + real_sirens
    - C01/C02/C04 with_siren
  음성 (siren=0):
    - C01/C02/C04/T09 untouched
    - 합성: T09_untouched + 다른 도로 소음

분석:
  1. 모든 score 수집
  2. ROC curve + AUC
  3. Precision-Recall curve
  4. 운영점 3종 결정 (보수/균형/민감)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import defaultdict

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
SYNTH = Path("/Users/robotjang/spatial-avla-pipeline/exports/synthetic/siren")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

AUDIO_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TARGET_LEN = 1024

# ── 템플릿 학습 데이터 (12j와 동일) ───────────
SIREN_TEMPLATE_SOURCES = ["G44-01-Sirens", "G44-02-Air Raid Siren", "G44-03-Whoopee Whistle Siren"]
ROAD_TEMPLATE_SOURCES = ["C01", "C02"]   # untouched

# ── 평가 데이터 ───────────────────────────────
SIREN_POSITIVE = [
    # real_sirens (학습에 없는 것 포함)
    ("G44-01-Sirens_16k", 1),
    ("G44-02-Air Raid Siren_16k", 1),
    ("G44-03-Whoopee Whistle Siren_16k", 1),
    ("G44-04-Ambulance Siren_16k", 1),
    ("G44-05-Warbling Boat Siren_16k", 1),
    ("G44-07-Emergency Siren_16k", 1),
]

ROAD_NEGATIVE = [
    # 클립 untouched
    ("C01", "untouched", 0),
    ("C02", "untouched", 0),
    ("C04", "untouched", 0),
    ("T09", "untouched", 0),
]

MIX_SIREN = [
    # 클립 with_siren (도로+사이렌)
    ("C01", "with_siren", 1),
    ("C02", "with_siren", 1),
    ("C04", "with_siren", 1),
]


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


def mix_audio(road_path, siren_path, alpha=0.3, out_path="/tmp/_mix.wav"):
    """도로 + α·사이렌 합성."""
    road, sr = sf.read(str(road_path))
    if road.ndim > 1:
        road = road.mean(axis=1)
    siren, sr2 = sf.read(str(siren_path))
    if siren.ndim > 1:
        siren = siren.mean(axis=1)

    # 길이 맞추기
    L = min(len(road), len(siren))
    mix = (1 - alpha) * road[:L] + alpha * siren[:L]
    # 정규화
    mix = mix / (np.abs(mix).max() + 1e-6) * 0.9
    sf.write(out_path, mix, sr)
    return out_path


def main():
    print(f"[device] {DEVICE}")

    # ── 1. AST + 템플릿 ──────────────────────
    print("\n[1] AST 로드 + 템플릿 학습")
    ast = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(DEVICE).eval()

    siren_feats = []
    for name in SIREN_TEMPLATE_SOURCES:
        p = REAL_SIRENS / f"{name}_16k.wav"
        if p.exists():
            siren_feats.append(get_ast_feature(ast, p))
    siren_template = F.normalize(torch.stack(siren_feats).mean(0), dim=-1)
    print(f"  siren_template: {len(siren_feats)} files")

    # 클립 메타 로드
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d

    road_feats = []
    for clip in ROAD_TEMPLATE_SOURCES:
        rel = metas[clip]["audio"]["untouched"]
        p = Path(metas[clip]["_meta_path"]).parent / rel
        road_feats.append(get_ast_feature(ast, p))
    road_template = F.normalize(torch.stack(road_feats).mean(0), dim=-1)
    print(f"  road_template: {len(road_feats)} files")

    def score(feat):
        return ((feat @ siren_template) - (feat @ road_template)).item()

    # ── 2. Score 수집 ────────────────────────
    print("\n[2] Score 수집")
    all_scores = []
    all_labels = []
    all_names = []

    # 양성: real_sirens
    print("  [siren] real_sirens")
    for name, label in SIREN_POSITIVE:
        p = REAL_SIRENS / f"{name}.wav"
        if p.exists():
            s = score(get_ast_feature(ast, p))
            all_scores.append(s)
            all_labels.append(label)
            all_names.append(name)
            print(f"    {name}: {s:+.4f}")

    # 양성: with_siren (도로+사이렌)
    print("  [siren] with_siren (mix)")
    for clip, key, label in MIX_SIREN:
        rel = metas[clip]["audio"].get(key)
        if isinstance(rel, str):
            p = Path(metas[clip]["_meta_path"]).parent / rel
            if p.exists():
                s = score(get_ast_feature(ast, p))
                all_scores.append(s)
                all_labels.append(label)
                all_names.append(f"{clip}/{key}")
                print(f"    {clip}/{key}: {s:+.4f}")

    # 양성: 합성 (T09 + real_sirens)
    print("  [siren] synthetic mix (T09 + real_sirens)")
    t09_untouched = Path(metas["T09"]["_meta_path"]).parent / metas["T09"]["audio"]["untouched"]
    for name in ["G44-04-Ambulance Siren", "G44-05-Warbling Boat Siren", "G44-07-Emergency Siren"]:
        p = REAL_SIRENS / f"{name}_16k.wav"
        if p.exists():
            mix_path = mix_audio(t09_untouched, p, alpha=0.3)
            s = score(get_ast_feature(ast, mix_path))
            all_scores.append(s)
            all_labels.append(1)
            all_names.append(f"T09+{name}")
            print(f"    T09+{name}: {s:+.4f}")

    # 음성: untouched
    print("  [road] untouched")
    for clip, key, label in ROAD_NEGATIVE:
        rel = metas[clip]["audio"].get(key)
        if isinstance(rel, str):
            p = Path(metas[clip]["_meta_path"]).parent / rel
            if p.exists():
                s = score(get_ast_feature(ast, p))
                all_scores.append(s)
                all_labels.append(label)
                all_names.append(f"{clip}/{key}")
                print(f"    {clip}/{key}: {s:+.4f}")

    # 음성: 합성 (T09 + 도로 소음, 다른 클립)
    print("  [road] synthetic mix (T09 + 다른 도로)")
    for clip in ["C04"]:
        rel = metas[clip]["audio"].get("untouched")
        if isinstance(rel, str):
            p = Path(metas[clip]["_meta_path"]).parent / rel
            if p.exists():
                mix_path = mix_audio(t09_untouched, p, alpha=0.3)
                s = score(get_ast_feature(ast, mix_path))
                all_scores.append(s)
                all_labels.append(0)
                all_names.append(f"T09+{clip}")
                print(f"    T09+{clip}: {s:+.4f}")

    # ── 3. ROC 분석 ──────────────────────────
    print("\n" + "=" * 65)
    print("[3] ROC 분석")
    print("=" * 65)

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    print(f"\n[data] N = {len(scores)} (pos={labels.sum()}, neg={(1-labels).sum()})")
    print(f"[score] mean(pos) = {scores[labels==1].mean():+.4f}")
    print(f"[score] mean(neg) = {scores[labels==0].mean():+.4f}")

    # sklearn 시도
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        print(f"\n[AUC] = {roc_auc:.4f}")

        # 최적 임계값 (Youden's J)
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_thr = thresholds[best_idx]
        print(f"[best threshold (Youden)] = {best_thr:+.4f}")
        print(f"  TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f}")

        # Precision-Recall
        prec, rec, pr_thr = precision_recall_curve(labels, scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        best_f1_idx = np.argmax(f1)
        best_f1_thr = pr_thr[best_f1_idx] if best_f1_idx < len(pr_thr) else 0
        print(f"\n[best F1] = {f1[best_f1_idx]:.3f}")
        print(f"  threshold = {best_f1_thr:+.4f}")
        print(f"  precision = {prec[best_f1_idx]:.3f}")
        print(f"  recall = {rec[best_f1_idx]:.3f}")

        # 운영점 3종
        print("\n[운영점 3종]")
        for mode, target_prec, target_rec in [
            ("보수적 (precision≥0.95)", 0.95, None),
            ("균형 (F1 max)", None, None),
            ("민감 (recall≥0.95)", None, 0.95),
        ]:
            # precision 기준
            if target_prec:
                idx = np.where(prec >= target_prec)[0]
                if len(idx) > 0:
                    thr = pr_thr[idx[0]] if idx[0] < len(pr_thr) else 0
                    print(f"  {mode}: threshold = {thr:+.4f} "
                          f"(prec={prec[idx[0]]:.3f}, rec={rec[idx[0]]:.3f})")
            elif target_rec:
                idx = np.where(rec >= target_rec)[0]
                if len(idx) > 0:
                    i = idx[-1]
                    thr = pr_thr[i] if i < len(pr_thr) else 0
                    print(f"  {mode}: threshold = {thr:+.4f} "
                          f"(prec={prec[i]:.3f}, rec={rec[i]:.3f})")
            else:
                print(f"  {mode}: threshold = {best_f1_thr:+.4f} "
                      f"(prec={prec[best_f1_idx]:.3f}, rec={rec[best_f1_idx]:.3f})")

        # 임계값별 성능
        print("\n[임계값별 성능]")
        print(f"{'threshold':>10s}  {'TPR':>6s}  {'FPR':>6s}  {'prec':>6s}  {'rec':>6s}")
        print("-" * 50)
        for thr in [-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30]:
            pred = (scores >= thr).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            tn = ((pred == 0) & (labels == 0)).sum()
            tpr_ = tp / (tp + fn + 1e-6)
            fpr_ = fp / (fp + tn + 1e-6)
            prec_ = tp / (tp + fp + 1e-6)
            rec_ = tpr_
            print(f"{thr:>+10.2f}  {tpr_:>6.3f}  {fpr_:>6.3f}  {prec_:>6.3f}  {rec_:>6.3f}")

    except ImportError:
        print("[warn] sklearn 없음. pip install scikit-learn")
        print("\n[수동 분석]")
        print(f"{'name':>35s}  {'label':>5s}  {'score':>8s}")
        for name, label, s in sorted(zip(all_names, all_labels, all_scores),
                                      key=lambda x: -x[2]):
            print(f"{name:>35s}  {label:>5d}  {s:>+8.4f}")


if __name__ == "__main__":
    main()