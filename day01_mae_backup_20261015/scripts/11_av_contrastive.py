"""Audio-Visual Contrastive: audio.untouched + with_siren 쌍."""
import sys, json
from pathlib import Path

MAE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAE_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ViTMAEConfig

from data.dataset import FrameMAEDataset
from data.transforms import val_tf

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
TL_A = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_clip_metas():
    metas = {}
    for p in TL_A.rglob("clip_meta.json"):
        d = json.load(open(p))
        d["_meta_path"] = str(p)
        metas[d["clip_id"]] = d
    return metas


def make_av_samples(df, metas):
    """프레임별 AV pair (audio 없으면 skip)."""
    samples = []
    skipped = {}

    for _, row in df.iterrows():
        clip = row["clip"]
        meta = metas[clip]
        audio = meta.get("audio", None)

        if not isinstance(audio, dict):
            skipped[clip] = "audio 없음"
            continue

        u = audio.get("untouched", None)
        w = audio.get("with_siren", None)

        if not isinstance(u, str) or not isinstance(w, str):
            skipped[clip] = f"untouched={u!r} with_siren={w!r}"
            continue

        clip_dir = Path(meta["_meta_path"]).parent
        u_path = clip_dir / u
        w_path = clip_dir / w

        if not (u_path.exists() and w_path.exists()):
            skipped[clip] = f"file 없음: {u_path.exists()}/{w_path.exists()}"
            continue

        samples.append({
            "frame": row["path"],
            "audio_neg": str(u_path),
            "audio_pos": str(w_path),
            "clip": clip,
        })

    print(f"[av] valid pairs = {len(samples)} / {len(df)}")
    for c, r in skipped.items():
        print(f"  [skip] {c}: {r}")
    return samples


def main():
    ds = FrameMAEDataset(CSV, transform=val_tf)
    metas = load_clip_metas()
    samples = make_av_samples(ds.df, metas)

    if len(samples) == 0:
        print("[warn] AV pair가 하나도 없습니다.")
        return

    # 클립별 통계
    from collections import Counter
    print(f"[av] by clip = {dict(Counter(s['clip'] for s in samples))}")

    # ViTMAE 로드
    config = ViTMAEConfig(
        image_size=224, patch_size=16,
        hidden_size=384, num_hidden_layers=6,
        num_attention_heads=6, intermediate_size=1536,
        mask_ratio=0.0,
    )
    model = ViTMAEForPreTraining(config).to(DEVICE).eval()

    # 프레임 feature (전체 80장)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    feats_all = []
    with torch.no_grad():
        for imgs in dl:
            out = model.vit(pixel_values=imgs.to(DEVICE),
                            interpolate_pos_encoding=True)
            feats_all.append(out.last_hidden_state[:, 0, :].cpu())
    feats = torch.cat(feats_all, dim=0)
    print(f"[feat] {feats.shape}")

    # 프레임 → AV pair 매핑
    path_to_idx = {row["path"]: i for i, row in ds.df.iterrows()}
    print(f"[av] sample pairs = {len(samples)}")
    print(f"[av] 예시: {samples[0]}")

    print()
    print("[info] AV pair 추출 완료.")
    print("[info] 다음: 오디오 인코더(AST, PANNs 등)로 audio_neg/pos feature 뽑고")
    print("        프레임 feature와 cross-modal contrastive loss 계산.")


if __name__ == "__main__":
    main()
    