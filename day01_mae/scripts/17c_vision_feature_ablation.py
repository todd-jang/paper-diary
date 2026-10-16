"""Compare ViT-MAE base CLS/patch features; opt in to the large checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from _siren_score_utils import (
    DEFAULT_CLIPS,
    DEFAULT_CSV,
    cosine_score,
    extract_audio_features,
    extract_vit_features,
    require_audio_pair,
    score_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-large", action="store_true",
                        help="Download and evaluate facebook/vit-mae-large (large checkpoint)")
    args = parser.parse_args()
    audio = extract_audio_features(clips=DEFAULT_CLIPS, device=args.device)
    variants = [("facebook/vit-mae-base", "base")]
    if args.include_large:
        variants.append(("facebook/vit-mae-large", "large"))

    for model_name, label in variants:
        frames = extract_vit_features(
            model_name=model_name, csv_path=args.csv, clips=DEFAULT_CLIPS,
            batch_size=args.batch_size, device=args.device,
        )
        for mode in ("cls", "patch"):
            values = []
            incompatible = False
            for clip in DEFAULT_CLIPS:
                audio_u, audio_s = require_audio_pair(audio, clip)
                for frame in frames[clip]:
                    vector = frame["features"][-1][mode]
                    if vector.numel() != audio_u.numel():
                        incompatible = True
                        break
                    values.append(cosine_score(vector, audio_u, audio_s))
                if incompatible:
                    break
            if incompatible:
                print(f"{label}_{mode}: not comparable (vision/audio embedding dimensions differ)")
                continue
            mean, std = score_stats(values)
            print(f"{label}_{mode}: mean={mean:+.5f} std={std:.5f}")

        # CLS and patch vectors remain in the model's native dimension. Fuse their
        # scores rather than comparing a concatenated vector with the AST space.
        if label == "base":
            both_scores = []
            for clip in DEFAULT_CLIPS:
                audio_u, audio_s = require_audio_pair(audio, clip)
                for frame in frames[clip]:
                    feature_pair = frame["features"][-1]
                    cls_score = cosine_score(feature_pair["cls"], audio_u, audio_s)
                    patch_score = cosine_score(feature_pair["patch"], audio_u, audio_s)
                    both_scores.append((cls_score + patch_score) / 2)
            mean, std = score_stats(both_scores)
            print(f"base_both(mean of CLS + patch scores): mean={mean:+.5f} std={std:.5f}")


if __name__ == "__main__":
    main()
