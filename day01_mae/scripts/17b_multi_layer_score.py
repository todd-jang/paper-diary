"""Compare CLS and patch-mean siren scores across ViT-MAE layers."""
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
    args = parser.parse_args()
    layers = (6, 9, 12)
    frames = extract_vit_features(
        csv_path=args.csv, clips=DEFAULT_CLIPS, layers=layers,
        batch_size=args.batch_size, device=args.device,
    )
    audio = extract_audio_features(clips=DEFAULT_CLIPS, device=args.device)

    scores: dict[tuple[int, str], list[float]] = {
        (layer, mode): [] for layer in layers for mode in ("cls", "patch")
    }
    for clip in DEFAULT_CLIPS:
        audio_u, audio_s = require_audio_pair(audio, clip)
        for frame in frames[clip]:
            for layer in layers:
                for mode in ("cls", "patch"):
                    scores[(layer, mode)].append(
                        cosine_score(frame["features"][layer][mode], audio_u, audio_s)
                    )

    all_means = []
    print(f"{'layer':>5} {'feature':>8} {'mean_score':>12} {'std':>10}")
    for (layer, mode), values in scores.items():
        mean, std = score_stats(values)
        all_means.append(mean)
        print(f"{layer:5d} {mode:>8} {mean:+12.5f} {std:10.5f}")
    print(f"combined mean={sum(all_means) / len(all_means):+.5f}")


if __name__ == "__main__":
    main()
