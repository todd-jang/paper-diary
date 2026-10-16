"""Compare cosine, negative Euclidean, and normal-reference Mahalanobis scores."""
from __future__ import annotations

import argparse
from pathlib import Path

from _siren_score_utils import (
    DEFAULT_CLIPS,
    DEFAULT_CSV,
    cosine_similarity,
    estimate_precision,
    euclidean_similarity,
    extract_audio_features,
    extract_vit_features,
    mahalanobis_similarity,
    require_audio_pair,
    score_stats,
)


def pair_score(metric, vector, audio_u, audio_s, precision=None):
    if metric == "cosine":
        similarity = cosine_similarity
    elif metric == "euclidean":
        similarity = euclidean_similarity
    else:
        similarity = lambda left, right: mahalanobis_similarity(left, right, precision)
    return similarity(vector, audio_u) - similarity(vector, audio_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    clips = (*DEFAULT_CLIPS, "T09")
    frames = extract_vit_features(
        csv_path=args.csv, clips=clips, batch_size=args.batch_size, device=args.device,
    )
    audio = extract_audio_features(clips=DEFAULT_CLIPS, device=args.device)
    precision = estimate_precision([frame["features"][-1]["cls"] for frame in frames["T09"]])

    for metric in ("cosine", "euclidean", "mahalanobis"):
        scores = []
        for clip in DEFAULT_CLIPS:
            audio_u, audio_s = require_audio_pair(audio, clip)
            for frame in frames[clip]:
                scores.append(pair_score(
                    metric, frame["features"][-1]["cls"], audio_u, audio_s, precision
                ))
        mean, std = score_stats(scores)
        print(f"{metric:12s} mean={mean:+.5f} std={std:.5f}")


if __name__ == "__main__":
    main()
