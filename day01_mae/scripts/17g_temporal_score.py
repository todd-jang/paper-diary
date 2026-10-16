"""Export per-frame score trajectories and first differences."""
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
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frames = extract_vit_features(
        csv_path=args.csv, clips=DEFAULT_CLIPS, batch_size=args.batch_size,
        device=args.device,
    )
    audio = extract_audio_features(clips=DEFAULT_CLIPS, device=args.device)
    rows = []
    for clip in DEFAULT_CLIPS:
        audio_u, audio_s = require_audio_pair(audio, clip)
        previous = None
        scores = []
        for frame in frames[clip]:
            score = cosine_score(frame["features"][-1]["cls"], audio_u, audio_s)
            delta = None if previous is None else score - previous
            rows.append({
                "clip": clip,
                "frame_idx": frame["frame_idx"],
                "siren_score": score,
                "delta_score": delta,
            })
            scores.append(score)
            previous = score
        mean, std = score_stats(scores)
        ups = sum(row["delta_score"] > 0 for row in rows
                  if row["clip"] == clip and row["delta_score"] is not None)
        downs = sum(row["delta_score"] < 0 for row in rows
                    if row["clip"] == clip and row["delta_score"] is not None)
        print(f"{clip}: mean={mean:+.5f} std={std:.5f} "
              f"positive_steps={ups} negative_steps={downs}")
    if args.output:
        write_csv(args.output, rows)
        print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
