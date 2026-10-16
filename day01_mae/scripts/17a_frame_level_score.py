"""Calculate and export per-frame siren scores for C01, C02, and C04."""
from __future__ import annotations

import argparse
from pathlib import Path

from _siren_score_utils import (
    DEFAULT_CLIPS,
    DEFAULT_CSV,
    cosine_similarity,
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
        scores = []
        for frame in frames[clip]:
            vision = frame["features"][-1]["cls"]
            cos_u = cosine_similarity(vision, audio_u)
            score = cosine_score(vision, audio_u, audio_s)
            cos_s = cosine_similarity(vision, audio_s)
            scores.append(score)
            rows.append({
                "clip": clip,
                "frame_idx": frame["frame_idx"],
                "cos_untouched": cos_u,
                "cos_with_siren": cos_s,
                "siren_score": score,
            })
        mean, std = score_stats(scores)
        print(f"{clip}: frames={len(scores)} mean={mean:+.5f} std={std:.5f} "
              f"min={min(scores):+.5f} max={max(scores):+.5f}")
    if args.output:
        write_csv(args.output, rows)
        print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
