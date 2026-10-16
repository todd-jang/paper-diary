"""Fuse per-frame audio-conditioned vision scores with RT-DETR object confidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from _siren_score_utils import (
    DEFAULT_CLIPS,
    DEFAULT_CSV,
    DEFAULT_DETECTIONS_ROOT,
    cosine_score,
    extract_audio_features,
    extract_vit_features,
    read_detections,
    require_audio_pair,
    rtdetr_frame_score,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.alpha < 0 or args.beta < 0 or args.alpha + args.beta == 0:
        parser.error("--alpha and --beta must be non-negative and not both zero")

    frames = extract_vit_features(
        csv_path=args.csv, clips=DEFAULT_CLIPS, batch_size=args.batch_size,
        device=args.device,
    )
    audio = extract_audio_features(clips=DEFAULT_CLIPS, device=args.device)
    siren_classes = {"bus", "truck", "person", "bicycle", "ambulance"}
    rows = []
    for clip in DEFAULT_CLIPS:
        audio_u, audio_s = require_audio_pair(audio, clip)
        detection_doc = read_detections(clip, args.detections)
        for frame in frames[clip]:
            mae_score = cosine_score(frame["features"][-1]["cls"], audio_u, audio_s)
            detector_score = rtdetr_frame_score(
                detection_doc, frame["frame_idx"], siren_classes
            )
            fused_score = args.alpha * mae_score + args.beta * detector_score
            rows.append({
                "clip": clip,
                "frame_idx": frame["frame_idx"],
                "mae_siren_score": mae_score,
                "rtdetr_siren_score": detector_score,
                "fused_score": fused_score,
            })
        clip_rows = [row for row in rows if row["clip"] == clip]
        average = sum(row["fused_score"] for row in clip_rows) / len(clip_rows)
        print(f"{clip}: frames={len(clip_rows)} fused_mean={average:+.5f}")

    if args.output:
        write_csv(args.output, rows)
        print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
