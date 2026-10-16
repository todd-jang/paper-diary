"""Export frame-wise mini-MAE distances from a normal-clip feature centroid."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT.parent.parent / "spatial-avla-pipeline"
MODEL_SRC = ROOT / "minimae" / "src"
FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "minimae" / "runs" / "mini_mae_tiny.pt",
    )
    parser.add_argument("--frames-root", type=Path, default=PIPELINE_ROOT / "exports" / "TL_A")
    parser.add_argument(
        "--output",
        type=Path,
        default=PIPELINE_ROOT / "outputs" / "mae_frame_report.json",
    )
    parser.add_argument("--normal-clip", default="T09")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(checkpoint_path: Path, device: str):
    sys.path.insert(0, str(MODEL_SRC))
    from model import mae_vit_small, mae_vit_tiny

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    constructors = {"vit_tiny": mae_vit_tiny, "vit_small": mae_vit_small}
    try:
        constructor = constructors[args["arch"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported MAE architecture: {args.get('arch')!r}") from exc
    model = constructor(
        img_size=int(args["img_size"]),
        patch_size=int(args["patch_size"]),
        mask_ratio=0.0,
        norm_pix_loss=bool(args.get("norm_pix_loss", 1)),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), args


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    array = np.asarray(cropped, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def frame_records(frames_root: Path) -> list[tuple[str, int, Path]]:
    records = []
    for path in sorted(frames_root.rglob("frame_*.jpg")):
        match = FRAME_RE.fullmatch(path.name)
        if match is None:
            continue
        clip = path.parent.parent.name if path.parent.name == "frames" else path.parent.name
        records.append((clip, int(match.group(1)) - 1, path))
    return records


@torch.inference_mode()
def extract_features(model, records, device: str, image_size: int, batch_size: int):
    features = {}
    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        images = torch.stack([load_image(path, image_size) for _, _, path in batch_records])
        x = model.patch_embed(images.to(device)) + model.enc_pos_embed
        for block in model.enc_blocks:
            x = block(x)
        x = model.enc_norm(x).mean(dim=1).cpu()
        for record, feature in zip(batch_records, x):
            clip, frame_idx, _ = record
            if frame_idx in features.setdefault(clip, {}):
                raise ValueError(f"Duplicate frame index {clip}:{frame_idx}")
            features[clip][frame_idx] = feature
    return features


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"MAE checkpoint not found: {args.checkpoint}")
    if not args.frames_root.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {args.frames_root}")

    records = frame_records(args.frames_root)
    if not records:
        raise ValueError(f"No frame_*.jpg files found under {args.frames_root}")
    if not any(clip == args.normal_clip for clip, _, _ in records):
        raise ValueError(f"Normal reference clip {args.normal_clip!r} has no frames")

    device = pick_device(args.device)
    model, model_args = build_model(args.checkpoint, device)
    features = extract_features(
        model, records, device, int(model_args["img_size"]), args.batch_size
    )
    normal_features = torch.stack(list(features[args.normal_clip].values()))
    centroid = normal_features.mean(dim=0)

    clips = {}
    for clip, frame_features in sorted(features.items()):
        frames = {}
        for frame_idx, feature in sorted(frame_features.items()):
            distance = torch.linalg.vector_norm(feature - centroid).item()
            frames[str(frame_idx)] = {"mae_euclidean_raw": round(distance, 6)}
        clips[clip] = {"frames": frames}

    report = {
        "schema_version": 1,
        "model": {
            "architecture": model_args["arch"],
            "checkpoint": args.checkpoint.name,
            "embedding": "mean_of_final_encoder_patch_tokens",
        },
        "score": {
            "name": "mae_euclidean_raw",
            "method": "euclidean_distance_to_normal_clip_feature_centroid",
            "normal_clip": args.normal_clip,
            "normal_frame_count": len(normal_features),
        },
        "device": device,
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temp_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(args.output)
    print(
        f"[saved] {args.output} clips={len(clips)} frames={len(records)} "
        f"normal={args.normal_clip}:{len(normal_features)} device={device}"
    )


if __name__ == "__main__":
    main()
