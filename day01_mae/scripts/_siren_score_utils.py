"""Shared helpers for the siren-scoring experiments.

ViT-MAE and AST are independently pretrained, so their raw similarities are
exploratory signals, not calibrated cross-modal probabilities.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PIPELINE_ROOT = ROOT.parent.parent / "spatial-avla-pipeline"
DEFAULT_CSV = ROOT / "exports" / "mae_smoke_meta.csv"
DEFAULT_CLIPS_ROOT = PIPELINE_ROOT / "exports" / "TL_A"
DEFAULT_DETECTIONS_ROOT = PIPELINE_ROOT / "outputs" / "detections"
DEFAULT_CLIPS = ("C01", "C02", "C04")
AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_metas(clips_root: Path = DEFAULT_CLIPS_ROOT) -> dict[str, dict[str, Any]]:
    metas = {}
    for path in sorted(clips_root.rglob("clip_meta.json")):
        with path.open(encoding="utf-8") as stream:
            meta = json.load(stream)
        meta["_meta_path"] = str(path)
        metas[meta["clip_id"]] = meta
    return metas


def frame_index(path: str, fallback: int) -> int:
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1]) - 1
    except (IndexError, ValueError):
        return fallback


def extract_vit_features(
    model_name: str = "facebook/vit-mae-base",
    csv_path: Path = DEFAULT_CSV,
    clips: tuple[str, ...] = DEFAULT_CLIPS,
    layers: tuple[int, ...] = (-1,),
    batch_size: int = 8,
    device: str = "auto",
) -> dict[str, list[dict[str, Any]]]:
    """Return ordered per-frame CLS, patch-mean, and concatenated features."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import ViTMAEForPreTraining

    from data.dataset import FrameMAEDataset
    from data.transforms import val_tf

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = pick_device(device)
    dataset = FrameMAEDataset(str(csv_path), transform=val_tf)
    model = (
        ViTMAEForPreTraining.from_pretrained(model_name)
        .vit.to(device)
        .eval()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    wanted = set(clips)
    records: dict[str, list[dict[str, Any]]] = {clip: [] for clip in clips}
    offset = 0

    for images in loader:
        with torch.inference_mode():
            output = model(
                pixel_values=images.to(device),
                interpolate_pos_encoding=True,
                output_hidden_states=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("ViT-MAE did not return hidden states")
        resolved_layers = {
            layer: layer if layer >= 0 else len(hidden_states) + layer
            for layer in layers
        }
        for layer, index in resolved_layers.items():
            if not 0 <= index < len(hidden_states):
                raise ValueError(
                    f"Requested layer {layer}; model has {len(hidden_states)} hidden states"
                )
        batch_end = min(offset + len(images), len(dataset))
        for row_index in range(offset, batch_end):
            row = dataset.df.iloc[row_index]
            clip = str(row["clip"])
            if clip not in wanted:
                continue
            batch_index = row_index - offset
            row_features = {}
            for layer, state_index in resolved_layers.items():
                hidden = hidden_states[state_index][batch_index].detach().float().cpu()
                cls = hidden[0]
                patch = hidden[1:].mean(dim=0)
                row_features[layer] = {
                    "cls": cls,
                    "patch": patch,
                    "both": torch.cat((cls, patch)),
                }
            path = str(row["path"])
            records[clip].append({
                "frame_idx": frame_index(path, row_index),
                "path": path,
                "features": row_features,
            })
        offset = batch_end

    for clip, frames in records.items():
        frames.sort(key=lambda item: item["frame_idx"])
        if not frames:
            raise ValueError(f"No frames for {clip!r} in {csv_path}")
    return records


def extract_audio_features(
    clips: tuple[str, ...] = DEFAULT_CLIPS,
    clips_root: Path = DEFAULT_CLIPS_ROOT,
    device: str = "auto",
    model_name: str = AST_NAME,
) -> dict[str, dict[str, torch.Tensor]]:
    import torch
    from transformers import ASTModel

    from scripts._ast_utils import load_logmel

    device = pick_device(device)
    metas = load_clip_metas(clips_root)
    model = ASTModel.from_pretrained(model_name).to(device).eval()
    features: dict[str, dict[str, torch.Tensor]] = {}
    for clip in clips:
        if clip not in metas:
            raise FileNotFoundError(f"clip_meta.json not found for {clip!r} under {clips_root}")
        meta = metas[clip]
        clip_dir = Path(meta["_meta_path"]).parent
        features[clip] = {}
        audio_meta = meta.get("audio", {})
        for key in ("untouched", "with_siren"):
            relative_path = audio_meta.get(key)
            if not isinstance(relative_path, str):
                continue
            audio_path = clip_dir / relative_path
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio path in {clip} metadata does not exist: {audio_path}")
            log_mel = load_logmel(str(audio_path)).unsqueeze(0)
            inputs = log_mel.transpose(1, 2)
            inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-6)
            with torch.inference_mode():
                output = model(inputs.to(device))
            features[clip][key] = output.pooler_output.squeeze(0).float().cpu()
    return features


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    import torch.nn.functional as F

    if left.shape[-1] != right.shape[-1]:
        raise ValueError(f"Embedding dimension mismatch: {left.shape[-1]} != {right.shape[-1]}")
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())


def cosine_score(vision: torch.Tensor, audio_untouched: torch.Tensor,
                 audio_siren: torch.Tensor) -> float:
    return (
        cosine_similarity(vision, audio_untouched)
        - cosine_similarity(vision, audio_siren)
    )


def normalized_cosine_score(vision: torch.Tensor, audio_untouched: torch.Tensor,
                            audio_siren: torch.Tensor) -> float:
    sim_untouched = cosine_similarity(vision, audio_untouched)
    sim_siren = cosine_similarity(vision, audio_siren)
    return (sim_untouched - sim_siren) / (sim_untouched + sim_siren + 1e-6)


def softmax_score(vision: torch.Tensor, audio_untouched: torch.Tensor,
                  audio_siren: torch.Tensor, temperature: float = 0.07) -> float:
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = torch.tensor([
        cosine_similarity(vision, audio_untouched),
        cosine_similarity(vision, audio_siren),
    ]) / temperature
    return float((F.softmax(logits, dim=-1)[0] - 0.5).item())


def euclidean_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    import torch

    if left.shape[-1] != right.shape[-1]:
        raise ValueError(f"Embedding dimension mismatch: {left.shape[-1]} != {right.shape[-1]}")
    return -float(torch.linalg.vector_norm(left - right).item())


def mahalanobis_similarity(
    left: torch.Tensor, right: torch.Tensor, covariance_inverse: torch.Tensor
) -> float:
    import torch

    if left.shape[-1] != right.shape[-1] or covariance_inverse.shape != (
        left.shape[-1], left.shape[-1]
    ):
        raise ValueError("Mahalanobis embedding and covariance dimensions must match")
    difference = (left - right).reshape(-1)
    distance_squared = difference @ covariance_inverse @ difference
    return -float(torch.sqrt(torch.clamp(distance_squared, min=0)).item())


def estimate_precision(features: list[torch.Tensor], ridge: float = 1e-4) -> torch.Tensor:
    import torch

    if len(features) < 2:
        raise ValueError("At least two normal frame features are required for covariance")
    matrix = torch.stack(features).double()
    centered = matrix - matrix.mean(dim=0)
    covariance = centered.T @ centered / (len(matrix) - 1)
    return torch.linalg.pinv(
        covariance + ridge * torch.eye(covariance.shape[0], dtype=covariance.dtype)
    ).float()


def read_detections(clip: str, detections_root: Path = DEFAULT_DETECTIONS_ROOT) -> dict:
    path = detections_root / f"{clip}__rtdetr-l.json"
    if not path.is_file():
        raise FileNotFoundError(f"RT-DETR detections not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def rtdetr_frame_score(document: dict, frame_idx: int,
                       classes: set[str]) -> float:
    frame = document.get("frames", {}).get(str(frame_idx), {})
    boxes = frame.get("boxes", [])
    return max(
        (float(box["conf"]) for box in boxes if box.get("class_name") in classes),
        default=0.0,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_stats(scores: list[float]) -> tuple[float, float]:
    if not scores:
        raise ValueError("Cannot calculate statistics for an empty score list")
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return mean, math.sqrt(variance)


def require_audio_pair(audio_features: dict[str, dict[str, torch.Tensor]],
                       clip: str) -> tuple[torch.Tensor, torch.Tensor]:
    pair = audio_features.get(clip, {})
    if "untouched" not in pair or "with_siren" not in pair:
        raise ValueError(f"{clip} must have both untouched and with_siren audio features")
    return pair["untouched"], pair["with_siren"]
