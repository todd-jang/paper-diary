"""Audio-video contrastive baseline for locally downloaded VGGSound clips."""

import argparse
import csv
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def find_video_path(row, video_root):
    supplied_path = row.get("video_path") or row.get("path")
    if supplied_path:
        path = Path(supplied_path)
        if not path.is_absolute():
            path = video_root / path
        return path if path.is_file() else None

    video_id = row.get("youtube_id") or row.get("video_id")
    start = (
        row.get("start_seconds")
        or row.get("start_sec")
        or row.get("start_timestamp")
    )
    if not video_id:
        return None

    if start:
        try:
            start_number = int(float(start))
        except ValueError:
            start_number = None
        for path in video_root.rglob(f"{video_id}_*.mp4"):
            suffix = path.stem[len(video_id) + 1:]
            try:
                if int(float(suffix)) == start_number:
                    return path
            except ValueError:
                continue
    matches = sorted(video_root.rglob(f"{video_id}*.mp4"))
    return matches[0] if matches else None


class VGGSoundDataset(Dataset):
    def __init__(self, metadata_csv, video_root, split, frames, image_size, duration):
        self.video_root = Path(video_root)
        self.frames = frames
        self.image_size = image_size
        self.duration = duration
        self.records = []

        with Path(metadata_csv).open(newline="", encoding="utf-8-sig") as metadata_file:
            rows = list(csv.reader(metadata_file))
        if not rows:
            raise ValueError("Metadata CSV is empty")

        normalize = lambda value: value.strip().lower().replace(" ", "_")
        known_headers = {
            "video_path", "path", "youtube_id", "video_id", "start_seconds",
            "start_sec", "start_timestamp", "split",
        }
        first_row = [normalize(value) for value in rows[0]]
        if any(value in known_headers for value in first_row):
            headers, data_rows = first_row, rows[1:]
        elif len(first_row) >= 4:
            headers, data_rows = ["youtube_id", "start_seconds", "label", "split"], rows
        else:
            raise ValueError("Metadata CSV needs a header or four VGGSound columns")

        for values in data_rows:
            row = dict(zip(headers, (value.strip() for value in values)))
            row_split = row.get("split")
            if row_split and split and row_split.lower() != split.lower():
                continue
            path = find_video_path(row, self.video_root)
            if path:
                self.records.append(path)

        if not self.records:
            raise ValueError(
                f"No videos found for split={split!r}. Check the CSV and --video-root."
            )

    def __len__(self):
        return len(self.records)

    def _decode(self, command):
        try:
            result = subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            details = error.stderr.decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"FFmpeg failed for {command[-1]}: {details}") from error
        return result.stdout

    def __getitem__(self, index):
        path = self.records[index]
        size = self.image_size
        fps = self.frames / self.duration
        video_filter = (
            f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
        )
        video_bytes = self._decode([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", str(self.duration),
            "-vf", video_filter, "-frames:v", str(self.frames), "-f", "rawvideo",
            "-pix_fmt", "rgb24", "pipe:1",
        ])
        frame_bytes = size * size * 3
        decoded_frames = np.frombuffer(video_bytes, dtype=np.uint8)
        decoded_frames = decoded_frames[:len(decoded_frames) // frame_bytes * frame_bytes]
        decoded_frames = decoded_frames.reshape(-1, size, size, 3)
        if len(decoded_frames) == 0:
            raise RuntimeError(f"No video frames decoded from {path}")
        frame_indices = np.linspace(0, len(decoded_frames) - 1, self.frames).round().astype(int)
        frames = torch.from_numpy(decoded_frames[frame_indices].copy()).permute(0, 3, 1, 2)
        frames = frames.float().div_(255)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        audio_bytes = self._decode([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", str(self.duration),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
        ])
        audio = torch.from_numpy(np.frombuffer(audio_bytes, dtype="<f4").copy())
        sample_count = int(self.duration * 16000)
        audio = F.pad(audio[:sample_count], (0, max(0, sample_count - len(audio))))
        return frames, audio


def symmetric_contrastive_loss(audio_features, video_features, temperature=0.07):
    audio_features = F.normalize(audio_features, dim=-1)
    video_features = F.normalize(video_features, dim=-1)
    logits = audio_features @ video_features.T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return loss, accuracy


class AudioVideoModel(nn.Module):
    def __init__(self, audio_checkpoint, projection_dim):
        super().__init__()
        from transformers import AutoModel
        from torchvision.models import ResNet18_Weights, resnet18

        self.audio_encoder = AutoModel.from_pretrained(audio_checkpoint)
        self.video_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.video_encoder.fc = nn.Identity()
        for encoder in (self.audio_encoder, self.video_encoder):
            encoder.requires_grad_(False)

        audio_dim = self.audio_encoder.config.hidden_size
        self.audio_projection = nn.Linear(audio_dim, projection_dim)
        self.video_projection = nn.Linear(512, projection_dim)

    def train(self, mode=True):
        super().train(mode)
        self.audio_encoder.eval()
        self.video_encoder.eval()
        return self

    def forward(self, frames, audio):
        batch_size, frame_count, channels, height, width = frames.shape
        with torch.no_grad():
            frame_features = self.video_encoder(
                frames.reshape(batch_size * frame_count, channels, height, width)
            )
            video_features = frame_features.reshape(batch_size, frame_count, -1).mean(dim=1)

            audio = (audio - audio.mean(dim=1, keepdim=True)) / (
                audio.std(dim=1, keepdim=True).clamp_min(1e-7)
            )
            audio_features = self.audio_encoder(input_values=audio).last_hidden_state.mean(dim=1)

        return self.audio_projection(audio_features), self.video_projection(video_features)


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/vggsound_av.pt")
    parser.add_argument("--inspect-data", action="store_true")
    args = parser.parse_args()

    if args.frames < 1 or args.duration <= 0 or args.batch_size < 2:
        parser.error("frames and duration must be positive; batch-size must be at least 2")

    dataset = VGGSoundDataset(
        args.metadata_csv, args.video_root, args.split,
        args.frames, args.image_size, args.duration,
    )
    if args.limit:
        dataset.records = dataset.records[:args.limit]
    print(f"[data] {len(dataset)} clips | split={args.split}")

    if args.inspect_data:
        frames, audio = dataset[0]
        print(f"[sample] frames={tuple(frames.shape)} audio={tuple(audio.shape)} @ 16 kHz")
        return

    if len(dataset) < args.batch_size:
        parser.error("dataset must contain at least --batch-size clips for in-batch negatives")

    device = choose_device(args.device)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = AudioVideoModel(args.audio_model, args.projection_dim).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[device] {device} | trainable projection params={trainable:,}")

    optimizer = torch.optim.AdamW(
        [*model.audio_projection.parameters(), *model.video_projection.parameters()],
        lr=args.learning_rate,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        for frames, audio in loader:
            frames = frames.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            audio_features, video_features = model(frames, audio)
            loss, accuracy = symmetric_contrastive_loss(audio_features, video_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += accuracy.item()

        steps = len(loader)
        print(
            f"[epoch {epoch}/{args.epochs}] loss={total_loss / steps:.4f} "
            f"audio->video top1={total_accuracy / steps:.3f}"
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "audio_model": args.audio_model,
        "audio_projection": model.audio_projection.state_dict(),
        "video_projection": model.video_projection.state_dict(),
        "video_encoder": "torchvision-resnet18-imagenet1k",
        "projection_dim": args.projection_dim,
    }, checkpoint_path)
    print(f"[saved] {checkpoint_path}")


if __name__ == "__main__":
    main()