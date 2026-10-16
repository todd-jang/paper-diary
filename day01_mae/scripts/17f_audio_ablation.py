"""Compare AST, PANNs, and Wav2Vec2 audio embeddings against ViT-MAE frames."""
from __future__ import annotations

import argparse
from pathlib import Path

from _siren_score_utils import (
    AST_NAME,
    DEFAULT_CLIPS,
    DEFAULT_CLIPS_ROOT,
    DEFAULT_CSV,
    cosine_score,
    extract_vit_features,
    load_clip_metas,
    pick_device,
    score_stats,
)


def audio_paths(clips: tuple[str, ...], clips_root: Path) -> dict[str, dict[str, Path]]:
    metas = load_clip_metas(clips_root)
    result = {}
    for clip in clips:
        if clip not in metas:
            raise FileNotFoundError(f"clip_meta.json not found for {clip!r} under {clips_root}")
        base = Path(metas[clip]["_meta_path"]).parent
        result[clip] = {}
        for key in ("untouched", "with_siren"):
            relative = metas[clip].get("audio", {}).get(key)
            if isinstance(relative, str):
                path = base / relative
                if not path.is_file():
                    raise FileNotFoundError(f"Audio path does not exist: {path}")
                result[clip][key] = path
    return result


def extract_ast(paths, device):
    import torch
    from transformers import ASTModel

    from scripts._ast_utils import load_logmel

    model = ASTModel.from_pretrained(AST_NAME).to(device).eval()
    result = {}
    with torch.inference_mode():
        for clip, pair in paths.items():
            result[clip] = {}
            for key, path in pair.items():
                inputs = load_logmel(str(path)).unsqueeze(0).transpose(1, 2)
                inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-6)
                result[clip][key] = model(inputs.to(device)).pooler_output[0].cpu()
    return result


def extract_panns(paths):
    import soundfile as sf
    import torch
    import torchaudio

    try:
        from panns_inference import AudioTagging
    except ImportError as exc:
        raise ImportError("PANNs is optional; install it with `pip install panns-inference`") from exc
    model = AudioTagging(checkpoint_path=None, device="cpu")
    result = {}
    for clip, pair in paths.items():
        result[clip] = {}
        for key, path in pair.items():
            waveform, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
            audio = torch.from_numpy(waveform.mean(axis=1))
            if sample_rate != 32000:
                audio = torchaudio.functional.resample(audio, sample_rate, 32000)
            _, embedding = model.inference(audio.unsqueeze(0).numpy())
            result[clip][key] = torch.as_tensor(embedding).reshape(-1).float()
    return result


def extract_wav2vec(paths, device):
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoModel, AutoProcessor

    model_name = "facebook/wav2vec2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    result = {}
    with torch.inference_mode():
        for clip, pair in paths.items():
            result[clip] = {}
            for key, path in pair.items():
                waveform, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
                audio = torch.from_numpy(waveform.mean(axis=1))
                if sample_rate != 16000:
                    audio = torchaudio.functional.resample(audio, sample_rate, 16000)
                inputs = processor(
                    audio.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
                )
                output = model(inputs.input_values.to(device))
                result[clip][key] = output.last_hidden_state.mean(dim=1)[0].cpu()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--clips-root", type=Path, default=DEFAULT_CLIPS_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoders", default="ast,panns,wav2vec2",
                        help="Comma-separated subset of ast,panns,wav2vec2")
    args = parser.parse_args()
    encoders = [name.strip() for name in args.encoders.split(",") if name.strip()]
    unknown = set(encoders) - {"ast", "panns", "wav2vec2"}
    if unknown or not encoders:
        parser.error(f"Unknown or empty encoder list: {sorted(unknown)}")
    device = pick_device(args.device)

    frames = extract_vit_features(
        csv_path=args.csv, clips=DEFAULT_CLIPS, batch_size=args.batch_size,
        device=device,
    )
    paths = audio_paths(DEFAULT_CLIPS, args.clips_root)
    extractors = {"ast": lambda: extract_ast(paths, device),
                  "panns": lambda: extract_panns(paths),
                  "wav2vec2": lambda: extract_wav2vec(paths, device)}
    for encoder in encoders:
        embeddings = extractors[encoder]()
        for clip in DEFAULT_CLIPS:
            pair = embeddings.get(clip, {})
            if "untouched" not in pair or "with_siren" not in pair:
                print(f"{encoder}/{clip}: missing audio pair")
                continue
            audio_u, audio_s = pair["untouched"], pair["with_siren"]
            vision_dim = frames[clip][0]["features"][-1]["cls"].numel()
            if audio_u.numel() != vision_dim or audio_s.numel() != vision_dim:
                print(f"{encoder}/{clip}: not comparable (vision={vision_dim}, "
                      f"audio={audio_u.numel()})")
                continue
            scores = [
                cosine_score(frame["features"][-1]["cls"], audio_u, audio_s)
                for frame in frames[clip]
            ]
            mean, std = score_stats(scores)
            print(f"{encoder}/{clip}: dim={audio_u.numel()} mean={mean:+.5f} std={std:.5f}")
    print("Note: matching dimensions do not imply aligned vision/audio embedding spaces.")


if __name__ == "__main__":
    main()
