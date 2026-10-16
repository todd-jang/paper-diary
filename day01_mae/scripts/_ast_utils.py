"""Shared audio preprocessing utilities for AST scripts."""

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


AUDIO_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TARGET_LEN = 1024


def load_logmel(
    path: str,
    sr: int = AUDIO_SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop: int = HOP,
    target_len: int = TARGET_LEN,
) -> torch.Tensor:
    """Load an audio file and return a padded or truncated log-mel spectrogram."""
    data, orig_sr = sf.read(str(path), always_2d=True)
    wav = torch.from_numpy(data.T).float()

    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    mel = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop,
    )(wav)
    log_mel = torch.log(mel + 1e-6)

    current_len = log_mel.shape[-1]
    if current_len < target_len:
        log_mel = F.pad(log_mel, (0, target_len - current_len))
    else:
        log_mel = log_mel[..., :target_len]

    return log_mel.squeeze(0)
