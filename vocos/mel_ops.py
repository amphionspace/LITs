import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn


_mel_basis_cache = {}
_hann_window_cache = {}


def hifigan_mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    fmin: float,
    fmax: float,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    """Mel spectrogram matching HIFIGAN/meldataset.py (center=False, reflect pad)."""
    device = y.device
    cache_key = f"{n_fft}_{n_mels}_{sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"
    if cache_key not in _mel_basis_cache:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[cache_key] = torch.from_numpy(mel).float().to(device)
        _hann_window_cache[cache_key] = torch.hann_window(win_length).to(device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        ((n_fft - hop_length) // 2, (n_fft - hop_length) // 2),
        mode="reflect",
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=_hann_window_cache[cache_key],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.abs(spec) + 1e-9
    spec = torch.matmul(_mel_basis_cache[cache_key], spec)
    return torch.log(torch.clamp(spec, min=clip_val))


class HiFiGANMelTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        hop_length: int = 384,
        win_length: int = 1536,
        n_mels: int = 100,
        fmin: float = 0.0,
        fmax: float = 12000.0,
        clip_val: float = 1e-5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return hifigan_mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.fmin,
            self.fmax,
            self.clip_val,
        )
