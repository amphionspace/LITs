import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

def _np_log_clip(x, C, clip_val):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def _torch_log_clip(x, C, clip_val):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def _torch_exp(x, C):
    return torch.exp(x) / C

def _get_mel_basis_and_window(fmax, y_device, n_fft, num_mels, sampling_rate, win_size):
    key = f"{str(fmax)}_{str(y_device)}"
    if key not in _get_mel_basis_and_window.cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=0, fmax=fmax)
        _get_mel_basis_and_window.cache[key] = (
            torch.from_numpy(mel).float().to(y_device),
            torch.hann_window(win_size).to(y_device)
        )
    return _get_mel_basis_and_window.cache[key]
_get_mel_basis_and_window.cache = {}

MAX_WAV_VALUE = 32768.0

def wav_read(path):
    sr, data = read(path)
    return data, sr

def drc_np(x, C=1, clip_val=1e-5):
    return _np_log_clip(x, C, clip_val)

def drd_np(x, C=1):
    return np.exp(x) / C

def drc_torch(x, C=1, clip_val=1e-5):
    return _torch_log_clip(x, C, clip_val)

def drd_torch(x, C=1):
    return _torch_exp(x, C)

def spec_norm_torch(mag):
    return drc_torch(mag)

def spec_denorm_torch(mag):
    return drd_torch(mag)

def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))
    mel_basis, hann_window = _get_mel_basis_and_window(fmax, y.device, n_fft, num_mels, sampling_rate, win_size)
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis, spec)
    spec = spec_norm_torch(spec)
    return spec
