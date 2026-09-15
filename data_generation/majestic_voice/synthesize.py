"""Validate and atomically save generated 48/24 kHz audio."""
import io
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def save_audio(payload, row):
    wav, rate = sf.read(io.BytesIO(payload), dtype='float32')
    if rate != 48000 or wav.ndim != 1:
        raise ValueError(f'Unexpected audio format: rate={rate}, shape={wav.shape}')
    duration = len(wav) / rate
    if not 0.25 <= duration <= 120 or not np.isfinite(wav).all():
        raise ValueError(f'Invalid duration or samples: {duration:.3f}s')
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < 0.0001:
        raise ValueError(f'Nearly silent output: RMS={rms}')
    if duration > max(15, len(row['text']) * 1.0):
        raise ValueError(f'Unusually long output for text: {duration:.3f}s')
    min_seconds_per_character = 0.02 if row.get('language') == 'English' else 0.045
    if duration < max(0.25, len(row['text']) * min_seconds_per_character):
        raise ValueError(f'Unusually short output for text: {duration:.3f}s')
    for field, audio, sample_rate in (
        ('output_48k', wav, 48000),
        ('output_24k', resample_poly(wav, 1, 2), 24000),
    ):
        path = Path(row[field])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.wav.tmp')
        sf.write(temporary, audio, sample_rate, format='WAV', subtype='PCM_16')
        temporary.replace(path)
    return {'duration': duration, 'rms': rms, 'peak': float(np.abs(wav).max())}
