"""Full-text conditioning, chunked ODE KV caches and chunked Vocos evaluation.

Uses the same chunk boundaries and waveform overlap as inference_stream.py.
"""
import torch

from lits.utils.model import denormalize
from meanflow_distill.kv_cache_distill import compute_chunk_starts


def decoder_chunks(model, mu, mask, speaker, z, grid, chunk_size=100):
    """Yield new frames with one persistent KV/conv cache per ODE step."""
    decoder = model.decoder
    decoder._decoder_caches = None
    starts = compute_chunk_starts(mu.shape[-1], chunk_size)
    span = torch.tensor(grid, device=z.device, dtype=z.dtype)
    try:
        for index, start in enumerate(starts):
            end = mu.shape[-1] if index == len(starts) - 1 else start + chunk_size
            result = decoder.solve_euler(z[..., :end], span, mu[..., :end],
                mask[..., :end], speaker, None, streaming=True,
                chunk_start=start, use_kv_cache=True)
            yield result[..., start:end]
    finally:
        decoder._decoder_caches = None


@torch.inference_mode()
def synthesize_streaming(model, vocoder, ids, lengths, speaker, tones, grid,
                         temperature=1., chunk_size=100, mel_cache_len=8):
    hidden = model.get_hidden_mel(ids, lengths, speaker, x_tones=tones)
    frames = int(hidden['y_max_length'])
    assert 1 <= frames <= 7500, 'Predicted duration outside 0..120 seconds'
    mu = model.decoder.encoder(hidden['mu_y'], hidden['y_mask'], streaming=False)
    mu, mask = mu[..., :frames], hidden['y_mask'][..., :frames]
    z = torch.randn(1, model.n_feats, frames, device=ids.device) * temperature
    mel_parts, wave_parts = [], []
    cache_mel = cache_wave = None
    overlap = mel_cache_len * 384
    window = torch.hann_window(2 * overlap, periodic=False, device=ids.device)
    starts = compute_chunk_starts(frames, chunk_size)
    for index, output in enumerate(decoder_chunks(model, mu, mask, hidden['spks'], z, grid, chunk_size)):
        mel = denormalize(output.float(), model.mel_mean, model.mel_std)
        assert torch.isfinite(mel).all(), 'Nonfinite predicted Mel'
        mel_parts.append(mel)
        block = torch.cat([cache_mel, mel], dim=-1) if cache_mel is not None else mel
        wave = vocoder(block).clamp(-1, 1).reshape(-1)[:block.shape[-1] * 384]
        if cache_wave is not None:
            wave[:overlap] = wave[:overlap] * window[:overlap] + cache_wave * window[overlap:]
        if index != len(starts) - 1:
            cache_mel, cache_wave = block[..., -mel_cache_len:], wave[-overlap:]
            wave_parts.append(wave[:-overlap])
        else:
            wave_parts.append(wave)
    audio = torch.cat(wave_parts)
    assert audio.numel() == frames * 384
    return dict(audio=audio, mel=torch.cat(mel_parts, dim=-1), mel_frames=frames)
