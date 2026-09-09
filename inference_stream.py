import datetime as dt
import inspect
from contextlib import contextmanager
from pathlib import Path
import os
import sys
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm
import argparse
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parent

from lits.infer_wav_merge import build_chunk_plan_index, load_chunk_manifest, merge_chunk_wavs
from lits.models.lits import LITS
from lits.models.components.utils import decoder_kv_cache_limit
from lits.runtime.text2id import EncodedPhonemes, read_jsonl
from lits.utils.model import denormalize
from vocos.vocoder import load_vocos_vocoder
_MEANFLOW_DISTILL_DIR = str(REPO_ROOT / "meanflow_distill")
if _MEANFLOW_DISTILL_DIR not in sys.path:
    sys.path.insert(0, _MEANFLOW_DISTILL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description=(
        "TTS inference with bidirectional full-utterance mu encoder precompute. "
        "Default: streaming ODE decoder with KV cache. "
        "Pass --non_streaming for full-utterance ODE decode."
    ),
)
parser.add_argument(
    '--model_lang',
    type=str,
    default='en-zh-dict',
    choices=[
        'en-zh', 'en-zh-dict',
        'en-zh-rhyme-body-tone', 'en-zh-dict-rhyme-body-tone',
    ],
    help='Model language pair; *-dict modes use English CMUdict G2P for Latin script; '
         'en-zh / en-zh-dict use rhyme-body-tone cleaners (n_vocab=173); '
         'en-zh is for pre-converted pinyin input',
)
parser.add_argument('--checkpoint', type=str, required=True, help='LITs checkpoint path')
parser.add_argument(
    '--input_txt',
    type=str,
    required=True,
    help='Input txt file: one synthesis utterance per line (plain text only, no wav name or spk_id)',
)
parser.add_argument(
    '--spk_id',
    type=int,
    required=True,
    help='Speaker id for every utterance (must be set explicitly; no auto-inference from text)',
)
parser.add_argument('--output_dir', type=str, required=True, help='Output audio folder')
parser.add_argument('--output_txt', type=str, required=True, help='Output txt file (each line: gen_audio_path|text)')
parser.add_argument(
    '--vocos_checkpoint',
    type=str,
    default=str(REPO_ROOT / 'vocos' / 'generator.ckpt'),
    help='Vocos generator checkpoint path (default: bundled 24 kHz checkpoint)',
)
parser.add_argument(
    '--vocos_root',
    type=str,
    default=str(REPO_ROOT),
    help='Path to repo root containing the vendored vocos package (default: this repo)',
)
parser.add_argument('--output_sample_rate', type=int, default=24000, help='Output wav sample rate')
parser.add_argument('--n_timesteps', type=int, default=10)
parser.add_argument('--t_grid', type=str, default=None,
                    help='Custom non-uniform time grid as comma-separated floats, e.g. "0,0.3,1.0" for 2 steps. '
                         'Overrides --n_timesteps. Must start at 0 and end at 1.')
parser.add_argument('--length_scale', type=float, default=1.0)
parser.add_argument('--temperature', type=float, default=0.667)
parser.add_argument(
    '--noise_seed',
    type=int,
    default=None,
    help='Fix ODE initial-noise RNG per sample (sample i uses noise_seed + i). '
         'Use when comparing n_timesteps so each step count shares the same z.',
)
parser.add_argument('--chunk_size', type=int, default=50, help='Streaming chunk size')
parser.add_argument('--mel_cache_len', type=int, default=8, help='Mel cache length for streaming')
parser.add_argument('--pre_lookahead_len', type=int, default=3, help='Pre-lookahead length for streaming')
parser.add_argument(
    '--always_context',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Last chunk passes embed-space zero context (concat path) instead of empty '
         'context (pad path). Numerically matches legacy; ONNX single-graph friendly.',
)
parser.add_argument(
    '--no_waveform_crossfade',
    action='store_true',
    help='Disable Hanning waveform crossfade between streaming vocoder chunks (ablation).',
)
parser.add_argument(
    '--vocoder_mode',
    type=str,
    default='chunked',
    choices=['chunked', 'full'],
    help='chunked=per-chunk Vocos (production streaming); full=accumulate mel then '
         'one Vocos pass (ablation: tests vocoder-boundary artifacts).',
)
_NUM_DECODING_LEFT_CHUNKS_HELP = (
    "Legacy left-chunk window for streaming Conformer encoder and ODE decoder attention. "
    "Defaults to -1 in inference_stream.py because decoder_left_frames "
    "is the intended decoder context control. "
    "When N>=0, each mel frame attends to at most the current chunk plus N chunks "
    "to the left; encoder KV and decoder per-step KV caches are physically trimmed to "
    "static_chunk_size×N mel frames (e.g. N=2 with static_chunk_size=50 → max 100 frames). "
    "Ignored by the ODE decoder when --decoder_left_frames or checkpoint decoder_left_frames is >=0. "
    "-1 keeps full history (caches grow with utterance length). "
    "0 restricts attention to the current chunk only. "
    "Verify cache growth: python monitor_decoder_kv_cache.py --num_decoding_left_chunks N"
)
parser.add_argument(
    '--num_decoding_left_chunks',
    type=int,
    default=-1,
    help=_NUM_DECODING_LEFT_CHUNKS_HELP,
)
parser.add_argument(
    '--decoder_left_frames',
    type=int,
    default=None,
    help='Frame-level left context for the ODE decoder. When set to >=0, overrides '
         '--num_decoding_left_chunks for decoder attention and decoder KV-cache trim. '
         'Default None keeps the value stored in the checkpoint/model config.',
)
parser.add_argument(
    '--encoder_num_decoding_left_chunks',
    type=int,
    default=None,
    help='How many previous chunks the mu encoder (ConformerEncoder) can attend to during '
         'streaming inference. Defaults to --num_decoding_left_chunks when not set. '
         'Set to -1 to give the encoder unlimited left context while keeping the decoder bounded.',
)
parser.add_argument(
    '--token_ids_jsonl',
    type=str,
    required=True,
    help='Numeric token/tone ID records prepared by the external frontend pipeline.',
)
parser.add_argument(
    '--frontend_n_vocab',
    type=int,
    required=True,
    help='Vocabulary size declared by the frontend token contract.',
)
parser.add_argument(
    '--fp16',
    action='store_true',
    help='Use CUDA autocast FP16 for LITs (Vocos stays FP32 for stability)',
)
parser.add_argument(
    '--infer_duration_patches',
    action='store_true',
    help='Apply inference-only duration patches (en/zh vowel floor & pause caps).',
)
parser.add_argument(
    '--limit',
    type=int,
    default=0,
    help='Only process first N input lines (0=all)',
)
parser.add_argument(
    '--non_streaming',
    action='store_true',
    help='Full-utterance ODE decode (no chunking, no crossfade). '
         'Use for quality comparison or when streaming window artifacts matter.',
)
parser.add_argument(
    '--chunk_manifest',
    type=str,
    default=None,
    help='TSV manifest from infer_e2e chunking; when set, merge each utterance after '
         'its last chunk is synthesized',
)
parser.add_argument(
    '--merged_output_dir',
    type=str,
    default=None,
    help='Directory for merged utterance wavs (required with --chunk_manifest)',
)
parser.add_argument(
    '--chunk_silence_ms',
    type=int,
    default=150,
    help='Silence inserted between merged text chunks (ms)',
)
parser.add_argument(
    '--keep_chunk_wavs',
    action='store_true',
    help='Keep temporary per-chunk wav files after merging each utterance',
)
args = parser.parse_args()

USE_FP16 = args.fp16 and torch.cuda.is_available()
if args.fp16 and not torch.cuda.is_available():
    print("[WARNING] --fp16 ignored: CUDA not available, using FP32")

print(f"[INFO] token_ids_jsonl={args.token_ids_jsonl} (model input; no text cleaner loaded)")

MODEL_LANG = args.model_lang
LITS_CHECKPOINT = args.checkpoint
OUTPUT_FOLDER = args.output_dir
N_TIMESTEPS = args.n_timesteps
LENGTH_SCALE = args.length_scale
TEMPERATURE = args.temperature
OUTPUT_SAMPLE_RATE = args.output_sample_rate


def _is_distilled_checkpoint(ckpt_data):
    """Detect IntMeanFlow-distilled student checkpoint format."""
    return (isinstance(ckpt_data, dict)
            and "metadata" in ckpt_data
            and "state_dict" in ckpt_data
            and any("interval_projector" in k for k in ckpt_data["state_dict"]))


def _read_distilled_student_t_grid(checkpoint_path: str) -> list[float] | None:
    """Return student_t_grid from distilled checkpoint metadata, if present."""
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not _is_distilled_checkpoint(ckpt_data):
        return None
    meta = ckpt_data.get("metadata") or {}
    grid = meta.get("student_t_grid")
    if grid is not None:
        return [float(x) for x in grid]
    student_steps = int(meta.get("student_steps", 0))
    if student_steps > 0:
        return [i / student_steps for i in range(student_steps + 1)]
    return None


CUSTOM_T_GRID = None
if args.t_grid is not None:
    CUSTOM_T_GRID = [float(x.strip()) for x in args.t_grid.split(',')]
    assert CUSTOM_T_GRID[0] == 0.0 and CUSTOM_T_GRID[-1] == 1.0, \
        f"--t_grid must start at 0 and end at 1, got {CUSTOM_T_GRID}"
    N_TIMESTEPS = len(CUSTOM_T_GRID) - 1
elif Path(LITS_CHECKPOINT).is_file():
    _ckpt_grid = _read_distilled_student_t_grid(LITS_CHECKPOINT)
    if _ckpt_grid is not None:
        CUSTOM_T_GRID = _ckpt_grid
        N_TIMESTEPS = len(CUSTOM_T_GRID) - 1

@contextmanager
def cuda_autocast():
    """FP16 matmul/conv via autocast; weights stay FP32 (avoids half() dtype clashes)."""
    if USE_FP16:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def _interval_conditioned_estimator_cls():
    try:
        from interval_estimator import IntervalConditionedEstimator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "meanflow_distill/interval_estimator.py is required for distilled "
            "student checkpoints. Pull the latest repo or copy that file into "
            "meanflow_distill/."
        ) from exc
    return IntervalConditionedEstimator


def _wrap_estimator_with_interval(model):
    """Wrap the decoder estimator with IntervalConditionedEstimator."""
    IntervalConditionedEstimator = _interval_conditioned_estimator_cls()
    model.decoder.estimator = IntervalConditionedEstimator(model.decoder.estimator)
    return model


def _filter_ctor_kwargs(model_cls, hparams: dict) -> dict:
    """Drop training-only hyper_parameters that are not LITS __init__ args."""
    allowed = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
    return {key: value for key, value in hparams.items() if key in allowed}


def load_model(checkpoint_path):
    ckpt_data = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if _is_distilled_checkpoint(ckpt_data):
        meta = ckpt_data["metadata"]
        teacher_ckpt = os.environ.get("TEACHER_CKPT", meta.get("teacher_ckpt", ""))
        if not teacher_ckpt or not Path(teacher_ckpt).exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_ckpt!r}. "
                "Set TEACHER_CKPT=/path/to/teacher.ckpt to override."
            )
        print(f"[distilled] Loading base model from teacher: {teacher_ckpt}")
        print(f"[distilled] student_steps={meta.get('student_steps')}, "
              f"teacher_steps={meta.get('teacher_steps')}")
        if meta.get("student_t_grid") is not None:
            print(f"[distilled] student_t_grid={meta.get('student_t_grid')}")
        model = LITS.load_from_checkpoint(teacher_ckpt, map_location=DEVICE)
        IntervalConditionedEstimator = _interval_conditioned_estimator_cls()
        model.decoder.estimator = IntervalConditionedEstimator(model.decoder.estimator).to(DEVICE)
        model.load_state_dict(ckpt_data["state_dict"])
        model.eval()
        return model.to(DEVICE)
    if isinstance(ckpt_data, dict) and "hyper_parameters" in ckpt_data and "state_dict" in ckpt_data:
        model = LITS(**_filter_ctor_kwargs(LITS, ckpt_data["hyper_parameters"]))
        model.load_state_dict(ckpt_data["state_dict"])
        model.eval()
        return model.to(DEVICE)
    model = LITS.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    model.eval()
    return model


print(f"Loading LITs checkpoint: {args.checkpoint} (device={DEVICE}) ...", flush=True)
model = load_model(LITS_CHECKPOINT)
model.apply_infer_duration_patches = args.infer_duration_patches
print(f"Model loaded successfully.")
if args.infer_duration_patches:
    print("[INFO] infer_duration_patches enabled (en/zh vowel floor & pause caps)")
else:
    print("[INFO] infer_duration_patches disabled (raw ceil durations)")
if USE_FP16:
    print("[INFO] FP16 inference enabled (CUDA autocast; LITs compute in FP16, weights FP32)")
print(f"Model stats: n_vocab={getattr(model, 'n_vocab', 'N/A')}, n_spks={getattr(model, 'n_spks', 'N/A')}")
if int(getattr(model, "n_vocab", 0)) != args.frontend_n_vocab:
    raise ValueError(
        f"Frontend token contract has n_vocab={args.frontend_n_vocab}, but checkpoint "
        f"has n_vocab={getattr(model, 'n_vocab', 'N/A')}"
    )
_model_has_tone_emb = getattr(getattr(model, "encoder", None), "tone_emb", None) is not None
if _model_has_tone_emb:
    raise ValueError(
        "This checkpoint uses the removed tone-embedding (toneless-rhyme) paradigm. "
        "Use a rhyme-body-tone checkpoint with model_lang en-zh or en-zh-dict."
    )
if MODEL_LANG.startswith("en-zh") and getattr(model, "n_tones", 0) > 0:
    raise ValueError(
        f"Checkpoint/mode mismatch: model_lang={MODEL_LANG!r} uses inline tone marks "
        "and expects encoder n_tones=0."
    )
if hasattr(model, "spk_emb"):
    print(f"Speaker embedding size: num_embeddings={model.spk_emb.num_embeddings}, dim={model.spk_emb.embedding_dim}")
if hasattr(model, "encoder") and hasattr(model.encoder, "emb"):
    print(f"Text embedding size: num_embeddings={model.encoder.emb.num_embeddings}, dim={model.encoder.emb.embedding_dim}")
if hasattr(model, "decoder") and hasattr(model.decoder, "num_decoding_left_chunks"):
    model.decoder.num_decoding_left_chunks = args.num_decoding_left_chunks
    if hasattr(model.decoder, "estimator"):
        model.decoder.estimator.num_decoding_left_chunks = args.num_decoding_left_chunks
    decoder_left_frames = args.decoder_left_frames
    if decoder_left_frames is None:
        decoder_left_frames = getattr(
            getattr(model.decoder, "estimator", None), "decoder_left_frames", -1,
        )
    else:
        model.decoder.decoder_left_frames = decoder_left_frames
        if hasattr(model.decoder, "estimator"):
            model.decoder.estimator.decoder_left_frames = decoder_left_frames
    enc_left = args.encoder_num_decoding_left_chunks
    if enc_left is None:
        enc_left = args.num_decoding_left_chunks
    if hasattr(model.decoder, "encoder_num_decoding_left_chunks"):
        model.decoder.encoder_num_decoding_left_chunks = enc_left
        if hasattr(model.decoder, "encoder"):
            model.decoder.encoder.num_decoding_left_chunks = enc_left
    static_chunk = getattr(model.decoder.estimator, "static_chunk_size", args.chunk_size)
    kv_limit = decoder_kv_cache_limit(
        static_chunk, args.num_decoding_left_chunks, decoder_left_frames,
    )
    kv_limit_str = str(kv_limit) if kv_limit >= 0 else "unlimited"
    enc_kv_limit = decoder_kv_cache_limit(static_chunk, enc_left)
    enc_kv_limit_str = str(enc_kv_limit) if enc_kv_limit >= 0 else "unlimited"
    print(
        "Streaming context window: "
        f"decoder num_decoding_left_chunks={model.decoder.num_decoding_left_chunks}, "
        f"decoder_left_frames={decoder_left_frames}, "
        f"encoder num_decoding_left_chunks={enc_left}, "
        f"static_chunk_size={static_chunk}, "
        f"decoder kv_cache_limit={kv_limit_str} mel frames, "
        f"encoder kv_cache_limit={enc_kv_limit_str} mel frames"
    )
    print(
        "Bidirectional mu encoder: enabled "
        "(full utterance mu_y is encoded once with streaming=False; "
        "encoder_num_decoding_left_chunks is ignored on the main path)"
    )
elif args.num_decoding_left_chunks != -1:
    print(
        "[WARNING] --num_decoding_left_chunks was set, but model.decoder does not expose "
        "num_decoding_left_chunks. The limit may not take effect."
    )


def setup_vocoder(vocos_checkpoint, vocos_root):
    print(f"Loading Vocos vocoder: {vocos_checkpoint} ...", flush=True)
    vocoder, vocoder_hparams = load_vocos_vocoder(vocos_checkpoint, DEVICE, vocos_root=vocos_root)
    vocoder_num_mels = int(vocoder_hparams.num_mels)
    vocoder_sample_rate = int(vocoder_hparams.sampling_rate)
    vocoder_output_hop = int(vocoder_hparams.hop_size)
    print(
        f"Vocoder loaded: Vocos ({vocos_checkpoint}, "
        f"n_mel={vocoder_num_mels}, hop={vocoder_output_hop}, sr={vocoder_sample_rate})"
    )
    return vocoder, vocoder_num_mels, vocoder_sample_rate, vocoder_output_hop


vocoder, vocoder_num_mels, vocoder_sample_rate, vocoder_output_hop = setup_vocoder(
    args.vocos_checkpoint,
    args.vocos_root,
)

if model.n_feats < vocoder_num_mels:
    raise ValueError(
        f"Mel channels mismatch: LITs outputs {model.n_feats} mel bins, "
        f"but vocoder expects {vocoder_num_mels}. "
        "Please use a vocoder checkpoint with matching mel bins."
    )
mel_trim_bins = vocoder_num_mels if model.n_feats > vocoder_num_mels else None
if mel_trim_bins is not None:
    print(
        f"[INFO] LITs outputs {model.n_feats} mel bands; "
        f"trimming to {mel_trim_bins} for vocoder."
    )
if OUTPUT_SAMPLE_RATE != vocoder_sample_rate:
    print(
        f"[WARNING] output_sample_rate={OUTPUT_SAMPLE_RATE} differs from vocoder config "
        f"sampling_rate={vocoder_sample_rate}. Consider setting --output_sample_rate {vocoder_sample_rate}."
    )


@torch.inference_mode()
def process_token_ids(encoded: EncodedPhonemes):
    x = torch.tensor(encoded.token_ids, dtype=torch.long, device=DEVICE)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=DEVICE)
    x_tones = None
    if encoded.tone_ids is not None:
        x_tones = torch.tensor(encoded.tone_ids, dtype=torch.long, device=DEVICE)[None]
    return {
        'x_orig': encoded.phonemes,
        'x': x,
        'x_lengths': x_lengths,
        'x_tones': x_tones,
    }


@torch.inference_mode()
def synthesise(*, spks=None, text_processed):
    start_t = dt.datetime.now()
    with cuda_autocast():
        output = model.get_hidden_mel(
            text_processed['x'],
            text_processed['x_lengths'],
            spks=spks,
            length_scale=LENGTH_SCALE,
            x_tones=text_processed.get('x_tones'),
        )
    output.update({'start_t': start_t, **text_processed})
    return output


def parse_input_line(line: str) -> str | None:
    """Return stripped synthesis text, or None for blank lines."""
    text = line.strip()
    return text or None


def _validate_token_range(x: torch.Tensor):
    if x.numel() == 0:
        raise ValueError("Empty token sequence after tokenization.")
    model_n_vocab = int(getattr(model, "n_vocab", 0))
    token_min = int(x.min().item())
    token_max = int(x.max().item())
    if token_min < 0 or token_max >= model_n_vocab:
        raise ValueError(
            f"Token id out of range: min={token_min}, max={token_max}, "
            f"but model.n_vocab={model_n_vocab}. "
            "Checkpoint and frontend tokenizer are likely mismatched."
        )


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel = fade_in_mel.cpu().clone()
    fade_out_mel = fade_out_mel.cpu()
    window = torch.as_tensor(window, dtype=fade_in_mel.dtype, device=fade_in_mel.device)
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel.to(device)


def emit_streaming_mel_chunk(
    mel_emit,
    *,
    is_last_chunk: bool,
    vocoder_cache: dict,
    speech_window,
    mel_cache_len: int,
    vocoder_mode: str,
    waveform_crossfade: bool,
    mel_parts: list,
):
    """Emit one streaming mel segment to waveform (or buffer for full vocoder)."""
    if vocoder_mode == 'full':
        mel_parts.append(mel_emit)
        if not is_last_chunk:
            return None
        full_mel = torch.cat(mel_parts, dim=2)
        return to_waveform(full_mel, vocoder)

    mel = mel_emit
    if vocoder_cache:
        mel = torch.concat([vocoder_cache['mel'], mel], dim=2)

    if not is_last_chunk:
        waveform = to_waveform(mel, vocoder).unsqueeze(0)
        if waveform_crossfade and 'waveform' in vocoder_cache:
            waveform = fade_in_out(waveform, vocoder_cache['waveform'], speech_window)
        mel = mel[:, :, -mel_cache_len:]
        vocoder_cache['mel'] = mel
        source_cache_len = int(mel_cache_len * vocoder_output_hop)
        vocoder_cache['waveform'] = waveform[:, -source_cache_len:]
        return waveform[:, :-source_cache_len][0]

    waveform = to_waveform(mel, vocoder).unsqueeze(0)
    if waveform_crossfade and 'waveform' in vocoder_cache:
        return fade_in_out(waveform, vocoder_cache['waveform'], speech_window)[0]
    return waveform[0]


@torch.inference_mode()
def to_waveform(mel, vocoder):
    if mel_trim_bins is not None:
        mel = mel[:, :mel_trim_bins, :]
    mel = mel.float()
    audio = vocoder(mel).clamp(-1, 1)
    target_len = int(mel.shape[-1] * vocoder_output_hop)
    audio = audio[..., :target_len]
    audio = audio.squeeze(0)
    return audio.cpu().squeeze()


def save_to_folder(filename: str, waveform: torch.Tensor, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    sf.write(folder / f'{filename}.wav', waveform.cpu().numpy(), OUTPUT_SAMPLE_RATE, 'PCM_24')


def _solve_euler_window(estimator, z, mu, mask, spks, n_timesteps, device,
                        streaming=False):
    """Standalone Euler ODE solver on a single window (no KV cache).

    Uses streaming=False by default so each window gets full bidirectional
    attention, avoiding chunk-boundary misalignment that occurs when the
    causal chunk mask is applied to windows of varying start positions.

    Supports IntMeanFlow interval-conditioned estimators (passes r= kwarg
    for start time when the estimator accepts it).
    """
    if CUSTOM_T_GRID is not None:
        t_span = torch.tensor(CUSTOM_T_GRID, device=device)
    else:
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
    x = z
    interval_conditioned = hasattr(estimator, 'interval_projector')
    for step in range(len(t_span) - 1):
        r = t_span[step]
        t = t_span[step + 1]
        dt = t - r
        if interval_conditioned:
            dphi_dt = estimator(x, mask, mu, t, spks, None, streaming, r=r)
        else:
            dphi_dt = estimator(x, mask, mu, r, spks, None, streaming)
        x = x + dt * dphi_dt
    return x


@torch.inference_mode()
def streaming_synthesise(
    text,
    chunk_size,
    mel_cache_len,
    pre_lookahead_len,
    speech_window,
    spks=None,
    always_context=True,
    noise_seed=None,
    vocoder_mode='chunked',
    waveform_crossfade=True,
    encoded=None,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    if encoded is None:
        raise ValueError("streaming_synthesise requires precomputed token IDs")
    text_processed = process_token_ids(encoded)
    _validate_token_range(text_processed["x"])
    output = synthesise(spks=spks, text_processed=text_processed)

    first_latency = None
    vocoder_cache = {}
    mel_parts = []

    pad = output['y_max_length'] % chunk_size
    slice_id = range(0, output['y_max_length']-pad, chunk_size) if output['y_max_length']-pad > 0 else [0]
    all_waveforms = []

    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    global_z = torch.randn(
        1, model.n_feats, output['y_max_length'],
        device=DEVICE,
    ) * TEMPERATURE

    if hasattr(model.decoder, "reset_encoder_cache"):
        model.decoder.reset_encoder_cache()

    decoder = model.decoder
    spks_emb = output['spks'] if 'spks' in output else None
    with cuda_autocast():
        full_mu_enc = decoder.encoder(
            output['mu_y'],
            output['y_mask'],
            streaming=False,
        )

    for j, start_idx in enumerate(slice_id):
        is_last_chunk = start_idx == slice_id[-1]

        if not is_last_chunk:
            end_idx = start_idx + chunk_size + pre_lookahead_len
        else:
            end_idx = output['y_max_length']

        if is_last_chunk:
            y_mask = output['y_mask'][:, :, :end_idx]
        else:
            y_mask = output['y_mask'][:, :, :end_idx - pre_lookahead_len]

        with cuda_autocast():
            mu_enc = full_mu_enc[:, :, :y_mask.size(-1)]
            z = global_z[:, :, :mu_enc.shape[-1]]
            if CUSTOM_T_GRID is not None:
                t_span = torch.tensor(CUSTOM_T_GRID, device=DEVICE)
            else:
                t_span = torch.linspace(0, 1, N_TIMESTEPS + 1, device=DEVICE)
            decoder_outputs = decoder.solve_euler(
                z,
                t_span=t_span,
                mu=mu_enc,
                mask=y_mask,
                spks=spks_emb,
                cond=None,
                streaming=True,
                chunk_start=start_idx,
            )
        mel = denormalize(decoder_outputs.float(), model.mel_mean, model.mel_std)
        mel = mel[0, :, start_idx:].unsqueeze(0)

        if first_latency is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            first_latency = (time.perf_counter() - start_time) * 1000

        waveform = emit_streaming_mel_chunk(
            mel,
            is_last_chunk=is_last_chunk,
            vocoder_cache=vocoder_cache,
            speech_window=speech_window,
            mel_cache_len=mel_cache_len,
            vocoder_mode=vocoder_mode,
            waveform_crossfade=waveform_crossfade,
            mel_parts=mel_parts,
        )
        if waveform is not None:
            all_waveforms.append(waveform.squeeze())

    final_waveform = torch.cat(all_waveforms, dim=0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    rtf = total_time * OUTPUT_SAMPLE_RATE / final_waveform.shape[-1]

    return {
        'waveform': final_waveform,
        'rtf': rtf,
        'first_latency': first_latency,
        'start_t': output['start_t'],
        'mel_frames': int(output['y_max_length']),
        'wav_samples': int(final_waveform.shape[-1]),
    }


@torch.inference_mode()
def non_streaming_synthesise(
    text,
    spks=None,
    noise_seed=None,
    encoded=None,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    if encoded is None:
        raise ValueError("non_streaming_synthesise requires precomputed token IDs")
    text_processed = process_token_ids(encoded)
    _validate_token_range(text_processed["x"])
    output = synthesise(spks=spks, text_processed=text_processed)
    y_len = int(output['y_max_length'])

    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    global_z = torch.randn(
        1, model.n_feats, y_len,
        device=DEVICE,
    ) * TEMPERATURE

    decoder = model.decoder
    spks_emb = output['spks'] if 'spks' in output else None
    y_mask = output['y_mask'][:, :, :y_len]
    with cuda_autocast():
        full_mu_enc = decoder.encoder(
            output['mu_y'],
            output['y_mask'],
            streaming=False,
        )
        full_mu_enc = full_mu_enc[:, :, :y_len]
        ode_out = _solve_euler_window(
            decoder.estimator, global_z, full_mu_enc, y_mask,
            spks_emb, N_TIMESTEPS, DEVICE,
        )
    mel = denormalize(ode_out.float(), model.mel_mean, model.mel_std)
    waveform = to_waveform(mel, vocoder)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    rtf = total_time * OUTPUT_SAMPLE_RATE / waveform.shape[-1]

    return {
        'waveform': waveform,
        'rtf': rtf,
        'first_latency': None,
        'start_t': output['start_t'],
        'mel_frames': y_len,
        'wav_samples': int(waveform.shape[-1]),
    }


def batch_inference(input_txt_path, output_dir, output_txt_path=None):
    chunk_size = args.chunk_size
    mel_cache_len = args.mel_cache_len
    source_cache_len = int(mel_cache_len * vocoder_output_hop)
    speech_window = np.hanning(2 * source_cache_len)
    pre_lookahead_len = args.pre_lookahead_len

    rtfs = []
    rtfs_w = []
    first_latency_list = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt_lines = []

    token_rows = read_jsonl(args.token_ids_jsonl)
    print(f"[INFO] Loaded {len(token_rows)} token ID record(s) from {args.token_ids_jsonl}")

    chunk_by_synth_line: dict[int, object] = {}
    chunk_by_orig_line: dict[int, list] = {}
    merged_output_dir: Path | None = None
    delete_chunk_wavs = not args.keep_chunk_wavs
    if args.chunk_manifest:
        if not args.merged_output_dir:
            raise ValueError("--merged_output_dir is required when --chunk_manifest is set")
        chunk_plan = load_chunk_manifest(Path(args.chunk_manifest))
        chunk_by_synth_line, chunk_by_orig_line = build_chunk_plan_index(chunk_plan)
        merged_output_dir = Path(args.merged_output_dir)
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[chunk_merge] enabled: manifest={args.chunk_manifest} "
            f"merged_output_dir={merged_output_dir} "
            f"silence_ms={args.chunk_silence_ms} keep_chunk_wavs={args.keep_chunk_wavs}"
        )

    with open(input_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if args.limit > 0:
        lines = lines[: args.limit]
        print(f"Processing first {len(lines)} line(s) (--limit {args.limit})")
    expected_records = sum(1 for line in lines if parse_input_line(line) is not None)
    if len(token_rows) != expected_records:
        raise ValueError(
            f"token_ids_jsonl record count ({len(token_rows)}) does not match "
            f"non-empty input utterances ({expected_records})"
        )

    spk_id = args.spk_id
    spk_upper_bound = None
    if getattr(model, "n_spks", 1) > 1:
        if hasattr(model, "spk_emb"):
            spk_upper_bound = int(model.spk_emb.num_embeddings) - 1
        else:
            spk_upper_bound = int(model.n_spks) - 1
        if spk_upper_bound is None:
            raise ValueError("Model indicates multi-speaker but speaker embedding is missing.")
        if not (0 <= spk_id <= spk_upper_bound):
            raise ValueError(
                f"--spk_id out of range: {spk_id}, expected [0, {spk_upper_bound}] "
                f"(model.n_spks={model.n_spks})"
            )
    print(f"Using speaker id: {spk_id}")

    spks = None
    if model.n_spks > 1:
        spks = torch.tensor([spk_id], device=DEVICE, dtype=torch.long)

    sample_count = 0
    for i, line in enumerate(tqdm(lines)):
        try:
            text = parse_input_line(line)
            if text is None:
                continue
            sample_count += 1
            base_name = str(sample_count)

            if sample_count > len(token_rows):
                raise ValueError(
                    f"token_ids_jsonl has fewer records ({len(token_rows)}) than "
                    f"input_txt utterances ({sample_count})"
                )
            encoded = token_rows[sample_count - 1]

            sample_noise_seed = (
                args.noise_seed + sample_count - 1 if args.noise_seed is not None else None
            )
            if args.non_streaming:
                result = non_streaming_synthesise(
                    text,
                    spks=spks,
                    noise_seed=sample_noise_seed,
                    encoded=encoded,
                )
            else:
                result = streaming_synthesise(
                    text,
                    chunk_size,
                    mel_cache_len,
                    pre_lookahead_len,
                    speech_window,
                    spks=spks,
                    always_context=args.always_context,
                    noise_seed=sample_noise_seed,
                    vocoder_mode=args.vocoder_mode,
                    waveform_crossfade=not args.no_waveform_crossfade,
                    encoded=encoded,
                )

            t = (dt.datetime.now() - result['start_t']).total_seconds()
            rtf_w = t * OUTPUT_SAMPLE_RATE / result['waveform'].shape[-1]

            rtfs.append(result['rtf'])
            rtfs_w.append(rtf_w)
            first_latency_list.append(result['first_latency'])

            save_to_folder(base_name, result['waveform'], output_dir)
            out_wav_path = str(output_dir / f"{base_name}.wav")
            chunk_entry = chunk_by_synth_line.get(sample_count)
            if chunk_entry is None:
                output_txt_lines.append(f"{out_wav_path}|{text}")
            print(
                f"[DEBUG_LENGTH] {base_name}: mel_frames={result['mel_frames']}, "
                f"wav_samples={result['wav_samples']}, "
                f"wav_sec={result['wav_samples'] / OUTPUT_SAMPLE_RATE:.3f}"
            )
            print(f"[{sample_count}] Synthesized and saved: {base_name}.wav")

            if chunk_entry is not None and chunk_entry.chunk_idx == chunk_entry.n_chunks:
                line_entries = chunk_by_orig_line[chunk_entry.orig_line]
                line_meta = merge_chunk_wavs(
                    line_entries,
                    output_dir,
                    merged_output_dir,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    silence_ms=args.chunk_silence_ms,
                    delete_chunks=delete_chunk_wavs,
                )
                output_txt_lines.extend(line_meta)
                out_merged = merged_output_dir / f"{chunk_entry.orig_line}.wav"
                print(
                    f"[chunk_merge] line {chunk_entry.orig_line}/{len(chunk_by_orig_line)} "
                    f"merged {len(line_entries)} chunk(s) -> {out_merged}"
                )

        except Exception as e:
            print(f"[line {i+1}] Error processing: {line.strip()}")
            print(e)
            traceback.print_exc()
            if torch.cuda.is_available() and "device-side assert triggered" in str(e):
                print("CUDA context is now invalid after device-side assert. Stop early and fix the first failing sample.")
                break

    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for l in output_txt_lines:
                f.write(l + "\n")
        print(f"Output list saved to: {output_txt_path}")

    print(f"Number of ODE steps: {N_TIMESTEPS}")
    if CUSTOM_T_GRID is not None:
        print(f"Custom t_grid: {CUSTOM_T_GRID}")
    if args.noise_seed is not None:
        print(f"Noise seed: {args.noise_seed} (sample i -> seed {args.noise_seed} + i)")
    if args.non_streaming:
        print("Mode: NON-STREAMING (full utterance ODE decode; chunk/overlap settings ignored)")
    else:
        print(f"Chunk size: {chunk_size}")
    static_chunk = getattr(
        getattr(model.decoder, "estimator", None), "static_chunk_size", chunk_size
    )
    decoder_left_frames = getattr(
        getattr(model.decoder, "estimator", None), "decoder_left_frames", -1,
    )
    kv_limit = decoder_kv_cache_limit(
        static_chunk, args.num_decoding_left_chunks, decoder_left_frames,
    )
    kv_limit_str = str(kv_limit) if kv_limit >= 0 else "unlimited"
    print(
        f"Decoder left context: chunks={args.num_decoding_left_chunks}, "
        f"frames={decoder_left_frames} "
        f"(kv_cache_limit={kv_limit_str} at static_chunk_size={static_chunk})"
    )
    print(f"Always context (last chunk concat path): {args.always_context}")
    print(f"Precision: {'fp16 autocast' if USE_FP16 else 'fp32'}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")
    if first_latency_list:
        latency_vals = [x for x in first_latency_list if x is not None]
        if latency_vals:
            print(f"First Latency: {sum(latency_vals)/len(latency_vals):.2f} ms")


if __name__ == "__main__":
    batch_inference(args.input_txt, args.output_dir, args.output_txt)
