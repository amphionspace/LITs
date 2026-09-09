#!/bin/bash
# E2E inference: C++ TN/G2P -> JSON Text2Id -> LITs -> Vocos
#
# Supported model_lang:
#   en-zh-dict                 — mixed zh/en text, dict G2P (E2E default)
#   en-zh                      — mixed zh/en pre-converted pinyin
#   en-zh-dict-rhyme-body-tone — alias for en-zh-dict
#   en-zh-rhyme-body-tone      — alias for en-zh
#
# Usage:
#   CKPT=/path/to/model.ckpt SPK_ID=<n> bash infer_e2e.sh <model_lang> <input_txt> [infer_id]
#
# Example:
#   SPK_ID=0 bash infer_e2e.sh en-zh-dict data/test.txt smoke_enzh
#
# Required env:
#   SPK_ID                     Speaker id
#   CKPT or CHECKPOINT         Model checkpoint path
#   VOCOS_CHECKPOINT           Override bundled 24 kHz Vocos checkpoint
#
# Optional env:
#   OUTPUT_DIR=/path/to/wavs  Override inference output directory
#   FP16=1                    CUDA autocast FP16 during synthesis (default: 1)
#   INFER_DURATION_PATCHES=1  Apply en/zh vowel floor & zh pause caps
#   PREPEND_SIL=0              Disable leading <sil> for legacy checkpoints
#   SKIP_TN_PREFLIGHT=1       Skip TN/ICU preflight (not recommended)
#   NON_STREAMING=1           Full-utterance ODE decode (no chunking/crossfade)
#   VOCODER_MODE=chunked|full chunked=per-chunk Vocos; full=one Vocos pass (ablation)
#   NO_WAVEFORM_CROSSFADE=1   Disable Hanning waveform crossfade (ablation)
#   T_GRID=0,0.6875,1.0       Custom ODE time grid (VSFM coasting for N=2)
#   MAX_INFER_TOKENS=256       Slice the single-pass frontend IDs to this final model budget (0=disable)
#   CHUNK_SILENCE_MS=150       Silence between merged chunk wavs (ms)
#   PYTHON=/path/to/python    Override Python (default: python from PATH)

# macOS still ships Bash 3.2, where expanding an empty array under `set -u`
# raises "unbound variable". Required scalar inputs are validated explicitly below.
set -eo pipefail

die() {
  echo "[infer_e2e] ERROR: $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  die "Python not found in PATH; activate your env or set PYTHON=/path/to/python"
fi

# shellcheck source=verify_e2e_tn.sh
source "$REPO_ROOT/verify_e2e_tn.sh"

for icu_root in \
  "${E2E_ICU_ROOT:-}" \
  "${HOME}/.local/icu"; do
  if [[ -n "$icu_root" && -f "${icu_root}/lib/libicui18n.so.78" ]]; then
    export ICU_ROOT="$icu_root"
    export E2E_ICU_LIB_DIR="${icu_root}/lib"
    break
  fi
done

OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-24000}"
N_TIMESTEPS="${N_TIMESTEPS:-10}"
T_GRID="${T_GRID:-}"
LENGTH_SCALE="${LENGTH_SCALE:-1}"
TEMPERATURE="${TEMPERATURE:-0.667}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MEL_CACHE_LEN="${MEL_CACHE_LEN:-8}"
PRE_LOOKAHEAD_LEN="${PRE_LOOKAHEAD_LEN:-3}"
DECODER_LEFT_FRAMES="${DECODER_LEFT_FRAMES:-20}"
NOISE_SEED="${NOISE_SEED:-1}"
FP16="${FP16:-1}"
INFER_DURATION_PATCHES="${INFER_DURATION_PATCHES:-0}"
PREPEND_SIL="${PREPEND_SIL:-1}"
MAX_INFER_TOKENS="${MAX_INFER_TOKENS:-256}"
CHUNK_SILENCE_MS="${CHUNK_SILENCE_MS:-150}"

SPK_ID="${SPK_ID:?SPK_ID is required (e.g. SPK_ID=0 CKPT=/path/to/model.ckpt $0 <model_lang> <input_txt> [infer_id])}"
MODEL_LANG="${1:?Usage: SPK_ID=N $0 <model_lang> <input_txt> [infer_id]}"
INPUT_TXT="${2:?Usage: SPK_ID=N $0 <model_lang> <input_txt> [infer_id]}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
INFER_ID="${3:-e2e_${MODEL_LANG}_${TIMESTAMP}}"

case "$MODEL_LANG" in
  en-zh|en-zh-dict|en-zh-rhyme-body-tone|en-zh-dict-rhyme-body-tone) ;;
  *) die "Unsupported model_lang: $MODEL_LANG (this branch is en-zh C++ frontend only)" ;;
esac

CKPT="${CKPT:-${CHECKPOINT:?CKPT or CHECKPOINT is required}}"
VOCOS_CHECKPOINT="${VOCOS_CHECKPOINT:-$REPO_ROOT/vocos/generator.ckpt}"
[[ -f "$VOCOS_CHECKPOINT" ]] || die "Vocos checkpoint not found: $VOCOS_CHECKPOINT (run git lfs pull --include=vocos/generator.ckpt)"

[[ -f "$INPUT_TXT" ]] || die "Input not found: $INPUT_TXT"
[[ -f "$CKPT" ]] || die "Checkpoint not found: $CKPT"
[[ -f "$REPO_ROOT/inference_stream.py" ]] || die "Missing inference_stream.py"

echo "[infer_e2e] C++ TN/G2P + JSON Text2Id mode=$MODEL_LANG spk_id=$SPK_ID ckpt=$CKPT input=$INPUT_TXT python=$PYTHON_BIN" >&2

verify_tn_preflight "$MODEL_LANG"

OUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/infer_output/$INFER_ID}"
mkdir -p "$OUT_DIR"

FP16_ARGS=()
[[ "${FP16:-}" == 1 ]] && FP16_ARGS=(--fp16)

DURATION_PATCH_ARGS=()
[[ "${INFER_DURATION_PATCHES:-}" == 1 ]] && DURATION_PATCH_ARGS=(--infer_duration_patches)

SIL_ARGS=()
[[ "$PREPEND_SIL" == 0 ]] && SIL_ARGS=(--no_prepend_sil)

DECODER_LEFT_FRAME_ARGS=()
if [[ -n "$DECODER_LEFT_FRAMES" ]]; then
  DECODER_LEFT_FRAME_ARGS=(--decoder_left_frames "$DECODER_LEFT_FRAMES")
fi

NON_STREAMING_ARGS=()
if [[ "${NON_STREAMING:-0}" == 1 ]]; then
  NON_STREAMING_ARGS=(--non_streaming)
fi

VOCODER_MODE="${VOCODER_MODE:-chunked}"
VOCODER_MODE_ARGS=()
if [[ "$VOCODER_MODE" != "chunked" ]]; then
  VOCODER_MODE_ARGS=(--vocoder_mode "$VOCODER_MODE")
fi

WAVEFORM_CROSSFADE_ARGS=()
if [[ "${NO_WAVEFORM_CROSSFADE:-0}" == 1 ]]; then
  WAVEFORM_CROSSFADE_ARGS=(--no_waveform_crossfade)
fi

NOISE_SEED_ARGS=()
if [[ -n "$NOISE_SEED" ]]; then
  NOISE_SEED_ARGS=(--noise_seed "$NOISE_SEED")
fi

T_GRID_ARGS=()
if [[ -n "$T_GRID" ]]; then
  T_GRID_ARGS=(--t_grid "$T_GRID")
fi

CHUNK_ARGS=()
if [[ "${MAX_INFER_TOKENS:-0}" -gt 0 ]]; then
  CHUNK_ARGS=(--max_infer_tokens "$MAX_INFER_TOKENS" --chunk_silence_ms "$CHUNK_SILENCE_MS")
fi

export PYTHONUNBUFFERED=1
if [[ -n "${E2E_ICU_LIB_DIR:-}" ]]; then
  export E2E_ICU_LIB_DIR
elif [[ -n "${ICU_ROOT:-}" ]]; then
  export E2E_ICU_LIB_DIR="${ICU_ROOT}/lib"
fi

"$PYTHON_BIN" "$REPO_ROOT/infer_e2e.py" \
  --model_lang "$MODEL_LANG" \
  --inference_script "$REPO_ROOT/inference_stream.py" \
  --checkpoint "$CKPT" \
  --vocos_checkpoint "$VOCOS_CHECKPOINT" \
  --input_txt "$INPUT_TXT" \
  --spk_id "$SPK_ID" \
  --output_dir "$OUT_DIR" \
  --output_txt "$OUT_DIR/meta.txt" \
  --keep_normalized \
  --output_sample_rate "$OUTPUT_SAMPLE_RATE" \
  --n_timesteps "$N_TIMESTEPS" \
  "${T_GRID_ARGS[@]}" \
  --length_scale "$LENGTH_SCALE" \
  --temperature "$TEMPERATURE" \
  --chunk_size "$CHUNK_SIZE" \
  --mel_cache_len "$MEL_CACHE_LEN" \
  --pre_lookahead_len "$PRE_LOOKAHEAD_LEN" \
  --num_decoding_left_chunks -1 \
  "${DECODER_LEFT_FRAME_ARGS[@]}" \
  "${NOISE_SEED_ARGS[@]}" \
  "${NON_STREAMING_ARGS[@]}" \
  "${VOCODER_MODE_ARGS[@]}" \
  "${WAVEFORM_CROSSFADE_ARGS[@]}" \
  "${FP16_ARGS[@]}" \
  "${DURATION_PATCH_ARGS[@]}" \
  "${SIL_ARGS[@]}" \
  "${CHUNK_ARGS[@]}"
