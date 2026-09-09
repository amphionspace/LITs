#!/bin/bash
# Inference with IntMeanFlow-distilled 2-step student.
#
# Streaming modes (STREAMING_MODE):
#   causal_kv     — causal KV-cache streaming (default)
#   non_streaming — full-utterance ODE (fastest offline, no first-packet latency)
#
# Usage:
#   STUDENT_CKPT=/path/to/student_step_NNNNNNN.pt \
#     SPK_ID=0 bash meanflow_distill/infer_distilled.sh en-zh-dict input.txt [output_id]
#
# Examples:
#   # Default: causal KV streaming
#   STUDENT_CKPT=... SPK_ID=0 bash meanflow_distill/infer_distilled.sh en-zh-dict input.txt out
#
#   # Offline / fastest
#   STREAMING_MODE=non_streaming STUDENT_CKPT=... SPK_ID=0 bash meanflow_distill/infer_distilled.sh ...
#
# Audio is written to meanflow_distill/infer_output/<output_id>/ by default.
#
# Optional:
#   TEACHER_CKPT=/path/to/teacher.ckpt   Override teacher (default: read from student metadata)
#   OUTPUT_DIR=/path/to/wavs             Override default output directory
#   T_GRID=0,0.6875,1.0                  Override ODE grid (default: read from student ckpt metadata)
#   T_GRID_PROFILE=coasting              Shortcut for T_GRID=0,0.6875,1.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse optional mode flags before positional args.
STREAMING_MODE="${STREAMING_MODE:-causal_kv}"
if [[ "${NON_STREAMING:-0}" == 1 ]]; then
  STREAMING_MODE="non_streaming"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --non-streaming|--non_streaming)
      STREAMING_MODE="non_streaming"
      shift
      ;;
    --streaming|--causal|--causal-kv|--causal_kv)
      STREAMING_MODE="causal_kv"
      shift
      ;;
    -*)
      echo "[infer_distilled] ERROR: unknown option: $1" >&2
      echo "Usage: STUDENT_CKPT=... SPK_ID=N $0 [--causal-kv|--non-streaming] <mode> <input_txt> [output_id]" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

MODEL_LANG="${1:?Usage: STUDENT_CKPT=... SPK_ID=N $0 [--causal-kv|--non-streaming] <mode> <input_txt> [output_id]}"
INPUT_TXT="${2:?Usage: STUDENT_CKPT=... SPK_ID=N $0 [--causal-kv|--non-streaming] <mode> <input_txt> [output_id]}"
if [[ $# -ge 3 ]]; then
  INFER_ID="$3"
else
  TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
  INFER_ID="distilled_${MODEL_LANG}_${TIMESTAMP}"
fi

export N_TIMESTEPS="${N_TIMESTEPS:-2}"
export CHUNK_SIZE="${CHUNK_SIZE:-100}"
export DECODER_LEFT_FRAMES="${DECODER_LEFT_FRAMES:-20}"
export PRE_LOOKAHEAD_LEN="${PRE_LOOKAHEAD_LEN:-3}"

case "$STREAMING_MODE" in
  causal_kv) ;;
  non_streaming)
    export NON_STREAMING=1
    ;;
  *)
    echo "[infer_distilled] ERROR: STREAMING_MODE must be causal_kv or non_streaming (got: $STREAMING_MODE)" >&2
    exit 1
    ;;
esac

export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/infer_output/$INFER_ID}"

if [[ -z "${T_GRID:-}" && "${T_GRID_PROFILE:-}" == "coasting" ]]; then
  export T_GRID="0,0.6875,1.0"
fi

STUDENT_CKPT="${STUDENT_CKPT:?STUDENT_CKPT is required (path to distilled student checkpoint)}"
export CHECKPOINT="$STUDENT_CKPT"

if [[ -n "${TEACHER_CKPT:-}" ]]; then
    export TEACHER_CKPT
fi

echo "[infer_distilled] mode=$STREAMING_MODE n_timesteps=$N_TIMESTEPS output=$OUTPUT_DIR" >&2
if [[ -n "${T_GRID:-}" ]]; then
  echo "[infer_distilled] t_grid=$T_GRID (explicit override)" >&2
else
  echo "[infer_distilled] t_grid: auto from student checkpoint metadata (or uniform if absent)" >&2
fi
case "$STREAMING_MODE" in
  causal_kv)
    echo "[infer_distilled] causal_kv: chunk_size=$CHUNK_SIZE decoder_left_frames=$DECODER_LEFT_FRAMES (KV-cache streaming)" >&2
    ;;
  non_streaming)
    echo "[infer_distilled] non_streaming: full-utterance ODE decode" >&2
    ;;
esac

exec bash "$REPO_ROOT/infer_e2e.sh" "$MODEL_LANG" "$INPUT_TXT" "$INFER_ID"
