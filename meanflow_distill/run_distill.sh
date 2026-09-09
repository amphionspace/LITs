#!/bin/bash
# Launch IntMeanFlow 2-step distillation.
#
# Edit meanflow_distill/configs/en-zh.yaml for defaults.
# Environment variables still override individual settings.
#
# Usage:
#   bash run_distill.sh <config> [output_suffix]
#   CUDA_VISIBLE_DEVICES=0,1 bash run_distill.sh <config>   # env overrides yaml
#   CONFIG=<config> bash run_distill.sh [output_suffix]
#
# Set cuda_visible_devices and batch_size (total effective batch) in the yaml config.
# Per-GPU batch size is computed automatically as batch_size / num_gpus.
#
# <config> can be a bare name (e.g. en-zh -> configs/en-zh.yaml),
# a filename (en-zh.yaml), or an explicit path.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash run_distill.sh en-zh
#   CONFIG=en-zh CUDA_VISIBLE_DEVICES=0 bash run_distill.sh enzh_3spk_t16_10k
#   RESUME=meanflow_distill/runs/prev/checkpoints/student_step_005000.pt \
#     CONFIG=en-zh bash run_distill.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

count_gpus_from_cuda_visible_devices() {
  local devices="${CUDA_VISIBLE_DEVICES:-0}"
  devices="${devices// /}"
  if [[ -z "$devices" ]]; then
    echo 1
    return
  fi

  local count=0
  local part
  IFS=',' read -ra parts <<< "$devices"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && count=$((count + 1))
  done

  if [[ "$count" -eq 0 ]]; then
    count=1
  fi
  echo "$count"
}

find_free_master_port() {
  python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

resolve_config_file() {
  local spec="$1"
  local candidate=""

  if [[ "$spec" == /* ]]; then
    candidate="$spec"
  elif [[ "$spec" == */* ]]; then
    candidate="$spec"
  elif [[ "$spec" == *.yaml || "$spec" == *.yml ]]; then
    candidate="$SCRIPT_DIR/configs/$spec"
  else
    candidate="$SCRIPT_DIR/configs/${spec}.yaml"
  fi

  if [[ "$candidate" != /* ]]; then
    candidate="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
  fi

  echo "$candidate"
}

CONFIG_FILE=""
if [[ -n "${CONFIG:-}" ]]; then
  CONFIG_FILE="$(resolve_config_file "$CONFIG")"
fi

OUTPUT_SUFFIX_ARG=""

if [[ $# -gt 0 ]]; then
  if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$(resolve_config_file "$1")"
    shift
  fi
  if [[ $# -gt 0 ]]; then
    OUTPUT_SUFFIX_ARG="$1"
    shift
  fi
fi

if [[ -z "$CONFIG_FILE" ]]; then
  echo "ERROR: Config is required. Usage: bash run_distill.sh <config> [output_suffix]" >&2
  echo "       or: CONFIG=<config> bash run_distill.sh [output_suffix]" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

eval "$(python "$SCRIPT_DIR/load_distill_config.py" --config "$CONFIG_FILE" --repo-root "$REPO_ROOT")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

NPROC="$(count_gpus_from_cuda_visible_devices)"
TOTAL_BATCH_SIZE="$BATCH_SIZE"
PER_GPU_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NPROC))
if [[ $((PER_GPU_BATCH_SIZE * NPROC)) -ne "$TOTAL_BATCH_SIZE" ]]; then
  echo "ERROR: batch_size ($TOTAL_BATCH_SIZE) must be divisible by number of GPUs ($NPROC)" >&2
  exit 1
fi
BATCH_SIZE="$PER_GPU_BATCH_SIZE"

SUFFIX="${OUTPUT_SUFFIX_ARG:-${OUTPUT_SUFFIX:-intmeanflow${STUDENT_STEPS}_t${TEACHER_STEPS}_$(date +%Y%m%d_%H%M%S)}}"
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/runs/$SUFFIX"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
# Resume always goes to a new directory to avoid conflicts
if [[ -n "${RESUME:-}" && "${OUTPUT_DIR}" == "$DEFAULT_OUTPUT_DIR" ]]; then
  RESUME_BASE="$(basename "$(dirname "$(dirname "$RESUME")")")"
  OUTPUT_DIR="$SCRIPT_DIR/runs/${RESUME_BASE}_resumed_$(date +%Y%m%d_%H%M%S)"
fi

[[ -f "$TEACHER_CKPT" ]] || { echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT" >&2; exit 1; }
[[ -f "$TRAIN_MANIFEST" ]] || { echo "ERROR: Train manifest not found: $TRAIN_MANIFEST" >&2; exit 1; }

VAL_ARGS=()
if [[ -n "$VAL_MANIFEST" && -f "$VAL_MANIFEST" ]]; then
  VAL_ARGS=(--val-manifest "$VAL_MANIFEST" --val-every "$VAL_EVERY" --val-batches "$VAL_BATCHES")
fi

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  [[ -f "$RESUME" ]] || { echo "ERROR: Resume checkpoint not found: $RESUME" >&2; exit 1; }
  RESUME_ARGS=(--resume "$RESUME")
  echo "  Resume from:  $RESUME"
fi

echo "=== IntMeanFlow Distillation ==="
echo "  Config:       $CONFIG_FILE"
echo "  Teacher:      $TEACHER_CKPT"
echo "  Train manifest: $TRAIN_MANIFEST ($(wc -l < "$TRAIN_MANIFEST") lines)"
[[ -n "$VAL_MANIFEST" && -f "$VAL_MANIFEST" ]] && echo "  Val manifest: $VAL_MANIFEST ($(wc -l < "$VAL_MANIFEST") lines)"
echo "  Student steps: $STUDENT_STEPS"
echo "  Teacher steps: $TEACHER_STEPS"
[[ -n "${STUDENT_T_GRID:-}" ]] && echo "  Student t_grid: $STUDENT_T_GRID"
echo "  Streaming: mu=${MU_STREAMING:-false} teacher_dec=${TEACHER_DECODER_STREAMING:-false} student_dec=${DECODER_STREAMING:-false}"
echo "  KV-cache distill: ${KV_CACHE_DISTILL:-true} chunk=${DISTILL_CHUNK_SIZE:-100} left_frames=${DECODER_LEFT_FRAMES:-20}"
echo "  GPUs:         $NPROC (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Batch size:   $TOTAL_BATCH_SIZE total ($PER_GPU_BATCH_SIZE per GPU)"
echo "  Output:       $OUTPUT_DIR"
echo "================================="

STUDENT_T_GRID_ARGS=()
if [[ -n "${STUDENT_T_GRID:-}" ]]; then
  STUDENT_T_GRID_ARGS=(--student-t-grid "$STUDENT_T_GRID")
fi

STREAMING_ARGS=()
case "${STREAMING:-}" in
  1|true|True|yes|YES) STREAMING_ARGS=(--streaming) ;;
esac

MU_STREAMING_ARGS=()
case "${MU_STREAMING:-}" in
  1|true|True|yes|YES) MU_STREAMING_ARGS=(--mu-streaming) ;;
  0|false|False|no|NO) MU_STREAMING_ARGS=(--no-mu-streaming) ;;
esac

TEACHER_DECODER_STREAMING_ARGS=()
case "${TEACHER_DECODER_STREAMING:-}" in
  1|true|True|yes|YES) TEACHER_DECODER_STREAMING_ARGS=(--teacher-decoder-streaming) ;;
  0|false|False|no|NO) TEACHER_DECODER_STREAMING_ARGS=(--no-teacher-decoder-streaming) ;;
esac

DECODER_STREAMING_ARGS=()
case "${DECODER_STREAMING:-}" in
  1|true|True|yes|YES) DECODER_STREAMING_ARGS=(--decoder-streaming) ;;
  0|false|False|no|NO) DECODER_STREAMING_ARGS=(--no-decoder-streaming) ;;
esac

KV_CACHE_DISTILL_ARGS=()
case "${KV_CACHE_DISTILL:-true}" in
  1|true|True|yes|YES) KV_CACHE_DISTILL_ARGS=(--kv-cache-distill) ;;
  0|false|False|no|NO) KV_CACHE_DISTILL_ARGS=(--no-kv-cache-distill) ;;
esac

DISTILL_CHUNK_ARGS=()
if [[ -n "${DISTILL_CHUNK_SIZE:-}" ]]; then
  DISTILL_CHUNK_ARGS=(--distill-chunk-size "$DISTILL_CHUNK_SIZE")
fi

DECODER_LEFT_FRAME_ARGS=()
if [[ -n "${DECODER_LEFT_FRAMES:-}" ]]; then
  DECODER_LEFT_FRAME_ARGS=(--decoder-left-frames "$DECODER_LEFT_FRAMES")
fi

PRE_LOOKAHEAD_ARGS=()
if [[ -n "${PRE_LOOKAHEAD_LEN:-}" ]]; then
  PRE_LOOKAHEAD_ARGS=(--pre-lookahead-len "$PRE_LOOKAHEAD_LEN")
fi

TRAIN_ARGS=(
  --teacher-ckpt "$TEACHER_CKPT"
  --manifest "$TRAIN_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --cleaner "$CLEANER"
  --default-spk "$DEFAULT_SPK"
  --student-steps "$STUDENT_STEPS"
  --teacher-steps "$TEACHER_STEPS"
  --temperature "$TEMPERATURE"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --precision "$PRECISION"
  "${STUDENT_T_GRID_ARGS[@]}"
  "${STREAMING_ARGS[@]}"
  "${MU_STREAMING_ARGS[@]}"
  "${TEACHER_DECODER_STREAMING_ARGS[@]}"
  "${DECODER_STREAMING_ARGS[@]}"
  "${KV_CACHE_DISTILL_ARGS[@]}"
  "${DISTILL_CHUNK_ARGS[@]}"
  "${DECODER_LEFT_FRAME_ARGS[@]}"
  "${PRE_LOOKAHEAD_ARGS[@]}"
  "${VAL_ARGS[@]}"
  "${RESUME_ARGS[@]}"
)

if [[ "$NPROC" -gt 1 ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$(find_free_master_port)"
  fi
  export MASTER_PORT
  echo "  DDP:          $NPROC processes on $MASTER_ADDR:$MASTER_PORT"
  # If multiple DDP jobs run on one host and NCCL conflicts, try:
  #   MASTER_PORT=29501 NCCL_SHM_DISABLE=1 bash run_distill.sh ...
  exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train_intmeanflow_distill.py" \
    "${TRAIN_ARGS[@]}" \
    --dist-backend nccl
else
  exec python "$SCRIPT_DIR/train_intmeanflow_distill.py" \
    "${TRAIN_ARGS[@]}"
fi
