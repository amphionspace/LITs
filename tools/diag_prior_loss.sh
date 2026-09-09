#!/usr/bin/env bash
# Run prior-loss-by-token diagnosis on a validation filelist.
#
# Modes:
#   single   — one checkpoint (absolute per-token prior stats)
#   compare  — two checkpoints on the same filelist (Δ mean / Δ total)
#
# Examples:
#   bash tools/diag_prior_loss.sh single
#   bash tools/diag_prior_loss.sh compare
#
# Override paths via env (optional):
#   CHECKPOINT=/path/to/new.ckpt \
#   BASELINE_CHECKPOINT=/path/to/old.ckpt \
#   FILELIST=/path/to/val.txt \
#   OUTPUT_DIR=./prior_diag/my_run \
#   bash tools/diag_prior_loss.sh compare
#
# Python env (optional):
#   PYTHON=/path/to/lits_5090/bin/python bash tools/diag_prior_loss.sh compare

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-compare}"

# --- defaults (edit for your machine) ---
CHECKPOINT="${CHECKPOINT:-/chenmingjie/xingwen/multiling_up-to-date/logs/ckpt_trained_on_transsion/0727_en-zh_1199_inference.ckpt}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/chenmingjie/xingwen/multiling_up-to-date/logs/ckpt_trained_on_transsion/0727_en-zh_1199_inference.ckpt}"
FILELIST="${FILELIST:-${REPO_ROOT}/tools/en-zh_validation_for_prior_diagnosis.txt}"
MODEL_LANG="${MODEL_LANG:-en-zh-dict}"

CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-new}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline}"

OUTPUT_DIR_SINGLE="${OUTPUT_DIR:-./prior_diag/single}"
OUTPUT_DIR_COMPARE="${OUTPUT_DIR:-./prior_diag/compare}"

TOP_K="${TOP_K:-40}"
TOP_K_RHYME_TONE="${TOP_K_RHYME_TONE:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MIN_RHYME_TONE_FRAMES="${MIN_RHYME_TONE_FRAMES:-30}"
MIN_COMPARE_FRAMES="${MIN_COMPARE_FRAMES:-30}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

PYTHON="${PYTHON:-python}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash tools/diag_prior_loss.sh single
  bash tools/diag_prior_loss.sh compare

Environment overrides:
  CHECKPOINT, BASELINE_CHECKPOINT (compare mode), FILELIST, MODEL_LANG,
  OUTPUT_DIR, CHECKPOINT_LABEL, BASELINE_LABEL,
  TOP_K, TOP_K_RHYME_TONE, BATCH_SIZE, NUM_WORKERS,
  MIN_RHYME_TONE_FRAMES, MIN_COMPARE_FRAMES, MAX_SAMPLES, PYTHON

For compare mode, set CHECKPOINT to the newer/higher-prior model and
BASELINE_CHECKPOINT to the reference model (defaults are identical — edit before running).
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${FILELIST}" ]]; then
  die "filelist not found: ${FILELIST}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  die "checkpoint not found: ${CHECKPOINT}"
fi

COMMON_ARGS=(
  --filelist "${FILELIST}"
  --model_lang "${MODEL_LANG}"
  --top_k "${TOP_K}"
  --top_k_rhyme_tone "${TOP_K_RHYME_TONE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --min_rhyme_tone_frames "${MIN_RHYME_TONE_FRAMES}"
  --no-add-blank
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

case "${MODE}" in
  single)
    echo "[INFO] Mode=single  checkpoint=${CHECKPOINT}"
    echo "[INFO] output_dir=${OUTPUT_DIR_SINGLE}"
    "${PYTHON}" tools/diagnose_prior_loss_by_token.py \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${OUTPUT_DIR_SINGLE}" \
      "${COMMON_ARGS[@]}"
    ;;
  compare)
    if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
      die "baseline checkpoint not found: ${BASELINE_CHECKPOINT}"
    fi
    if [[ "${CHECKPOINT}" == "${BASELINE_CHECKPOINT}" ]]; then
      die "CHECKPOINT and BASELINE_CHECKPOINT are the same path — set both env vars before compare"
    fi
    echo "[INFO] Mode=compare  baseline=${BASELINE_CHECKPOINT} (${BASELINE_LABEL})"
    echo "[INFO]            new=${CHECKPOINT} (${CHECKPOINT_LABEL})"
    echo "[INFO] output_dir=${OUTPUT_DIR_COMPARE}"
    "${PYTHON}" tools/diagnose_prior_loss_by_token.py \
      --checkpoint "${CHECKPOINT}" \
      --baseline_checkpoint "${BASELINE_CHECKPOINT}" \
      --checkpoint_label "${CHECKPOINT_LABEL}" \
      --baseline_label "${BASELINE_LABEL}" \
      --output_dir "${OUTPUT_DIR_COMPARE}" \
      --min_compare_frames "${MIN_COMPARE_FRAMES}" \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    usage
    die "unknown mode: ${MODE} (use single or compare)"
    ;;
esac
