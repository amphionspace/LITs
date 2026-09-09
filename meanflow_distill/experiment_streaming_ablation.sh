#!/bin/bash
# Streaming artifact ablation: isolate vocoder-boundary vs ODE-path causes.
#
# Symptom target: tremor / shakiness / hollowness at chunk boundaries (not clicks).
# All variants use causal_kv ODE streaming unless noted.
#
# Usage:
#   STUDENT_CKPT=... SPK_ID=3 bash meanflow_distill/experiment_streaming_ablation.sh
#   STUDENT_CKPT=... SPK_ID=3 bash meanflow_distill/experiment_streaming_ablation.sh A C E16
#
# Variants:
#   A      baseline: causal_kv + chunked vocoder + waveform crossfade
#   B      non_streaming full-utterance ODE (upper-bound quality reference)
#   C      causal_kv + full mel vocoder (single Vocos pass; tests vocoder chunking)
#   D      causal_kv + chunked vocoder + no waveform crossfade
#   E16    causal_kv + mel_cache_len=16
#   E24    causal_kv + mel_cache_len=24
#
# Outputs: meanflow_distill/infer_output/ablation_<variant>/
#
# Interpretation:
#   C fixes tremor  -> vocoder chunking is the main cause
#   D worsens       -> waveform crossfade helps (amplitude smoothing)
#   E* changes      -> mel cache length sensitivity

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STUDENT_CKPT="${STUDENT_CKPT:?Set STUDENT_CKPT to a distilled checkpoint}"
SPK_ID="${SPK_ID:-1}"
INPUT_TXT="${INPUT_TXT:-$SCRIPT_DIR/infer_output/shahenda_ru_distilled/normalized.txt}"
MODEL_LANG="${MODEL_LANG:-en-zh-dict}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
PYTHON="${PYTHON:-${REPO_ROOT}/e2e_test/.venv/bin/python}"

ALL_VARIANTS=(A B C D E16 E24)
if [[ $# -gt 0 ]]; then
  VARIANTS=("$@")
else
  VARIANTS=("${ALL_VARIANTS[@]}")
fi

run_variant() {
  local id="$1"
  local out_id="ablation_${id}"
  local non_streaming=0
  local mel_cache_len="${MEL_CACHE_LEN:-8}"
  local vocoder_mode="chunked"
  local no_xfade=0

  case "$id" in
    A) ;;
    B) non_streaming=1 ;;
    C) vocoder_mode="full" ;;
    D) no_xfade=1 ;;
    E16) mel_cache_len=16 ;;
    E24) mel_cache_len=24 ;;
    *)
      echo "[ablation] ERROR: unknown variant '$id' (choose: ${ALL_VARIANTS[*]})" >&2
      exit 1
      ;;
  esac

  echo "" >&2
  echo "========== [$id] non_streaming=$non_streaming mel_cache=$mel_cache_len vocoder=$vocoder_mode no_xfade=$no_xfade ==========" >&2

  NON_STREAMING="$non_streaming" \
    MEL_CACHE_LEN="$mel_cache_len" \
    VOCODER_MODE="$vocoder_mode" \
    NO_WAVEFORM_CROSSFADE="$no_xfade" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    PYTHON="$PYTHON" \
    STUDENT_CKPT="$STUDENT_CKPT" \
    SPK_ID="$SPK_ID" \
    OUTPUT_DIR="$SCRIPT_DIR/infer_output/$out_id" \
    bash "$SCRIPT_DIR/infer_distilled.sh" "$MODEL_LANG" "$INPUT_TXT" "$out_id"
}

[[ -f "$STUDENT_CKPT" ]] || { echo "[ablation] ERROR: student ckpt not found: $STUDENT_CKPT" >&2; exit 1; }
[[ -f "$INPUT_TXT" ]] || { echo "[ablation] ERROR: input txt not found: $INPUT_TXT" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "[ablation] WARNING: python not executable at $PYTHON; using PATH python" >&2; PYTHON=""; }

echo "[ablation] student=$STUDENT_CKPT spk=$SPK_ID input=$INPUT_TXT variants=${VARIANTS[*]}" >&2

for v in "${VARIANTS[@]}"; do
  run_variant "$v"
done

echo "" >&2
echo "[ablation] Done. Listen under: $SCRIPT_DIR/infer_output/ablation_*/" >&2
echo "[ablation] Compare same utterance across folders, e.g. ablation_A/1.wav vs ablation_C/1.wav" >&2
