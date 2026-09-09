#!/usr/bin/env bash
# Split filelist(s) by wav — all text versions of one utterance stay in the same split.
#
# Usage (merged filelist, multi-row per wav):
#   ./data/filelists/split_train_valid_by_wav.sh \
#     data/filelists/all_mixed.txt \
#     data/filelists/train_mixed.txt \
#     data/filelists/valid_mixed.txt
#
# Usage (no-diac + with-diac, split before merge):
#   PAIRED_INPUT=data/filelists/all_with_diac.txt \
#   PAIRED_TRAIN_OUT=data/filelists/train_with_diac.txt \
#   PAIRED_VALID_OUT=data/filelists/valid_with_diac.txt \
#   ./data/filelists/split_train_valid_by_wav.sh \
#     data/filelists/all_no_diac.txt \
#     data/filelists/train_no_diac.txt \
#     data/filelists/valid_no_diac.txt
#
# Environment:
#   RATIO=0.9
#   SEED=42
#   SHUFFLE=1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INVOCATION_CWD="$(pwd)"

abspath() {
  local path="$1"
  if [[ "$path" != /* ]]; then
    path="$INVOCATION_CWD/$path"
  fi
  echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
}

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-1}"
}

INPUT="${1:-}"
TRAIN_OUT="${2:-}"
VALID_OUT="${3:-}"
RATIO="${RATIO:-0.9}"
SEED="${SEED:-42}"
SHUFFLE="${SHUFFLE:-1}"

if [[ -z "$INPUT" || -z "$TRAIN_OUT" || -z "$VALID_OUT" ]]; then
  usage
fi
INPUT="$(abspath "$INPUT")"
TRAIN_OUT="$(abspath "$TRAIN_OUT")"
VALID_OUT="$(abspath "$VALID_OUT")"

if [[ ! -f "$INPUT" ]]; then
  echo "error: input filelist not found: $INPUT" >&2
  exit 1
fi

if [[ -n "${PAIRED_INPUT:-}" ]]; then
  if [[ -z "${PAIRED_TRAIN_OUT:-}" || -z "${PAIRED_VALID_OUT:-}" ]]; then
    echo "error: PAIRED_TRAIN_OUT and PAIRED_VALID_OUT required when PAIRED_INPUT is set" >&2
    exit 1
  fi
  PAIRED_INPUT="$(abspath "$PAIRED_INPUT")"
  PAIRED_TRAIN_OUT="$(abspath "$PAIRED_TRAIN_OUT")"
  PAIRED_VALID_OUT="$(abspath "$PAIRED_VALID_OUT")"
  if [[ ! -f "$PAIRED_INPUT" ]]; then
    echo "error: paired input filelist not found: $PAIRED_INPUT" >&2
    exit 1
  fi
fi

cd "$REPO_ROOT"

cmd=(python data/filelists/split_train_valid_by_wav.py
  --input "$INPUT"
  --train-output "$TRAIN_OUT"
  --valid-output "$VALID_OUT"
  --ratio "$RATIO"
  --seed "$SEED"
)

if [[ -n "${PAIRED_INPUT:-}" ]]; then
  cmd+=(--paired-input "$PAIRED_INPUT"
    --paired-train-output "$PAIRED_TRAIN_OUT"
    --paired-valid-output "$PAIRED_VALID_OUT")
fi

if [[ "$SHUFFLE" == "1" ]]; then
  cmd+=(--shuffle)
fi

echo "[split_by_wav] repo:        $REPO_ROOT"
echo "[split_by_wav] invoked from: $INVOCATION_CWD"
echo "[split_by_wav] input:       $INPUT"
echo "[split_by_wav] train:       $TRAIN_OUT"
echo "[split_by_wav] valid:       $VALID_OUT"
echo "[split_by_wav] ratio:       $RATIO"
echo "[split_by_wav] seed:        $SEED"
echo "[split_by_wav] shuffle:     $SHUFFLE"
if [[ -n "${PAIRED_INPUT:-}" ]]; then
  echo "[split_by_wav] paired:      $PAIRED_INPUT"
  echo "[split_by_wav] paired train: $PAIRED_TRAIN_OUT"
  echo "[split_by_wav] paired valid: $PAIRED_VALID_OUT"
fi
echo

"${cmd[@]}"
