#!/bin/bash
# Print per-token duration (mel frames) for one utterance.
#
# Usage:
#   ./text_duration_debug.sh "嗯?你说还行啊"
#   ./text_duration_debug.sh --lang en-zh-dict "嗯?你说还行啊"
#   ./text_duration_debug.sh --lang en-zh-dict "嗯？"
#   ./text_duration_debug.sh --no-patches "嗯?"
#   ./text_duration_debug.sh --compact "你好"
#   CHECKPOINT=/path/to/model.ckpt SPK_ID=0 ./text_duration_debug.sh "你好"
#
# All flags are handled by: python -m lits.utils.text_duration_debug --help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  PYTHON_BIN=python3
fi

DEFAULT_CKPT="$REPO_ROOT/logs/ckpt_trained_on_transsion/0706_en-zh_1999_inference.ckpt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
SPK_ID="${SPK_ID:-0}"

exec "$PYTHON_BIN" -m lits.utils.text_duration_debug \
  --checkpoint "$CHECKPOINT" \
  --spk-id "$SPK_ID" \
  "$@"
