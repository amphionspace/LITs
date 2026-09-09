#!/bin/bash
# Quick TN smoke test (verbose). Does not load LITs / vocoder.
#
# Usage:
#   bash test_tn.sh [zh|en] [probe_text]
#
# Examples:
#   bash test_tn.sh
#   bash test_tn.sh zh '给我$10'
#   bash test_tn.sh en 'it cost me $10.'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${TN_BIN_DIR:-$REPO_ROOT/e2e_infer/bin}"
LANG="${1:-zh}"
PROBE="${2:-}"

case "$LANG" in
  zh) PROBE="${PROBE:-给我\$10}" ;;
  en) PROBE="${PROBE:-it cost me \$10.}" ;;
  *) echo "Usage: $0 [zh|en] [probe_text]" >&2; exit 1 ;;
esac

BIN="$BIN_DIR/${LANG}_tts"
echo "[test_tn] bin_dir=$BIN_DIR"
echo "[test_tn] lang=$LANG probe=$PROBE"

if [[ ! -x "$BIN" ]]; then
  echo "[test_tn] ERROR: missing executable $BIN" >&2
  echo "Run: bash install_e2e_tn.sh" >&2
  exit 1
fi

if [[ -f "$BIN_DIR/icu_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$BIN_DIR/icu_env.sh"
  echo "[test_tn] sourced $BIN_DIR/icu_env.sh"
fi

if command -v file >/dev/null 2>&1; then
  echo "[test_tn] file: $(file -b "$BIN")"
fi

if [[ -f "$BIN_DIR/tts_cli" ]]; then
  echo "[test_tn] ldd tts_cli:"
  ldd "$BIN_DIR/tts_cli" 2>&1 | head -5 || true
else
  echo "[test_tn] ldd ${LANG}_tts:"
  ldd "$BIN" 2>&1 | head -5 || true
fi

echo "[test_tn] running (timeout 15s) ..."
set +e
OUT="$(printf '%s\n' "$PROBE" | timeout 15s "$BIN" 2>&1)"
RC=$?
set -e

echo "[test_tn] exit=$RC"
echo "[test_tn] output: ${OUT//$'\n'/ }"

if [[ "$RC" -eq 124 ]]; then
  echo "[test_tn] ERROR: timed out — TN binary hung (check ICU / reinstall: bash install_e2e_tn.sh)" >&2
  exit 124
fi
if [[ "$RC" -ne 0 ]]; then
  echo "[test_tn] ERROR: TN binary failed" >&2
  exit "$RC"
fi
if [[ -z "${OUT//$'\n'/}" ]]; then
  echo "[test_tn] ERROR: empty output" >&2
  exit 1
fi
if [[ "$OUT" == *'$'* ]]; then
  echo "[test_tn] WARNING: output still contains '\$' (TN may not have normalized currency)" >&2
fi

echo "[test_tn] OK"
