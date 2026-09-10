#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$ROOT/bin/tts_cli}"

ICU_ROOT="${ICU_ROOT:-}"
if [[ -z "$ICU_ROOT" ]]; then
  for cand in /usr /opt/homebrew/opt/icu4c /usr/local/opt/icu4c \
    /opt/homebrew/opt/icu4c@* /usr/local/opt/icu4c@*; do
    if [[ -f "$cand/include/unicode/locid.h" ]]; then
      ICU_ROOT="$cand"
      break
    fi
  done
fi
[[ -n "$ICU_ROOT" ]] || {
  echo 'Set ICU_ROOT to the icu4c installation prefix.' >&2
  exit 1
}

mkdir -p "$(dirname "$OUT")"
g++ -std=c++17 -O2 \
  "$ROOT/unified/tts_cli.cpp" \
  "$ROOT/unified/text_normalizer.cpp" \
  "$ROOT/unified/frontend_ops.cpp" \
  "$ROOT/unified/frontend_rules_merge.cpp" \
  "$ROOT/unified/backend.cpp" \
  "$ROOT/unified/icu_backend.cpp" \
  "$ROOT/unified/english_backend.cpp" \
  "$ROOT/unified/chinese_backend.cpp" \
  "$ROOT/unified/chinese_g2p_ops.cpp" \
  "$ROOT/unified/en_zh_g2p_backend.cpp" \
  "$ROOT/tts_normalizer_engine.cpp" \
  -I"$ROOT" -I"$ROOT/third_party" -I"$ROOT/unified" \
  -I"$ICU_ROOT/include" \
  -L"$ICU_ROOT/lib" -licui18n -licuuc -licudata -lpthread \
  -o "$OUT"
echo "Built $OUT"
