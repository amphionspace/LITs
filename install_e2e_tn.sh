#!/bin/bash
# Build the unified Transsion TextNormalizer and install it for infer_e2e.py.
#
# e2e_infer/ is NOT in git (see .gitignore). This script creates it on first run.
# One ``tts_cli`` serves the Chinese and English profiles used here;
# ``--data data/<locale>`` selects the backend at process initialization.
#
# Usage (from repo root):
#   git submodule update --init Transsion_Multilingual_Text_Normalization_for_TTS
#   export ICU_ROOT="$HOME/.local/icu"   # see latest_inference_guide.md
#   bash install_e2e_tn.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN_ROOT="$REPO_ROOT/Transsion_Multilingual_Text_Normalization_for_TTS"
BUILD_OUT="$TN_ROOT/test/bin"
INSTALL_DIR="$REPO_ROOT/e2e_infer/bin"

if [[ ! -d "$TN_ROOT" ]]; then
  echo "Missing TN submodule: $TN_ROOT" >&2
  echo "Run: git submodule update --init Transsion_Multilingual_Text_Normalization_for_TTS" >&2
  exit 1
fi

if [[ ! -f "$TN_ROOT/test/scripts/build_unified.sh" ]]; then
  echo "Missing TN build script: $TN_ROOT/test/scripts/build_unified.sh" >&2
  echo "Your TN submodule may be too old or incomplete. Try:" >&2
  echo "  git submodule update --init --recursive Transsion_Multilingual_Text_Normalization_for_TTS" >&2
  exit 1
fi

if [[ -z "${ICU_ROOT:-}" ]]; then
  for cand in \
    "${HOME}/.local/icu" \
    "${CONDA_PREFIX:-}" \
    /opt/homebrew/opt/icu4c \
    /usr/local/opt/icu4c \
    /opt/homebrew/opt/icu4c@* \
    /usr/local/opt/icu4c@* \
    /usr; do
    if [[ -n "$cand" && -f "${cand}/include/unicode/locid.h" ]]; then
      ICU_ROOT="$cand"
      break
    fi
  done
fi
if [[ -z "${ICU_ROOT:-}" ]]; then
  echo "Set ICU_ROOT to your icu4c prefix (e.g. export ICU_ROOT=\"\$HOME/.local/icu\")." >&2
  echo "See latest_inference_guide.md — section on uv + building ICU from source." >&2
  exit 1
fi
export ICU_ROOT

echo "[install_e2e_tn] Using ICU_ROOT=$ICU_ROOT"
for resource in \
  config.json \
  model_tokens.json \
  resources/chinese_lexicon.txt \
  resources/cmudict-en-zh-merged.txt \
  resources/pinyin_2_bpmf.txt \
  resources/user_dict.txt; do
  if [[ ! -f "$TN_ROOT/data/en-zh-g2p/$resource" ]]; then
    echo "[install_e2e_tn] ERROR: missing self-contained en-zh-g2p resource: $resource" >&2
    exit 1
  fi
done
echo "[install_e2e_tn] Building unified TextNormalizer CLI ..."
bash "$TN_ROOT/test/scripts/build_unified.sh"

if [[ ! -f "$BUILD_OUT/tts_cli" ]]; then
  echo "[install_e2e_tn] ERROR: build succeeded but $BUILD_OUT/tts_cli is missing" >&2
  exit 1
fi

echo "[install_e2e_tn] Installing to $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
cp -f "$BUILD_OUT/tts_cli" "$INSTALL_DIR/tts_cli"
chmod +x "$INSTALL_DIR/tts_cli"

if [[ -n "${ICU_ROOT:-}" && -d "${ICU_ROOT}/lib" ]]; then
  printf 'export LD_LIBRARY_PATH="%s"\n' "${ICU_ROOT}/lib" > "$INSTALL_DIR/icu_env.sh"
elif [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  printf 'export LD_LIBRARY_PATH="%s"\n' "$LD_LIBRARY_PATH" > "$INSTALL_DIR/icu_env.sh"
fi

echo "[install_e2e_tn] Done. Installed:"
ls -1 "$INSTALL_DIR/tts_cli"
echo "[install_e2e_tn] Profiles used by this repository: en zh en-zh-g2p"
