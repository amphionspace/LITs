#!/bin/bash
# Build the vendored Chinese-English TextNormalizer and install it for inference.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN_ROOT="$REPO_ROOT/frontend"
INSTALL_DIR="$REPO_ROOT/e2e_infer/bin"

[[ -x "$TN_ROOT/build.sh" ]] || {
  echo "Missing vendored frontend build script: $TN_ROOT/build.sh" >&2
  exit 1
}

if [[ -z "${ICU_ROOT:-}" ]]; then
  for cand in "${HOME}/.local/icu" "${CONDA_PREFIX:-}" /usr \
    /opt/homebrew/opt/icu4c /usr/local/opt/icu4c \
    /opt/homebrew/opt/icu4c@* /usr/local/opt/icu4c@*; do
    if [[ -n "$cand" && -f "$cand/include/unicode/locid.h" ]]; then
      ICU_ROOT="$cand"
      break
    fi
  done
fi
[[ -n "${ICU_ROOT:-}" ]] || {
  echo 'Set ICU_ROOT to the icu4c installation prefix.' >&2
  exit 1
}
export ICU_ROOT

for resource in \
  config.json model_tokens.json \
  resources/chinese_lexicon.txt \
  resources/cmudict-en-zh-merged.txt \
  resources/pinyin_2_bpmf.txt \
  resources/user_dict.txt; do
  [[ -f "$TN_ROOT/data/en-zh-g2p/$resource" ]] || {
    echo "Missing vendored en-zh-g2p resource: $resource" >&2
    exit 1
  }
done

mkdir -p "$INSTALL_DIR"
bash "$TN_ROOT/build.sh" "$INSTALL_DIR/tts_cli"
chmod +x "$INSTALL_DIR/tts_cli"
if [[ -d "$ICU_ROOT/lib" ]]; then
  printf 'export LD_LIBRARY_PATH="%s"\n' "$ICU_ROOT/lib" > "$INSTALL_DIR/icu_env.sh"
fi
echo "Installed $INSTALL_DIR/tts_cli"
echo 'Profiles: en zh en-zh-g2p'
