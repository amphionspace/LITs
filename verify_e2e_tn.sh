#!/bin/bash
# Preflight: unified TextNormalizer + resources + one-line smoke tests.
#
# Usage (standalone):
#   bash verify_e2e_tn.sh <model_lang>
#
# Or source from infer_e2e.sh (so LD_LIBRARY_PATH / ICU_ROOT persist):
#   source verify_e2e_tn.sh && verify_tn_preflight en-zh-dict
#
# Env: TN_BIN_DIR, TN_DATA_ROOT, ICU_ROOT, SKIP_TN_PREFLIGHT=1

verify_e2e_tn_die() {
  echo "[verify_e2e_tn] ERROR: $*" >&2
  exit 1
}

verify_e2e_tn_repo_root() {
  if [[ -n "${REPO_ROOT:-}" ]]; then
    printf '%s\n' "$REPO_ROOT"
    return 0
  fi
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

# After TN preflight, clear LD_LIBRARY_PATH before PyTorch/CUDA inference.
# PyTorch ships its own CUDA/cuBLAS and resolves them via RPATH when
# LD_LIBRARY_PATH is unset.  Forcing CONDA_PREFIX/lib or tools/lib into
# LD_LIBRARY_PATH (e.g. via icu_env.sh) shadows libcublas and triggers
# CUBLAS_STATUS_INVALID_VALUE under FP16 autocast.
verify_e2e_tn_prepare_inference_ld_path() {
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    echo "[verify_e2e_tn] clearing LD_LIBRARY_PATH for GPU inference (was: ${LD_LIBRARY_PATH})" >&2
    unset LD_LIBRARY_PATH
  fi
}

verify_e2e_tn_icu_lib_candidates() {
  local cand dir
  for cand in \
    "${ICU_ROOT:-}" \
    "${E2E_ICU_ROOT:-}" \
    "${HOME}/.local/icu" \
    "${CONDA_PREFIX:-}" \
    /opt/homebrew/opt/icu4c \
    /usr/local/opt/icu4c; do
    [[ -n "$cand" && -d "${cand}/lib" ]] || continue
    dir="$(cd "${cand}/lib" && pwd)"
    printf '%s\n' "$dir"
  done
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    IFS=':' read -r -a _ld_dirs <<< "$LD_LIBRARY_PATH"
    for dir in "${_ld_dirs[@]}"; do
      [[ -n "$dir" && -d "$dir" ]] || continue
      dir="$(cd "$dir" && pwd)"
      printf '%s\n' "$dir"
    done
  fi
}

verify_e2e_tn_configure_icu_runtime() {
  local bin="$1"
  local soname seen="" dir

  [[ -f "$bin" ]] || verify_e2e_tn_die "internal error: missing binary $bin"

  # macOS/Homebrew binaries normally carry an absolute ICU install name.  The
  # Linux build may need LD_LIBRARY_PATH when ICU lives outside the system path.
  if ! command -v ldd >/dev/null 2>&1; then
    return 0
  fi

  soname="$(ldd "$bin" 2>/dev/null | awk '/libicui18n\.so/{print $1; exit}')"
  if [[ -z "$soname" ]]; then
    return 0
  fi

  if ! ldd "$bin" 2>/dev/null | grep -q 'libicui18n\.so.*not found'; then
    return 0
  fi

  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    [[ ":$seen:" == *":$dir:"* ]] && continue
    seen="${seen:+$seen:}$dir"
    if [[ -f "$dir/$soname" ]]; then
      if [[ -f "$dir/libcublas.so.12" || -f "$dir/libcublasLt.so.12" ]]; then
        echo "[verify_e2e_tn] skip ICU dir with CUDA libs: $dir" >&2
        continue
      fi
      # Record ICU runtime for TN smoke tests; do not mutate the parent shell
      # LD_LIBRARY_PATH (verify_e2e_tn_prepare_inference_ld_path runs after preflight).
      export E2E_ICU_LIB_DIR="$dir"
      if [[ -z "${ICU_ROOT:-}" || ! -f "${ICU_ROOT}/lib/$soname" ]]; then
        export ICU_ROOT="$(cd "$dir/.." && pwd)"
      fi
      echo "[verify_e2e_tn] ICU: $soname <= $dir (ICU_ROOT=${ICU_ROOT})"
      return 0
    fi
  done < <(verify_e2e_tn_icu_lib_candidates)

  echo "[verify_e2e_tn] ERROR: $bin requires $soname but it was not found." >&2
  echo "Checked lib directories:" >&2
  local any_dir=0
  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    any_dir=1
    if [[ -f "$dir/$soname" ]]; then
      echo "  - $dir (has $soname)" >&2
    else
      echo "  - $dir (no $soname)" >&2
    fi
  done < <(verify_e2e_tn_icu_lib_candidates)
  if [[ "$any_dir" -eq 0 ]]; then
    echo "  (none — set ICU_ROOT or install ICU under \$HOME/.local/icu)" >&2
  fi
  echo "ldd $bin:" >&2
  ldd "$bin" >&2 || true
  echo "Fix: see latest_inference_guide.md — install ICU, set ICU_ROOT, rerun bash install_e2e_tn.sh" >&2
  exit 1
}

verify_e2e_tn_required_langs() {
  case "$1" in
    en-zh|en-zh-dict|en-zh-rhyme-body-tone|en-zh-dict-rhyme-body-tone) echo "zh en" ;;
    *) verify_e2e_tn_die "Unsupported model_lang: $1" ;;
  esac
}

verify_e2e_tn_probe_text() {
  case "$1" in
    zh) echo '给我$10' ;;
    en) echo 'it cost me $10.' ;;
    *) verify_e2e_tn_die "No TN probe for lang=$1" ;;
  esac
}

verify_e2e_tn_icu_binary() {
  local bin_dir="$1"
  if [[ -f "$bin_dir/tts_cli" ]]; then
    printf '%s\n' "$bin_dir/tts_cli"
    return 0
  fi
  # Legacy installs: per-locale ELF binaries without unified tts_cli.
  local lang
  for lang in zh en; do
    if [[ -f "$bin_dir/${lang}_tts" ]]; then
      printf '%s\n' "$bin_dir/${lang}_tts"
      return 0
    fi
  done
  verify_e2e_tn_die "No TN binary found in $bin_dir (expected tts_cli or zh_tts; run: bash install_e2e_tn.sh)"
}

# Run a TN CLI command with ICU-only LD_LIBRARY_PATH in a subshell so the parent
# shell stays clean for PyTorch/CUDA inference.
verify_e2e_tn_run_with_icu_runtime() {
  local bin_dir="$1"
  shift
  (
    if [[ -f "$bin_dir/icu_env.sh" ]]; then
      # shellcheck source=/dev/null
      source "$bin_dir/icu_env.sh"
    elif [[ -n "${E2E_ICU_LIB_DIR:-}" ]]; then
      export LD_LIBRARY_PATH="${E2E_ICU_LIB_DIR}"
    elif [[ -n "${ICU_ROOT:-}" && -d "${ICU_ROOT}/lib" ]]; then
      export LD_LIBRARY_PATH="${ICU_ROOT}/lib"
    fi
    "$@"
  )
}

verify_e2e_tn_smoke_test() {
  local lang="$1"
  local bin="$2"
  local data_root="$3"
  local probe out rc
  local bin_dir

  [[ -x "$bin" ]] || verify_e2e_tn_die "Missing executable: $bin"
  bin_dir="$(dirname "$bin")"

  probe="$(verify_e2e_tn_probe_text "$lang")"
  echo "[verify_e2e_tn] smoke data/${lang} probe=$(printf '%q' "$probe")" >&2
  if command -v timeout >/dev/null 2>&1; then
    out="$(printf '%s\n' "$probe" | verify_e2e_tn_run_with_icu_runtime "$bin_dir" timeout 30s "$bin" --data "$data_root/$lang" 2>&1)" || rc=$?
  else
    out="$(printf '%s\n' "$probe" | verify_e2e_tn_run_with_icu_runtime "$bin_dir" "$bin" --data "$data_root/$lang" 2>&1)" || rc=$?
  fi
  rc="${rc:-0}"

  if [[ "$rc" -eq 124 ]]; then
    verify_e2e_tn_die "${lang}_tts smoke test timed out after 30s (ICU/runtime or broken wrapper? try: bash test_tn.sh $lang)"
  fi

  if [[ "$rc" -ne 0 ]]; then
    echo "[verify_e2e_tn] ERROR: data/${lang} smoke test failed (exit $rc)." >&2
    echo "  probe: $probe" >&2
    echo "  output: $out" >&2
    exit 1
  fi
  if [[ -z "${out//$'\n'/}" ]]; then
    verify_e2e_tn_die "data/${lang} returned empty output for probe: $probe"
  fi
  if [[ "$out" == *'$'* ]]; then
    verify_e2e_tn_die "data/${lang} did not normalize currency probe (output still contains '\$'): $out"
  fi
  echo "[verify_e2e_tn] OK: data/${lang} probe -> ${out//$'\n'/ }"
}

verify_e2e_g2p_smoke_test() {
  local bin="$1"
  local data_root="$2"
  local input='我有一个AI助手。'
  local out bin_dir

  bin_dir="$(dirname "$bin")"
  out="$(printf '%s\n' "$input" | verify_e2e_tn_run_with_icu_runtime "$bin_dir" "$bin" --data "$data_root/en-zh-g2p")" \
    || verify_e2e_tn_die "data/en-zh-g2p smoke test failed"
  [[ -n "$out" && "$out" == *"EY1 AY1"* ]] \
    || verify_e2e_tn_die "data/en-zh-g2p returned unexpected output: '$out'"
  echo "[verify_e2e_tn] OK: data/en-zh-g2p probe -> $out"
}

verify_tn_preflight() {
  local model_lang="$1"
  local repo_root langs lang bin
  local bin_dir data_root

  echo "[verify_e2e_tn] start model_lang=$model_lang" >&2

  if [[ "${SKIP_TN_PREFLIGHT:-}" == 1 ]]; then
    echo "[verify_e2e_tn] SKIP_TN_PREFLIGHT=1: skipping"
    return 0
  fi

  repo_root="$(verify_e2e_tn_repo_root)"
  bin_dir="${TN_BIN_DIR:-$repo_root/e2e_infer/bin}"
  data_root="${TN_DATA_ROOT:-$repo_root/frontend/data}"
  bin="$bin_dir/tts_cli"
  [[ -x "$bin" ]] \
    || verify_e2e_tn_die "Missing TextNormalizer CLI: $bin (run: bash install_e2e_tn.sh)"

  langs="$(verify_e2e_tn_required_langs "$model_lang")"
  for lang in $langs; do
    [[ -f "$data_root/$lang/config.json" ]] \
      || verify_e2e_tn_die "Missing TN resource: $data_root/$lang/config.json"
  done
  [[ -f "$data_root/en-zh-g2p/config.json" ]] \
    || verify_e2e_tn_die "Missing G2P resource: $data_root/en-zh-g2p/config.json"
  [[ -f "$data_root/en-zh-g2p/model_tokens.json" ]] \
    || verify_e2e_tn_die "Missing Text2Id resource: $data_root/en-zh-g2p/model_tokens.json"

  verify_e2e_tn_configure_icu_runtime "$bin"
  echo "[verify_e2e_tn] preflight model_lang=$model_lang backend=cli artifact=$bin data_root=$data_root"
  for lang in $langs; do
    verify_e2e_tn_smoke_test "$lang" "$bin" "$data_root"
  done
  verify_e2e_g2p_smoke_test "$bin" "$data_root"

  verify_e2e_tn_prepare_inference_ld_path
  echo "[verify_e2e_tn] inference LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}" >&2
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  [[ "${1:-}" ]] || { echo "Usage: $0 <model_lang>" >&2; exit 1; }
  verify_tn_preflight "$1"
fi
