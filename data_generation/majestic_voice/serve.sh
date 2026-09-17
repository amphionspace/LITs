#!/usr/bin/env bash
set -euo pipefail
TASK_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ASSETS_ROOT="${MAJESTIC_VOICE_ASSETS_ROOT:-/119010446/tts-assets}"
TASK_DATA_ROOT="${MAJESTIC_VOICE_DATA_ROOT:-/ai_sds_wuzz/DATA_TTS/MajesticVoice}"
TASK_GPU="${1:?Usage: serve.sh GPU_ID PORT}"
TASK_PORT="${2:?Usage: serve.sh GPU_ID PORT}"
export CUDA_VISIBLE_DEVICES="$TASK_GPU"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR="$TASK_DATA_ROOT/runtime_cache/triton_gpu$TASK_GPU"
export TORCHINDUCTOR_CACHE_DIR="$TASK_DATA_ROOT/runtime_cache/inductor_gpu$TASK_GPU"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exec "$TASK_ASSETS_ROOT/.venv-vllm-voxcpm2/bin/vllm-omni" serve "$TASK_ASSETS_ROOT/VoxCPM2" \
  --omni --host 127.0.0.1 --port "$TASK_PORT" \
  --deploy-config "$TASK_CODE_DIR/deploy.yaml"
