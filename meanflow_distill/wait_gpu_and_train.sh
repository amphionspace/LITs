#!/bin/bash
# Monitor GPU usage in tmux; start distillation when watched GPUs are idle.
#
# Usage (inside tmux, with your conda/env already activated):
#   bash meanflow_distill/wait_gpu_and_train.sh
#
# Examples:
#   # Wait for GPUs 0-3 to clear, then train en-zh
#   WAIT_GPUS=0,1,2,3 bash meanflow_distill/wait_gpu_and_train.sh
#
#   # Wait for specific PIDs from a running job (auto-snapshotted at start)
#   SNAPSHOT_PIDS=1 WAIT_GPUS=0,1,2,3 bash meanflow_distill/wait_gpu_and_train.sh
#
#   # Custom config / poll interval
#   CONFIG=en-zh.yaml POLL_INTERVAL=1800 bash meanflow_distill/wait_gpu_and_train.sh
#
# Environment variables:
#   CONFIG            Distill config (default: en-zh.yaml)
#   WAIT_GPUS         Comma-separated GPU indices to watch (default: 0,1,2,3)
#   POLL_INTERVAL     Seconds between checks (default: 1800 = 30 min)
#   IDLE_CONFIRM      Consecutive idle polls required before launch (default: 2)
#   SNAPSHOT_PIDS     If 1, also wait for compute PIDs present on WAIT_GPUS at start (default: 1)
#   MIN_FREE_MEM_MIB  Require at least this much free memory per watched GPU (default: 0 = disabled)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="${CONFIG:-en-zh.yaml}"
WAIT_GPUS="${WAIT_GPUS:-0,1,2,3}"
POLL_INTERVAL="${POLL_INTERVAL:-1800}"
IDLE_CONFIRM="${IDLE_CONFIRM:-2}"
SNAPSHOT_PIDS="${SNAPSHOT_PIDS:-1}"
MIN_FREE_MEM_MIB="${MIN_FREE_MEM_MIB:-0}"

IFS=',' read -ra GPU_LIST <<< "${WAIT_GPUS// /}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

print_header() {
  echo
  echo "========== $(timestamp) =========="
  echo "Watching GPUs: ${WAIT_GPUS}"
  if [[ ${#SNAPSHOT_PID_LIST[@]} -gt 0 ]]; then
    echo "Snapshot PIDs: ${SNAPSHOT_PID_LIST[*]}"
  fi
  echo "Config:        $CONFIG"
  echo "Poll interval: ${POLL_INTERVAL}s | idle confirm: ${IDLE_CONFIRM}x"
  echo "----------------------------------"
  nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader,nounits | awk -F', ' -v gpus="$WAIT_GPUS" '
      BEGIN {
        split(gpus, watch, ",");
        for (i in watch) wanted[watch[i]] = 1;
      }
      {
        idx = $1;
        if (idx in wanted) {
          printf "GPU %s | %s | %sC | util %s%%/%s%% | mem %s/%s MiB\n",
            idx, $2, $3, $4, $5, $6, $7;
        }
      }'
  echo "----------------------------------"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
}

snapshot_compute_pids() {
  local -a pids=()
  local gpu pid

  for gpu in "${GPU_LIST[@]}"; do
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      pids+=("$pid")
    done < <(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true)
  done

  local -A seen=()
  local -a unique=()
  for pid in "${pids[@]}"; do
    [[ -n "${seen[$pid]:-}" ]] && continue
    seen["$pid"]=1
    unique+=("$pid")
  done

  SNAPSHOT_PID_LIST=("${unique[@]}")
}

gpu_has_compute_process() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "$pids" ]]; then
    echo 1
  else
    echo 0
  fi
}

gpu_free_mem_mib() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

any_snapshot_pid_alive() {
  local pid
  for pid in "${SNAPSHOT_PID_LIST[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

gpus_are_idle() {
  local gpu
  for gpu in "${GPU_LIST[@]}"; do
    if [[ "$(gpu_has_compute_process "$gpu")" == "1" ]]; then
      return 1
    fi
    if [[ "$MIN_FREE_MEM_MIB" -gt 0 ]]; then
      local free_mem
      free_mem="$(gpu_free_mem_mib "$gpu")"
      if [[ -z "$free_mem" || "$free_mem" -lt "$MIN_FREE_MEM_MIB" ]]; then
        return 1
      fi
    fi
  done

  if [[ "$SNAPSHOT_PIDS" == "1" && ${#SNAPSHOT_PID_LIST[@]} -gt 0 ]]; then
    any_snapshot_pid_alive && return 1
  fi

  return 0
}

SNAPSHOT_PID_LIST=()
if [[ "$SNAPSHOT_PIDS" == "1" ]]; then
  snapshot_compute_pids
fi

echo "[wait_gpu_and_train] Repo: $REPO_ROOT"
echo "[wait_gpu_and_train] Will launch: bash $SCRIPT_DIR/run_distill.sh $CONFIG"
echo "[wait_gpu_and_train] Press Ctrl+C to cancel."

idle_streak=0
while true; do
  print_header

  if gpus_are_idle; then
    idle_streak=$((idle_streak + 1))
    echo "Status: idle (${idle_streak}/${IDLE_CONFIRM})"
    if [[ "$idle_streak" -ge "$IDLE_CONFIRM" ]]; then
      echo
      echo "========== $(timestamp) GPUs idle — starting training =========="
      cd "$REPO_ROOT"
      exec bash "$SCRIPT_DIR/run_distill.sh" "$CONFIG"
    fi
  else
    idle_streak=0
    echo "Status: busy — waiting ${POLL_INTERVAL}s ..."
  fi

  sleep "$POLL_INTERVAL"
done
