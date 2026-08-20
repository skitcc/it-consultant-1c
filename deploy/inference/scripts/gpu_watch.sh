#!/usr/bin/env bash
# Keep nvidia-smi ticking while another command (or Ctrl-C) runs.
# Usage: ./gpu_watch.sh                 # until Ctrl-C
#        ./gpu_watch.sh -- ./scripts/ollama_pipeline_trace.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Run gpu_watch.sh on GPU server 2." >&2
  exit 1
fi

interval="${GPU_WATCH_INTERVAL:-1}"
out="$ROOT/out/gpu_watch.log"
{
  echo "# $(date -Is) interval=${interval}s"
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv
} >"$out"

echo "Writing GPU samples to $out (every ${interval}s). Ctrl-C to stop."

if [[ "${1:-}" == "--" ]]; then
  shift
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv --loop="$interval" >>"$out" &
  watcher=$!
  trap 'kill "$watcher" 2>/dev/null || true' EXIT
  "$@"
  kill "$watcher" 2>/dev/null || true
  echo "Saved $out"
  tail -n 20 "$out"
  exit 0
fi

exec nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv --loop="$interval"
