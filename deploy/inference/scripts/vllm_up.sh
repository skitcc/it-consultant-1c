#!/usr/bin/env bash
# Start vLLM containers. Optional args are compose service names:
#   ./vllm_up.sh
#   ./vllm_up.sh vllm-llm
# After changing .env, run this again (compose recreates changed services).
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "Created $ROOT/.env from example — edit GPU_ID / HF_HOME / *_GPU_UTIL, then rerun."
fi

"$SCRIPT_DIR/vram_budget.sh"

hf_home="${HF_HOME:-/opt/hf-cache}"
mkdir -p "$hf_home"

echo "Starting vLLM with image ${VLLM_IMAGE:-vllm/vllm-openai:latest}"
cd "$ROOT"
docker compose --env-file "$ROOT/.env" -f "$ROOT/compose.yml" up -d "$@"
echo
"$SCRIPT_DIR/vllm_status.sh" || true
echo
echo "First start of gpt-oss-120b can take several minutes (load ~61GB)."
echo "Logs: ./scripts/vllm_logs.sh llm"
