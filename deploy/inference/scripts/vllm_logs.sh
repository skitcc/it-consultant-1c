#!/usr/bin/env bash
# Tail logs. Names: llm | rerank | vlm | embed | all
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$ROOT"
target="${1:-all}"
shift || true
case "$target" in
  all) services=() ;;
  llm|rerank|vlm|embed) services=("vllm-$target") ;;
  vllm-*) services=("$target") ;;
  *)
    echo "Usage: $0 [llm|rerank|vlm|embed|all]" >&2
    exit 2
    ;;
esac
docker compose --env-file "$ROOT/.env" -f "$ROOT/compose.yml" logs -f --tail=100 "${services[@]}" "$@"
