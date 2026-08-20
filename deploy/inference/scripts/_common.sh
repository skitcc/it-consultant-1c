#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export INFERENCE_ROOT="$ROOT"
mkdir -p "$ROOT/out"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
elif [[ -f "$ROOT/.env.example" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.example"
  set +a
fi

py() {
  PYTHONPATH="$ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
}
