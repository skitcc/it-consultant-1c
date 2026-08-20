#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$ROOT"
docker compose --env-file "$ROOT/.env" -f "$ROOT/compose.yml" down "$@"
