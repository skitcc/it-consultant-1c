#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$ROOT"

echo "== compose =="
docker compose --env-file "$ROOT/.env" -f "$ROOT/compose.yml" ps || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "== nvidia-smi =="
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi

echo
echo "== /v1/models =="
py - <<'PY'
from catalog import load_dotenv, load_models
from _http import request_json

load_dotenv()
ok = True
for item in load_models():
    url = f"http://127.0.0.1:{item['port']}/v1/models"
    try:
        status, data = request_json(url, timeout=5)
    except Exception as exc:
        print(f"  {item['id']:8} :{item['port']}  DOWN  {exc}")
        ok = False
        continue
    ids = []
    if isinstance(data, dict):
        for row in data.get("data") or []:
            if isinstance(row, dict) and row.get("id"):
                ids.append(row["id"])
    print(f"  {item['id']:8} :{item['port']}  HTTP {status}  models={ids or data}")
    if status >= 400:
        ok = False
raise SystemExit(0 if ok else 1)
PY
