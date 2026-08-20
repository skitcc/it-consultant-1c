#!/usr/bin/env bash
# Sum gpu_memory_utilization across models.yaml + .env overrides.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

py - <<'PY'
from catalog import gpu_util_sum, load_dotenv, load_models

load_dotenv()
models = load_models()
total = gpu_util_sum(models)
print(f"{'id':8} {'port':>6} {'util':>6} {'max_len':>8} {'seqs':>6}  model")
for item in models:
    print(
        f"{item['id']:8} {item['port']:6d} {item['gpu_memory_utilization']:6.2f} "
        f"{item['max_model_len']:8d} {item['max_num_seqs']:6d}  {item['hf_id']}"
    )
print(f"\nSum gpu_memory_utilization = {total:.2f}  (keep <= 0.90)")
if total > 0.90:
    print("FAIL: sum is above 0.90. Lower LLM_GPU_UTIL / side-model util in .env")
    raise SystemExit(1)
if total > 0.85:
    print("WARN: tight budget. First boot of gpt-oss-120b may still OOM; watch nvidia-smi.")
else:
    print("OK: budget has headroom for CUDA context.")
PY
