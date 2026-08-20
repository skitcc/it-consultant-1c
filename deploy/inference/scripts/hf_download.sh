#!/usr/bin/env bash
# Download Hugging Face weights into HF_HOME (mounted into vLLM containers).
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

hf_home="${HF_HOME:-/opt/hf-cache}"
mkdir -p "$hf_home"
export HF_HOME="$hf_home"
export HUGGINGFACE_HUB_CACHE="$hf_home"

if command -v hf >/dev/null 2>&1; then
  dl=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  dl=(huggingface-cli download)
else
  echo "Install Hugging Face CLI first:" >&2
  echo "  pip install -U huggingface_hub" >&2
  echo "  hf download --help" >&2
  exit 1
fi

mapfile -t models < <(py - <<'PY'
from catalog import load_dotenv, load_models
load_dotenv()
for item in load_models():
    print(item["hf_id"])
PY
)

echo "HF_HOME=$hf_home"
echo "Downloading ${#models[@]} models (gpt-oss-120b alone is ~61GB; total ~95GB+)"
for model in "${models[@]}"; do
  echo
  echo "==> $model"
  "${dl[@]}" "$model"
done

echo
echo "Cache size:"
du -sh "$hf_home" || true
echo "If a model is gated, run: huggingface-cli login"
echo "Offline copy: rsync -aP \$HF_HOME/ user@server2:$hf_home/"
