#!/usr/bin/env python3
"""Load four Ollama models with keep_alive=-1 and print /api/ps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from _http import ns_to_sec, request_json  # noqa: E402
from catalog import load_dotenv  # noqa: E402


def main() -> int:
    load_dotenv()
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    jobs = [
        (
            "embed",
            f"{base}/api/embed",
            {
                "model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
                "input": "ping",
                "keep_alive": -1,
            },
        ),
        (
            "rerank",
            f"{base}/api/chat",
            {
                "model": os.environ.get(
                    "RERANK_MODEL_OLLAMA",
                    "dengcao/Qwen3-Reranker-8B:Q8_0",
                ),
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {"num_predict": 1},
            },
        ),
        (
            "vlm",
            f"{base}/api/chat",
            {
                "model": os.environ.get("VLM_MODEL_OLLAMA", "qwen3-vl:8b"),
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 1},
            },
        ),
        (
            "llm",
            f"{base}/api/generate",
            {
                "model": os.environ.get("OLLAMA_MODEL", "gpt-oss:120b"),
                "prompt": "",
                "keep_alive": -1,
            },
        ),
    ]
    print(f"Preload against {base}")
    failed = 0
    for name, url, payload in jobs:
        try:
            status, data = request_json(url, method="POST", payload=payload, timeout=600)
        except Exception as exc:
            print(f"  {name}: ERROR {exc}")
            failed += 1
            continue
        load = ns_to_sec(data.get("load_duration") if isinstance(data, dict) else None)
        err = None
        if isinstance(data, dict):
            err = data.get("error")
        if status >= 400 or err:
            print(f"  {name}: HTTP {status} load={load} error={err or data}")
            failed += 1
        else:
            print(f"  {name}: ok load_sec={load}")
    status, ps = request_json(f"{base}/api/ps", timeout=15)
    models = (ps or {}).get("models") if isinstance(ps, dict) else []
    print(f"\n/api/ps ({len(models or [])} loaded):")
    print(json.dumps(ps, indent=2, ensure_ascii=False))
    count = len(models or [])
    if count < 4:
        print(
            f"\nOnly {count}/4 models stayed loaded. They do not fit together "
            "(VRAM or OLLAMA_MAX_LOADED_MODELS)."
        )
    else:
        print("\nAll four stayed loaded. Eviction on a real letter is less likely.")
    (ROOT / "out").mkdir(parents=True, exist_ok=True)
    (ROOT / "out" / "ollama_preload.json").write_text(
        json.dumps({"ps": ps, "failed": failed}, indent=2) + "\n"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
