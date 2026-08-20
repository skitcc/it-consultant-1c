#!/usr/bin/env python3
"""Compare one embedding from Ollama vs vLLM (cosine + dimension)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from _http import request_json  # noqa: E402
from catalog import env, load_dotenv, load_models  # noqa: E402

TEXT = "настройка обмена данными в 1С"


def main() -> int:
    load_dotenv()
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    models = {item["id"]: item for item in load_models()}
    vllm_url = (
        env("EMBEDDING_BASE_URL")
        or f"http://127.0.0.1:{models['embed']['port']}/v1"
    ).rstrip("/")
    vllm_model = env("EMBED_SERVED_NAME", "embed")

    print(f"Ollama {ollama_url} model={ollama_model}")
    status, ollama = request_json(
        f"{ollama_url}/api/embed",
        method="POST",
        payload={"model": ollama_model, "input": TEXT, "keep_alive": -1},
        timeout=120,
    )
    if status >= 400:
        print("Ollama failed:", ollama)
        return 1
    vec_a = _first_vector(ollama)
    print(f"vLLM   {vllm_url} model={vllm_model}")
    status, vllm = request_json(
        f"{vllm_url}/embeddings",
        method="POST",
        payload={"model": vllm_model, "input": TEXT},
        timeout=120,
    )
    if status >= 400:
        print("vLLM failed:", vllm)
        return 1
    vec_b = _openai_vector(vllm)
    if vec_a is None or vec_b is None:
        print("Could not parse vectors", vec_a is None, vec_b is None)
        return 1
    print(f"dim ollama={len(vec_a)}  vllm={len(vec_b)}")
    if len(vec_a) != len(vec_b):
        print("FAIL: different dimensions. Recreate the Qdrant collection and reindex.")
        return 1
    cosine = _cosine(vec_a, vec_b)
    print(f"cosine(same text) = {cosine:.6f}")
    print("How to read:")
    print("  > 0.99  vectors match — keep the current Qdrant index")
    print("  0.90–0.99  close but not identical — spot-check RAG, prefer reindex")
    print("  < 0.90  different space (prefixes/model). Run: python -m reindex --once")
    return 0 if cosine >= 0.99 else 2


def _first_vector(data: object) -> list[float] | None:
    if not isinstance(data, dict):
        return None
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        item = embeddings[0]
        if isinstance(item, list):
            return [float(x) for x in item]
    embedding = data.get("embedding")
    if isinstance(embedding, list):
        return [float(x) for x in embedding]
    return None


def _openai_vector(data: object) -> list[float] | None:
    if not isinstance(data, dict):
        return None
    rows = data.get("data")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        embedding = rows[0].get("embedding")
        if isinstance(embedding, list):
            return [float(x) for x in embedding]
    return None


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    na = math.sqrt(sum(a * a for a in left))
    nb = math.sqrt(sum(b * b for b in right))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


if __name__ == "__main__":
    raise SystemExit(main())
