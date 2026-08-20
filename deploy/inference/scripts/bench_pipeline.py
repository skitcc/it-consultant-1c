#!/usr/bin/env python3
"""Benchmark embed → rerank → LLM against Ollama or vLLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from _http import ns_to_sec, request_json  # noqa: E402
from catalog import env, load_dotenv, load_models  # noqa: E402

QUERY = "Как в 1С настроить обмен данными с бухгалтерией?"
DOC = (
    "В разделе Администрирование — Обмен данными укажите узел, "
    "правила регистрации и расписание."
)
SYSTEM = (
    "Ты внутренний IT-консультант. Ответь кратко по документации: " + DOC
)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ollama", "vllm"), required=True)
    parser.add_argument("--full", action="store_true", help="20 rerank candidates")
    parser.add_argument("--candidates", type=int, default=0)
    parser.add_argument("--warm", action="store_true", help="one discarded warmup pass")
    parser.add_argument(
        "--evict",
        action="store_true",
        help="Ollama only: keep_alive=0 on all models before the run (cold-ish)",
    )
    parser.add_argument("--compare", type=Path, help="previous JSON to print a delta table")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    n = args.candidates
    if n <= 0:
        n = int(os.environ.get("RAG_CANDIDATES", "20") if args.full else 3)

    if args.backend == "ollama" and args.evict:
        _ollama_evict()
    if args.warm:
        print("Warmup pass (discarded)...")
        _run(args.backend, n, warmup=True)

    report = _run(args.backend, n, warmup=False)
    out = args.out or (ROOT / "out" / f"bench_{args.backend}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _print_report(report)
    if args.compare:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        _print_compare(other, report)
    print(f"JSON: {out}")
    return 0 if report["ok"] else 1


def _run(backend: str, candidates: int, *, warmup: bool) -> dict[str, Any]:
    steps: list[dict] = []
    if backend == "ollama":
        steps.append(_step("embed", _ollama_embed))
        for index in range(1, candidates + 1):
            steps.append(_step(f"rerank_{index}/{candidates}", _ollama_rerank))
        steps.append(_step("llm", _ollama_llm))
        rerank_http = candidates
    else:
        steps.append(_step("embed", _vllm_embed))
        steps.append(_step("rerank_batch", lambda: _vllm_rerank(candidates)))
        steps.append(_step("llm", _vllm_llm))
        rerank_http = 1
    totals = {
        "wall_sec": round(sum(s["wall_sec"] for s in steps), 3),
        "load_sec": round(sum(s.get("load_sec") or 0 for s in steps), 3),
        "rerank_http_calls": rerank_http,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "warmup": warmup,
        "candidates": candidates,
        "ok": all(step["ok"] for step in steps),
        "steps": steps,
        "totals": totals,
    }


def _step(name: str, fn) -> dict:
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    error = None
    ok = True
    try:
        extra = fn() or {}
    except Exception as exc:
        error = str(exc)
        ok = False
    wall = round(time.perf_counter() - started, 3)
    row = {
        "step": name,
        "ok": ok,
        "error": error,
        "wall_sec": wall,
        **extra,
    }
    print(
        f"{name:16} {wall:8.3f}s  "
        + (f"load={extra.get('load_sec')}s  " if extra.get("load_sec") is not None else "")
        + (f"ttft={extra.get('ttft_sec')}s  " if extra.get("ttft_sec") is not None else "")
        + (f"ERROR {error}" if error else "")
    )
    return row


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _ollama_embed() -> dict:
    status, data = request_json(
        f"{_ollama_base()}/api/embed",
        method="POST",
        payload={
            "model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
            "input": QUERY,
            "keep_alive": -1,
        },
        timeout=180,
    )
    _raise(status, data)
    return {"load_sec": ns_to_sec(data.get("load_duration"))}


def _ollama_rerank() -> dict:
    status, data = request_json(
        f"{_ollama_base()}/api/chat",
        method="POST",
        payload={
            "model": os.environ.get(
                "RERANK_MODEL_OLLAMA",
                "dengcao/Qwen3-Reranker-8B:Q8_0",
            ),
            "messages": [{"role": "user", "content": f"{QUERY}\n{DOC}"}],
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": {"num_predict": 16, "temperature": 0},
        },
        timeout=180,
    )
    _raise(status, data)
    return {"load_sec": ns_to_sec(data.get("load_duration"))}


def _ollama_llm() -> dict:
    status, data = request_json(
        f"{_ollama_base()}/api/chat",
        method="POST",
        payload={
            "model": os.environ.get("OLLAMA_MODEL", "gpt-oss:120b"),
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUERY},
            ],
            "stream": False,
            "think": "medium",
            "keep_alive": -1,
            "options": {"num_predict": 64, "temperature": 0, "num_ctx": 8192},
        },
        timeout=420,
    )
    _raise(status, data)
    eval_count = data.get("eval_count")
    eval_sec = ns_to_sec(data.get("eval_duration"))
    tps = None
    if isinstance(eval_count, int) and eval_sec:
        tps = round(eval_count / eval_sec, 2)
    return {
        "load_sec": ns_to_sec(data.get("load_duration")),
        "eval_sec": eval_sec,
        "eval_tokens": eval_count,
        "tokens_per_sec": tps,
    }


def _ollama_evict() -> None:
    base = _ollama_base()
    models = [
        os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
        os.environ.get("RERANK_MODEL_OLLAMA", "dengcao/Qwen3-Reranker-8B:Q8_0"),
        os.environ.get("VLM_MODEL_OLLAMA", "qwen3-vl:8b"),
        os.environ.get("OLLAMA_MODEL", "gpt-oss:120b"),
    ]
    print("Evicting Ollama models (keep_alive=0)...")
    for model in models:
        try:
            request_json(
                f"{base}/api/generate",
                method="POST",
                payload={"model": model, "prompt": "", "keep_alive": 0},
                timeout=60,
            )
            print(f"  unload {model}")
        except Exception as exc:
            print(f"  skip {model}: {exc}")


def _vllm_urls() -> dict[str, str]:
    models = {item["id"]: item for item in load_models()}
    return {
        "llm": env("LLM_BASE_URL")
        or f"http://127.0.0.1:{models['llm']['port']}/v1",
        "rerank": env("RERANK_BASE_URL")
        or f"http://127.0.0.1:{models['rerank']['port']}/v1",
        "embed": env("EMBEDDING_BASE_URL")
        or f"http://127.0.0.1:{models['embed']['port']}/v1",
    }


def _vllm_embed() -> dict:
    urls = _vllm_urls()
    status, data = request_json(
        f"{urls['embed'].rstrip('/')}/embeddings",
        method="POST",
        payload={
            "model": env("EMBED_SERVED_NAME", "embed"),
            "input": QUERY,
        },
        timeout=180,
    )
    _raise(status, data)
    return {}


def _vllm_rerank(candidates: int) -> dict:
    urls = _vllm_urls()
    docs = [f"{DOC} вариант {i}" for i in range(candidates)]
    status, data = request_json(
        f"{urls['rerank'].rstrip('/')}/rerank",
        method="POST",
        payload={
            "model": env("RERANK_SERVED_NAME", "rerank"),
            "query": QUERY,
            "documents": docs,
        },
        timeout=180,
    )
    _raise(status, data)
    results = data.get("results") if isinstance(data, dict) else None
    return {"rerank_results": len(results) if isinstance(results, list) else 0}


def _vllm_llm() -> dict:
    urls = _vllm_urls()
    payload = {
        "model": env("LLM_SERVED_NAME", "llm"),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUERY},
        ],
        "max_tokens": 64,
        "temperature": 0,
        "stream": False,
        "reasoning_effort": "medium",
    }
    started = time.perf_counter()
    status, data = request_json(
        f"{urls['llm'].rstrip('/')}/chat/completions",
        method="POST",
        payload=payload,
        timeout=420,
    )
    wall = time.perf_counter() - started
    _raise(status, data)
    usage = data.get("usage") if isinstance(data, dict) else {}
    completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
    tps = None
    if isinstance(completion, int) and wall:
        tps = round(completion / wall, 2)
    return {"eval_tokens": completion, "tokens_per_sec": tps, "ttft_sec": None}


def _raise(status: int, data: Any) -> None:
    if status >= 400:
        raise RuntimeError(data)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])


def _print_report(report: dict) -> None:
    print()
    print(
        f"backend={report['backend']} candidates={report['candidates']} "
        f"wall={report['totals']['wall_sec']}s "
        f"rerank_http={report['totals']['rerank_http_calls']}"
    )


def _print_compare(old: dict, new: dict) -> None:
    print()
    print(f"{'metric':24} {old.get('backend'):>12} {new.get('backend'):>12}")
    pairs = [
        ("wall_sec", old.get("totals", {}).get("wall_sec"), new["totals"]["wall_sec"]),
        (
            "rerank_http_calls",
            old.get("totals", {}).get("rerank_http_calls"),
            new["totals"]["rerank_http_calls"],
        ),
        (
            "load_sec",
            old.get("totals", {}).get("load_sec"),
            new["totals"]["load_sec"],
        ),
    ]
    for name, left, right in pairs:
        print(f"{name:24} {left!s:>12} {right!s:>12}")


if __name__ == "__main__":
    raise SystemExit(main())
