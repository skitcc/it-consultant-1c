#!/usr/bin/env python3
"""Replay a mail-like Ollama pipeline and print load vs compute times."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from _http import ns_to_sec, request_json  # noqa: E402
from catalog import load_dotenv  # noqa: E402

QUERY = "Как в 1С настроить обмен данными с бухгалтерией?"
DOC = (
    "В разделе Администрирование — Обмен данными укажите узел, "
    "правила регистрации и расписание."
)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="rerank RAG_CANDIDATES times (default: 3 smoke calls)",
    )
    parser.add_argument("--candidates", type=int, default=0)
    args = parser.parse_args()
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    llm = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
    embed_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    rerank_model = os.environ.get(
        "RERANK_MODEL_OLLAMA",
        "dengcao/Qwen3-Reranker-8B:Q8_0",
    )
    n = args.candidates
    if n <= 0:
        if args.full:
            n = int(os.environ.get("RAG_CANDIDATES", "20"))
        else:
            n = 3

    steps: list[dict] = []
    print(f"Trace against {base}")
    print(f"embed={embed_model}  rerank={rerank_model} x{n}  llm={llm}")
    steps.append(_timed("embed", _embed, base, embed_model))
    for index in range(1, n + 1):
        steps.append(
            _timed(
                f"rerank_{index}/{n}",
                _rerank,
                base,
                rerank_model,
            )
        )
    steps.append(_timed("llm", _llm, base, llm))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "ollama",
        "base_url": base,
        "candidates": n,
        "steps": steps,
        "totals": _totals(steps),
        "reading": _reading(steps),
    }
    out = ROOT / "out" / "ollama_pipeline_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _print_table(steps)
    print("Totals:", report["totals"])
    print("How to read this:")
    for line in report["reading"]:
        print(f"  * {line}")
    print(f"JSON: {out}")
    return 0 if all(step.get("ok") for step in steps) else 1


def _timed(name: str, fn, *args) -> dict:
    started = time.perf_counter()
    try:
        payload = fn(*args)
        error = None
        ok = True
    except Exception as exc:
        payload = {}
        error = str(exc)
        ok = False
    wall = time.perf_counter() - started
    row = {
        "step": name,
        "ok": ok,
        "error": error,
        "wall_sec": round(wall, 3),
        "load_sec": _sec(payload.get("load_duration")),
        "prompt_eval_sec": _sec(payload.get("prompt_eval_duration")),
        "eval_sec": _sec(payload.get("eval_duration")),
        "prompt_tokens": payload.get("prompt_eval_count"),
        "eval_tokens": payload.get("eval_count"),
        "done_reason": payload.get("done_reason"),
    }
    print(
        f"{name:16} wall={row['wall_sec']:8.3f}s  "
        f"load={_fmt(row['load_sec'])}  "
        f"prefill={_fmt(row['prompt_eval_sec'])}  "
        f"eval={_fmt(row['eval_sec'])}  "
        f"tok={row['prompt_tokens']}/{row['eval_tokens']}"
        + (f"  ERROR {error}" if error else "")
    )
    return row


def _embed(base: str, model: str) -> dict:
    _status, data = request_json(
        f"{base}/api/embed",
        method="POST",
        payload={
            "model": model,
            "input": QUERY,
            "keep_alive": -1,
            "options": {"num_ctx": 2048},
        },
        timeout=180,
    )
    if not isinstance(data, dict):
        raise RuntimeError("embed: bad response")
    if _status >= 400:
        raise RuntimeError(data)
    return data


def _rerank(base: str, model: str) -> dict:
    _status, data = request_json(
        f"{base}/api/chat",
        method="POST",
        payload={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"<Query>: {QUERY}\n<Document>: {DOC}",
                }
            ],
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 16},
        },
        timeout=180,
    )
    if not isinstance(data, dict) or _status >= 400:
        raise RuntimeError(data)
    return data


def _llm(base: str, model: str) -> dict:
    _status, data = request_json(
        f"{base}/api/chat",
        method="POST",
        payload={
            "model": model,
            "messages": [
                {"role": "system", "content": "Ответь одним коротким предложением."},
                {"role": "user", "content": QUERY},
            ],
            "stream": False,
            "think": "medium",
            "keep_alive": -1,
            "options": {
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": 64,
            },
        },
        timeout=420,
    )
    if not isinstance(data, dict) or _status >= 400:
        raise RuntimeError(data)
    return data


def _sec(value: object) -> float | None:
    parsed = ns_to_sec(value)
    return None if parsed is None else round(parsed, 3)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:7.3f}s"


def _totals(steps: list[dict]) -> dict:
    def _sum(key: str) -> float:
        return round(sum(float(step.get(key) or 0) for step in steps), 3)

    return {
        "wall_sec": _sum("wall_sec"),
        "load_sec": _sum("load_sec"),
        "prompt_eval_sec": _sum("prompt_eval_sec"),
        "eval_sec": _sum("eval_sec"),
    }


def _reading(steps: list[dict]) -> list[str]:
    notes: list[str] = []
    llm = next((step for step in steps if step["step"] == "llm"), None)
    embed = next((step for step in steps if step["step"] == "embed"), None)
    reranks = [step for step in steps if step["step"].startswith("rerank_")]
    if llm and (llm.get("load_sec") or 0) >= 2:
        notes.append(
            f"LLM load_sec={llm['load_sec']} after embed/rerank → weights were "
            "not resident. This is eviction (H1/H2), not 'slow thinking'."
        )
    elif llm and (llm.get("load_sec") or 0) < 0.5:
        notes.append(
            "LLM load_sec is tiny: the model was already in VRAM. "
            "If the user still waits, look at eval_sec (thinking) or queueing."
        )
    if embed and (embed.get("load_sec") or 0) >= 1:
        notes.append(f"Embedding load_sec={embed['load_sec']}: nomic was not in VRAM.")
    if reranks:
        first = reranks[0].get("load_sec") or 0
        rest = [step.get("load_sec") or 0 for step in reranks[1:]]
        if first >= 2 and rest and max(rest) < 0.5:
            notes.append(
                "Reranker paid load on the first candidate only. Later calls hit a "
                "warm runner — still 20 sequential generations on a real letter."
            )
        if rest and max(rest) >= 2:
            notes.append(
                "Reranker load_sec stays high on later candidates: runner is "
                "crashing or reloading (OOM / num_ctx)."
            )
    if llm and (llm.get("eval_sec") or 0) >= 10 and (llm.get("load_sec") or 0) < 1:
        notes.append(
            "LLM eval_sec is large with almost no load. That is generation "
            "(think=medium), not GPU reload."
        )
    if not notes:
        notes.append("No strong eviction signal in this single trace. Repeat after idle.")
    return notes


def _print_table(steps: list[dict]) -> None:
    print()
    print(f"{'step':16} {'wall':>10} {'load':>10} {'prefill':>10} {'eval':>10}")
    for step in steps:
        print(
            f"{step['step']:16} {_fmt(step['wall_sec']):>10} "
            f"{_fmt(step['load_sec']):>10} {_fmt(step['prompt_eval_sec']):>10} "
            f"{_fmt(step['eval_sec']):>10}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
