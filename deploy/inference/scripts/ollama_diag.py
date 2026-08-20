#!/usr/bin/env python3
"""Snapshot of a running Ollama: tags, loaded models, GPU, env, verdict."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from _http import request_json  # noqa: E402
from catalog import load_dotenv  # noqa: E402

WATCH_ENV = (
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_MAX_QUEUE",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_CONTEXT_LENGTH",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_GPU_OVERHEAD",
    "OLLAMA_DEBUG",
)


def main() -> int:
    load_dotenv()
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    expected = [
        os.environ.get("OLLAMA_MODEL", "gpt-oss:120b"),
        os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
        os.environ.get("RERANK_MODEL_OLLAMA", "dengcao/Qwen3-Reranker-8B:Q8_0"),
        os.environ.get("VLM_MODEL_OLLAMA", "qwen3-vl:8b"),
    ]
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ollama_base_url": base,
        "expected_models": expected,
        "version": None,
        "tags": [],
        "loaded": [],
        "ollama_env": {},
        "nvidia_smi": None,
        "verdict": [],
        "errors": [],
    }

    try:
        _status, version = request_json(f"{base}/api/version", timeout=10)
        report["version"] = version
    except Exception as exc:
        report["errors"].append(f"version: {exc}")

    try:
        _status, tags = request_json(f"{base}/api/tags", timeout=15)
        models = tags.get("models") if isinstance(tags, dict) else None
        report["tags"] = [
            {
                "name": item.get("name"),
                "size": item.get("size"),
                "parameter_size": (item.get("details") or {}).get("parameter_size"),
                "quantization": (item.get("details") or {}).get("quantization_level"),
            }
            for item in (models or [])
            if isinstance(item, dict)
        ]
    except Exception as exc:
        report["errors"].append(f"tags: {exc}")

    try:
        _status, ps = request_json(f"{base}/api/ps", timeout=15)
        loaded = ps.get("models") if isinstance(ps, dict) else None
        report["loaded"] = [
            {
                "name": item.get("name") or item.get("model"),
                "size": item.get("size"),
                "size_vram": item.get("size_vram"),
                "expires_at": item.get("expires_at"),
                "processor": _processor(item),
            }
            for item in (loaded or [])
            if isinstance(item, dict)
        ]
    except Exception as exc:
        report["errors"].append(f"ps: {exc}")

    report["ollama_env"] = {
        key: os.environ.get(key)
        for key in WATCH_ENV
        if os.environ.get(key) is not None
    }
    if shutil.which("printenv"):
        try:
            proc = subprocess.run(
                ["printenv"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in proc.stdout.splitlines():
                if line.startswith("OLLAMA_"):
                    key, _, value = line.partition("=")
                    report["ollama_env"].setdefault(key, value)
        except Exception:
            pass

    report["nvidia_smi"] = _nvidia_smi()
    report["verdict"] = _verdict(report, expected)

    out_dir = ROOT / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ollama_diag.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _print_human(report, out_path)
    return 1 if report["errors"] and not report["version"] else 0


def _processor(item: dict) -> str:
    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    for key in ("processor", "PROCESSOR"):
        value = item.get(key) or details.get(key)
        if value:
            return str(value)
    vram = item.get("size_vram")
    size = item.get("size")
    if isinstance(vram, int) and isinstance(size, int) and size > 0:
        pct = 100.0 * vram / size
        if pct < 50:
            return f"mostly-cpu ({pct:.0f}% vram)"
        return f"gpu ({pct:.0f}% vram)"
    return "unknown"


def _nvidia_smi() -> dict | str | None:
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not on PATH (run this on GPU server 2)"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return proc.stderr.strip() or proc.stdout.strip()
        rows = []
        for line in proc.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "name": parts[0],
                        "memory_total_mib": parts[1],
                        "memory_used_mib": parts[2],
                        "memory_free_mib": parts[3],
                        "utilization_gpu_pct": parts[4],
                    }
                )
        procs = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "gpus": rows,
            "compute_apps": procs.stdout.strip().splitlines(),
        }
    except Exception as exc:
        return str(exc)


def _verdict(report: dict, expected: list[str]) -> list[str]:
    notes: list[str] = []
    loaded = report.get("loaded") or []
    names = [str(item.get("name") or "") for item in loaded]
    notes.append(f"Loaded models now: {len(loaded)} (Ollama default max is 3 per GPU).")
    missing = [name for name in expected if not any(name in loaded_name for loaded_name in names)]
    if missing:
        notes.append(
            "Not resident: "
            + ", ".join(missing)
            + ". If a mail request needs these, Ollama will load them and may evict gpt-oss."
        )
    else:
        notes.append("All four expected models are in /api/ps right now.")
    for item in loaded:
        processor = str(item.get("processor") or "")
        if "cpu" in processor.lower() and "gpu (100" not in processor.lower():
            notes.append(
                f"{item.get('name')} looks CPU-offloaded ({processor}). "
                "That is slow even without reload."
            )
        expires = item.get("expires_at")
        if expires and str(expires) not in {"", "0001-01-01T00:00:00Z"}:
            if "0001-01-01" not in str(expires):
                notes.append(
                    f"{item.get('name')} expires_at={expires}. "
                    "keep_alive is not infinite; idle unload is possible."
                )
    max_loaded = (report.get("ollama_env") or {}).get("OLLAMA_MAX_LOADED_MODELS")
    if max_loaded in {None, ""}:
        notes.append(
            "OLLAMA_MAX_LOADED_MODELS is unset (default 3 on one GPU). "
            "Four roles will evict each other even if VRAM would fit."
        )
    elif str(max_loaded) in {"1", "2", "3"}:
        notes.append(
            f"OLLAMA_MAX_LOADED_MODELS={max_loaded} cannot hold 4 models at once."
        )
    smi = report.get("nvidia_smi")
    if isinstance(smi, dict):
        for gpu in smi.get("gpus") or []:
            notes.append(
                f"GPU {gpu.get('name')}: used {gpu.get('memory_used_mib')} / "
                f"{gpu.get('memory_total_mib')} MiB, util {gpu.get('utilization_gpu_pct')}%."
            )
    if report.get("errors"):
        notes.append("Some API calls failed — see errors[]. Is OLLAMA_BASE_URL reachable?")
    notes.append(
        "Next: ./scripts/ollama_pipeline_trace.sh  "
        "(if LLM load_duration is large AFTER embed/rerank → eviction)."
    )
    return notes


def _print_human(report: dict, out_path: Path) -> None:
    print(f"Ollama {report.get('ollama_base_url')}  version={report.get('version')}")
    print("Installed tags:")
    if not report["tags"]:
        print("  (none or API failed)")
    for item in report["tags"]:
        print(
            f"  - {item.get('name')}  params={item.get('parameter_size')}  "
            f"quant={item.get('quantization')}  size={item.get('size')}"
        )
    print("Currently loaded (/api/ps):")
    if not report["loaded"]:
        print("  (nothing loaded)")
    for item in report["loaded"]:
        print(
            f"  - {item.get('name')}  vram={item.get('size_vram')}  "
            f"processor={item.get('processor')}  expires={item.get('expires_at')}"
        )
    env = report.get("ollama_env") or {}
    print("OLLAMA_* env (process + printenv):")
    if not env:
        print("  (none visible in this shell; check systemd unit on server 2)")
    for key in WATCH_ENV:
        if key in env:
            print(f"  {key}={env[key]}")
    smi = report.get("nvidia_smi")
    print("GPU:")
    print(f"  {smi}")
    print("Verdict:")
    for note in report["verdict"]:
        print(f"  * {note}")
    if report["errors"]:
        print("Errors:")
        for err in report["errors"]:
            print(f"  ! {err}")
    print(f"JSON: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
