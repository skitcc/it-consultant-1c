"""Load models.yaml and apply .env overrides. Stdlib only."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODELS_YAML = ROOT / "models.yaml"
ENV_FILE = ROOT / ".env"

_PREFIX = {
    "llm": "LLM",
    "rerank": "RERANK",
    "vlm": "VLM",
    "embed": "EMBED",
}

_OVERRIDE_KEYS = {
    "MODEL": "hf_id",
    "SERVED_NAME": "served_name",
    "PORT": "port",
    "GPU_UTIL": "gpu_memory_utilization",
    "MAX_MODEL_LEN": "max_model_len",
    "MAX_NUM_SEQS": "max_num_seqs",
}


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE from .env without overriding existing process env."""
    env_path = path or ENV_FILE
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        loaded[key] = value
        if key not in os.environ:
            os.environ[key] = value
    return loaded


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def load_models(path: Path | None = None) -> list[dict[str, Any]]:
    models = _parse_models_yaml(path or MODELS_YAML)
    return [_apply_overrides(model) for model in models]


def gpu_util_sum(models: list[dict[str, Any]] | None = None) -> float:
    items = models if models is not None else load_models()
    return sum(float(item["gpu_memory_utilization"]) for item in items)


def _apply_overrides(model: dict[str, Any]) -> dict[str, Any]:
    prefix = _PREFIX.get(str(model.get("id") or model.get("role") or ""))
    updated = dict(model)
    if not prefix:
        return updated
    for suffix, field in _OVERRIDE_KEYS.items():
        raw = env(f"{prefix}_{suffix}")
        if raw is None:
            continue
        if field in {"port", "max_model_len", "max_num_seqs"}:
            updated[field] = int(raw)
        elif field == "gpu_memory_utilization":
            updated[field] = float(raw)
        else:
            updated[field] = raw
    extra = env(f"{prefix}_EXTRA_ARGS")
    if extra is not None:
        updated["extra_args"] = shlex.split(extra)
    return updated


def _parse_models_yaml(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    models: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    extra: list[str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped == "models:":
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if stripped.startswith("- ") and indent <= 2:
            if current is not None:
                models.append(current)
            current = {}
            extra = None
            rest = stripped[2:]
            if ":" in rest:
                key, value = rest.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
            continue
        if current is None:
            continue
        if stripped == "extra_args:":
            extra = []
            current["extra_args"] = extra
            continue
        if extra is not None and stripped.startswith("- "):
            extra.append(_parse_scalar(stripped[2:]))
            continue
        if ":" in stripped:
            extra = None
            key, value = stripped.split(":", 1)
            current[key.strip()] = _parse_scalar(value)
    if current is not None:
        models.append(current)
    if not models:
        raise ValueError(f"no models parsed from {path}")
    return models


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text
