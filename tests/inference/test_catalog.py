from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "deploy" / "inference"
sys.path.insert(0, str(ROOT))

from catalog import _apply_overrides, _parse_models_yaml, gpu_util_sum  # noqa: E402


def test_models_yaml_parses_four_services() -> None:
    models = _parse_models_yaml(ROOT / "models.yaml")
    ids = [item["id"] for item in models]
    assert ids == ["llm", "rerank", "vlm", "embed"]
    llm = models[0]
    assert llm["hf_id"] == "openai/gpt-oss-120b"
    assert llm["port"] == 8001
    assert "--reasoning-parser" in llm["extra_args"]
    rerank = models[1]
    assert any("Qwen3ForSequenceClassification" in str(arg) for arg in rerank["extra_args"])
    total = gpu_util_sum(models)
    assert 0.8 <= total <= 0.91


def test_env_overrides_gpu_util(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GPU_UTIL", "0.50")
    monkeypatch.setenv("LLM_PORT", "9001")
    model = _apply_overrides(
        {
            "id": "llm",
            "hf_id": "openai/gpt-oss-120b",
            "port": 8001,
            "gpu_memory_utilization": 0.58,
            "max_model_len": 8192,
            "max_num_seqs": 4,
            "extra_args": [],
        }
    )
    assert model["gpu_memory_utilization"] == 0.50
    assert model["port"] == 9001
