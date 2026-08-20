"""Select Ollama vs vLLM clients from ``INFERENCE_BACKEND``."""

from __future__ import annotations

from common.embeddings import Embedder, OllamaEmbedder, OpenAIEmbedder
from common.settings import Settings


class InferenceConfigError(ValueError):
    """vLLM backend is selected but a required URL/model is missing."""


def is_vllm(settings: Settings) -> bool:
    return settings.inference_backend == "vllm"


def resolved_llm_model(settings: Settings) -> str:
    if is_vllm(settings):
        return (settings.llm_model or settings.ollama_model).strip()
    return settings.ollama_model.strip()


def build_embedder(settings: Settings) -> Embedder:
    if is_vllm(settings):
        return OpenAIEmbedder(
            base_url=_require(settings.embedding_base_url, "EMBEDDING_BASE_URL"),
            model=settings.embedding_model,
            timeout_sec=settings.embedding_timeout_sec,
        )
    return OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        timeout_sec=settings.embedding_timeout_sec,
    )


def llm_base_url(settings: Settings) -> str:
    if is_vllm(settings):
        return _require(settings.llm_base_url, "LLM_BASE_URL")
    return settings.ollama_base_url


def rerank_base_url(settings: Settings) -> str:
    if is_vllm(settings):
        return _require(
            settings.rerank_base_url or None,
            "RERANK_BASE_URL",
        )
    return settings.rerank_base_url or settings.ollama_base_url


def vlm_base_url(settings: Settings) -> str:
    if is_vllm(settings):
        return _require(settings.vlm_base_url, "VLM_BASE_URL")
    return settings.ollama_base_url


def _require(value: str | None, name: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise InferenceConfigError(
            f"{name} is required when INFERENCE_BACKEND=vllm"
        )
    return cleaned
