"""Tests for INFERENCE_BACKEND factory helpers."""

from __future__ import annotations

import pytest

from common.embeddings import OllamaEmbedder, OpenAIEmbedder
from common.inference import (
    InferenceConfigError,
    build_embedder,
    llm_base_url,
    rerank_base_url,
    resolved_llm_model,
    vlm_base_url,
)
from common.settings import Settings
from mail_gateway.adapters.assistant.ollama_assistant import OllamaAssistant
from mail_gateway.adapters.assistant.openai_assistant import OpenAIAssistant
from mail_gateway.adapters.rag.ollama_reranker import OllamaReranker
from mail_gateway.adapters.rag.vllm_reranker import VllmReranker
from mail_gateway.main.app import build_assistant, build_reranker


def _settings(**overrides) -> Settings:
    base = {
        "EWS_SERVER": "mail.example.com",
        "EWS_EMAIL": "bot@example.com",
        "EWS_PASSWORD": "secret",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_ollama_factory_keeps_native_clients() -> None:
    settings = _settings()
    assert isinstance(build_embedder(settings), OllamaEmbedder)
    assert isinstance(build_assistant(settings), OllamaAssistant)
    assert isinstance(build_reranker(settings), OllamaReranker)
    assert llm_base_url(settings) == "http://127.0.0.1:11434"
    assert rerank_base_url(settings) == "http://127.0.0.1:11434"
    assert vlm_base_url(settings) == "http://127.0.0.1:11434"
    assert resolved_llm_model(settings) == "llama3.2"


def test_ollama_ignores_vllm_urls() -> None:
    settings = _settings(
        LLM_BASE_URL="http://vllm:8001/v1",
        EMBEDDING_BASE_URL="http://vllm:8004/v1",
        RERANK_BASE_URL="http://vllm:8002/v1",
        VLM_BASE_URL="http://vllm:8003/v1",
        LLM_MODEL="openai/gpt-oss-120b",
    )
    assert isinstance(build_embedder(settings), OllamaEmbedder)
    assert isinstance(build_assistant(settings), OllamaAssistant)
    assert llm_base_url(settings) == "http://127.0.0.1:11434"
    assert resolved_llm_model(settings) == "llama3.2"


def test_vllm_factory_requires_urls_and_uses_openai_clients() -> None:
    settings = _settings(
        INFERENCE_BACKEND="vllm",
        LLM_BASE_URL="http://gpu:8001/v1",
        LLM_MODEL="openai/gpt-oss-120b",
        EMBEDDING_BASE_URL="http://gpu:8004/v1",
        EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5",
        RERANK_BASE_URL="http://gpu:8002/v1",
        RERANK_MODEL="Qwen/Qwen3-Reranker-8B",
        VLM_BASE_URL="http://gpu:8003/v1",
        VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct",
    )
    embedder = build_embedder(settings)
    assistant = build_assistant(settings)
    reranker = build_reranker(settings)
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model == "nomic-ai/nomic-embed-text-v1.5"
    assert isinstance(assistant, OpenAIAssistant)
    assert isinstance(reranker, VllmReranker)
    assert llm_base_url(settings) == "http://gpu:8001/v1"
    assert vlm_base_url(settings) == "http://gpu:8003/v1"


def test_vllm_without_urls_raises() -> None:
    settings = _settings(INFERENCE_BACKEND="vllm")
    with pytest.raises(InferenceConfigError, match="EMBEDDING_BASE_URL"):
        build_embedder(settings)
    with pytest.raises(InferenceConfigError, match="LLM_BASE_URL"):
        build_assistant(settings)
