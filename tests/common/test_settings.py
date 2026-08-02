"""Tests for shared Settings."""

from __future__ import annotations

from common import Settings


def test_settings_mail_and_reindex_fields() -> None:
    settings = Settings(
        _env_file=None,
        EWS_SERVER="mail.example.com",
        EWS_EMAIL="bot@example.com",
        EWS_PASSWORD="secret",
        OLLAMA_BASE_URL="http://127.0.0.1:11434",
        OLLAMA_MODEL="llama3.2",
        QDRANT_URL="http://127.0.0.1:6333",
        QDRANT_COLLECTION="docs",
        EMBEDDING_MODEL="nomic-embed-text",
        RAG_CANDIDATES=20,
        RAG_TOP_K=8,
        RERANK_ENABLED=True,
        RERANK_MODEL="bge-reranker-v2-m3",
        WATCH_PATH="/var/db",
        DEBOUNCE_SECONDS=1.5,
        LOG_LEVEL="DEBUG",
    )
    assert settings.ews_server == "mail.example.com"
    assert settings.ollama_model == "llama3.2"
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "docs"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.rag_candidates == 20
    assert settings.rag_top_k == 8
    assert settings.rerank_enabled is True
    assert settings.rerank_model == "bge-reranker-v2-m3"
    assert settings.watch_path == "/var/db"
    assert settings.debounce_seconds == 1.5
    assert settings.log_level == "DEBUG"
