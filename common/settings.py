"""Shared application settings for all services."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single settings object for mail_gateway and reindex (env / .env)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- mail_gateway ---
    ews_server: str = Field(alias="EWS_SERVER")
    ews_email: str = Field(alias="EWS_EMAIL")
    ews_password: str = Field(alias="EWS_PASSWORD")
    ews_username: str | None = Field(default=None, alias="EWS_USERNAME")
    ews_auth: str = Field(default="ntlm", alias="EWS_AUTH")
    ews_verify_ssl: bool = Field(default=True, alias="EWS_VERIFY_SSL")
    ews_streaming_timeout_minutes: int = Field(
        default=30,
        alias="EWS_STREAMING_TIMEOUT_MINUTES",
    )
    ews_session_pool_size: int = Field(default=2, alias="EWS_SESSION_POOL_SIZE")
    ews_ignore_own_mail: bool = Field(default=True, alias="EWS_IGNORE_OWN_MAIL")
    ews_catchup_minutes: int = Field(default=30, alias="EWS_CATCHUP_MINUTES")

    ai_system_prompt: str | None = Field(default=None, alias="AI_SYSTEM_PROMPT")
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    ollama_timeout_sec: float = Field(default=300.0, alias="OLLAMA_TIMEOUT_SEC")

    reconnect_delay_sec: float = Field(default=5.0, alias="RECONNECT_DELAY_SEC")

    # --- RAG / Qdrant (shared by mail_gateway and reindex) ---
    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="docs", alias="QDRANT_COLLECTION")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    embedding_timeout_sec: float = Field(
        default=120.0,
        alias="EMBEDDING_TIMEOUT_SEC",
    )
    rag_top_k: int = Field(default=8, alias="RAG_TOP_K")
    rag_candidates: int = Field(default=20, alias="RAG_CANDIDATES")
    rag_score_threshold: float | None = Field(
        default=None,
        alias="RAG_SCORE_THRESHOLD",
    )
    rag_neighbor_window: int = Field(default=1, alias="RAG_NEIGHBOR_WINDOW")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_model: str = Field(
        default="dengcao/Qwen3-Reranker-8B:Q8_0",
        alias="RERANK_MODEL",
    )
    rerank_base_url: str | None = Field(default=None, alias="RERANK_BASE_URL")
    rerank_timeout_sec: float = Field(default=60.0, alias="RERANK_TIMEOUT_SEC")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    # --- reindex picture description (Docling enrichment via Ollama VLM) ---
    picture_description_enabled: bool = Field(
        default=True,
        alias="PICTURE_DESCRIPTION_ENABLED",
    )
    vlm_model: str = Field(default="qwen3-vl:8b", alias="VLM_MODEL")
    vlm_timeout_sec: float = Field(default=90.0, alias="VLM_TIMEOUT_SEC")
    picture_area_threshold: float = Field(
        default=0.02,
        alias="PICTURE_AREA_THRESHOLD",
    )

    # --- reindex ---
    watch_path: str = Field(
        default="/var/lib/it-consultant/db",
        alias="WATCH_PATH",
    )
    debounce_seconds: float = Field(default=1.0, alias="DEBOUNCE_SECONDS")
    index_extensions: str = Field(
        default=".txt,.md,.markdown,.rst,.log,.csv,.pdf,.docx,.pptx,.xlsx,.xls,.html,.htm",
        alias="INDEX_EXTENSIONS",
    )

    # --- shared ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("ews_username", mode="before")
    @classmethod
    def empty_username_as_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value
