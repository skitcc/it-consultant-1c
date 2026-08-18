"""Runtime configuration shared by API Gateway and knowledge_sync."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KnowledgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    ollama_timeout_sec: float = Field(default=300.0, alias="OLLAMA_TIMEOUT_SEC")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    embedding_timeout_sec: float = Field(
        default=120.0,
        alias="EMBEDDING_TIMEOUT_SEC",
    )
    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="docs", alias="QDRANT_COLLECTION")
    rag_candidates: int = Field(default=20, alias="RAG_CANDIDATES")
    rag_top_k: int = Field(default=8, alias="RAG_TOP_K")
    rag_neighbor_window: int = Field(default=1, alias="RAG_NEIGHBOR_WINDOW")
    rag_score_threshold: float | None = Field(
        default=None,
        alias="RAG_SCORE_THRESHOLD",
    )
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_model: str = Field(
        default="dengcao/Qwen3-Reranker-8B:Q8_0",
        alias="RERANK_MODEL",
    )
    rerank_base_url: str | None = Field(default=None, alias="RERANK_BASE_URL")
    rerank_timeout_sec: float = Field(default=60.0, alias="RERANK_TIMEOUT_SEC")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
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
    document_registry_path: str = Field(
        default="/var/lib/it-consultant/registry.sqlite3",
        alias="DOCUMENT_REGISTRY_PATH",
    )
    default_knowledge_id: str = Field(
        default="main",
        alias="DEFAULT_KNOWLEDGE_ID",
    )
    index_extensions: str = Field(
        default=".txt,.md,.markdown,.rst,.log,.csv,.pdf,.docx,.pptx,.xlsx,.xls,.html,.htm",
        alias="INDEX_EXTENSIONS",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def allowed_extensions(self) -> frozenset[str]:
        values: set[str] = set()
        for item in self.index_extensions.split(","):
            suffix = item.strip().lower()
            if not suffix:
                continue
            values.add(suffix if suffix.startswith(".") else f".{suffix}")
        return frozenset(values)
