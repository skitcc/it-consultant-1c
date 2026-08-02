"""Qdrant-backed document retrieval for RAG."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from common.embeddings import OllamaEmbedder
from mail_gateway.domain.models import DocumentChunk
from mail_gateway.ports import DocumentRetriever

logger = logging.getLogger(__name__)


class QdrantRetriever(DocumentRetriever):
    """Embed the query with Ollama and search the Qdrant collection."""

    def __init__(
        self,
        *,
        qdrant_url: str,
        collection: str,
        embedder: OllamaEmbedder,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> None:
        self._client = QdrantClient(url=qdrant_url, check_compatibility=False)
        self._collection = collection
        self._embedder = embedder
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(self, query: str) -> list[DocumentChunk]:
        cleaned = query.strip()
        if not cleaned:
            return []

        if not self._client.collection_exists(self._collection):
            logger.warning(
                "Qdrant collection %s does not exist; returning no chunks",
                self._collection,
            )
            return []

        vector = self._embedder.embed(cleaned)
        kwargs: dict = {
            "collection_name": self._collection,
            "query": vector,
            "limit": self._top_k,
            "with_payload": True,
        }
        if self._score_threshold is not None:
            kwargs["score_threshold"] = self._score_threshold

        response = self._client.query_points(**kwargs)
        hits = getattr(response, "points", None) or []
        chunks: list[DocumentChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            chunks.append(
                DocumentChunk(
                    text=text,
                    source_path=str(payload.get("source_path") or ""),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    score=float(hit.score) if hit.score is not None else None,
                )
            )

        logger.info(
            "Qdrant retrieve collection=%s query_len=%s hits=%s",
            self._collection,
            len(cleaned),
            len(chunks),
        )
        return chunks
