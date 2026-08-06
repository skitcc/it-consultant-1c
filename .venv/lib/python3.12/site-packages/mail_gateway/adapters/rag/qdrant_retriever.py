"""Qdrant-backed document retrieval for RAG."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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
        limit: int = 20,
        score_threshold: float | None = None,
    ) -> None:
        self._client = QdrantClient(url=qdrant_url, check_compatibility=False)
        self._collection = collection
        self._embedder = embedder
        self._limit = limit
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
            "limit": self._limit,
            "with_payload": True,
        }
        if self._score_threshold is not None:
            kwargs["score_threshold"] = self._score_threshold

        response = self._client.query_points(**kwargs)
        hits = getattr(response, "points", None) or []
        chunks = [_chunk_from_hit(hit) for hit in hits]
        chunks = [chunk for chunk in chunks if chunk is not None]

        logger.info(
            "Qdrant retrieve collection=%s query_len=%s hits=%s",
            self._collection,
            len(cleaned),
            len(chunks),
        )
        return chunks

    def load_neighbors(
        self,
        seeds: Sequence[DocumentChunk],
        *,
        window: int = 1,
    ) -> list[DocumentChunk]:
        """Load chunks with nearby chunk_index for each seed source_path."""
        if not seeds or window < 0:
            return []
        if not self._client.collection_exists(self._collection):
            return []

        wanted_by_source: dict[str, set[int]] = {}
        for seed in seeds:
            indexes = wanted_by_source.setdefault(seed.source_path, set())
            for delta in range(-window, window + 1):
                index = seed.chunk_index + delta
                if index >= 0:
                    indexes.add(index)

        loaded: list[DocumentChunk] = []
        for source_path, indexes in wanted_by_source.items():
            if not indexes:
                continue
            scroll_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_path",
                        match=qmodels.MatchValue(value=source_path),
                    ),
                    qmodels.FieldCondition(
                        key="chunk_index",
                        match=qmodels.MatchAny(any=sorted(indexes)),
                    ),
                ]
            )
            points, _next = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                with_payload=True,
                limit=max(len(indexes) * 2, 16),
            )
            for point in points:
                chunk = _chunk_from_payload(getattr(point, "payload", None), score=None)
                if chunk is not None:
                    loaded.append(chunk)

        logger.info(
            "Qdrant neighbors sources=%s loaded=%s",
            len(wanted_by_source),
            len(loaded),
        )
        return loaded


def _chunk_from_hit(hit: object) -> DocumentChunk | None:
    payload = getattr(hit, "payload", None)
    score = getattr(hit, "score", None)
    return _chunk_from_payload(
        payload,
        score=float(score) if score is not None else None,
    )


def _chunk_from_payload(
    payload: object,
    *,
    score: float | None,
) -> DocumentChunk | None:
    if not isinstance(payload, dict):
        return None
    text = str(payload.get("text") or "").strip()
    if not text:
        return None
    return DocumentChunk(
        text=text,
        source_path=str(payload.get("source_path") or ""),
        chunk_index=int(payload.get("chunk_index") or 0),
        score=score,
    )
