"""Retrieve candidates from Qdrant, then rerank and expand neighbors."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mail_gateway.adapters.rag.neighbors import expand_neighbor_chunks
from mail_gateway.domain.models import DocumentChunk
from mail_gateway.ports import DocumentRetriever, Reranker

logger = logging.getLogger(__name__)


class RerankingRetriever:
    """DocumentRetriever: candidates → rerank → top_k → same-section neighbors."""

    def __init__(
        self,
        *,
        base: DocumentRetriever,
        reranker: Reranker,
        top_k: int = 8,
        neighbor_window: int = 1,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._base = base
        self._reranker = reranker
        self._top_k = top_k
        self._neighbor_window = neighbor_window

    def retrieve(self, query: str) -> list[DocumentChunk]:
        candidates = list(self._base.retrieve(query))
        if not candidates:
            return []

        ranked = list(self._reranker.rerank(query, candidates))
        selected = ranked[: self._top_k]

        pool: Sequence[DocumentChunk] = candidates
        load_neighbors = getattr(self._base, "load_neighbors", None)
        if callable(load_neighbors) and self._neighbor_window > 0:
            try:
                pool = list(load_neighbors(selected, window=self._neighbor_window))
                # Keep original candidates too so expand can still use them.
                merged: dict[tuple[str, int], DocumentChunk] = {
                    (chunk.source_path, chunk.chunk_index): chunk for chunk in candidates
                }
                for chunk in pool:
                    merged[(chunk.source_path, chunk.chunk_index)] = chunk
                pool = list(merged.values())
            except Exception:
                logger.exception("Neighbor load failed; expanding from candidates only")
                pool = candidates

        expanded = expand_neighbor_chunks(
            selected,
            pool,
            window=self._neighbor_window,
        )
        logger.info(
            "RerankingRetriever query_len=%s candidates=%s selected=%s expanded=%s",
            len(query.strip()),
            len(candidates),
            len(selected),
            len(expanded),
        )
        return expanded
