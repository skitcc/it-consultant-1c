"""Knowledge retrieval pipeline."""

from __future__ import annotations

from knowledge.core.domain import DocumentChunk
from knowledge.core.ports import Embedder, Reranker, VectorIndex


class RetrieveKnowledge:
    """Embed query, search candidates, rerank, select, then load neighbors."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_index: VectorIndex,
        reranker: Reranker,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._candidate_limit = candidate_limit
        self._score_threshold = score_threshold

    def execute(
        self,
        query: str,
        *,
        knowledge_id: str = "main",
        top_k: int = 5,
        neighbor_window: int = 1,
    ) -> list[DocumentChunk]:
        cleaned = query.strip()
        if not cleaned:
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")

        query_vector = self._embedder.embed(cleaned)
        candidates = list(
            self._vector_index.search(
                query_vector,
                knowledge_id=knowledge_id,
                limit=self._candidate_limit,
                score_threshold=self._score_threshold,
            )
        )
        if not candidates:
            return []
        selected = list(self._reranker.rerank(cleaned, candidates))[:top_k]
        if not selected:
            return []
        expanded = list(
            self._vector_index.load_neighbors(
                selected,
                knowledge_id=knowledge_id,
                window=neighbor_window,
            )
        )
        return expanded or selected

    __call__ = execute
