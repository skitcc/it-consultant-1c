"""Rerank adapters for RAG candidate reordering."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

import httpx

from mail_gateway.domain.models import DocumentChunk
from mail_gateway.ports import Reranker

logger = logging.getLogger(__name__)


class ScorePassthroughReranker:
    """Keep Qdrant vector-score order (fallback / rerank disabled)."""

    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> list[DocumentChunk]:
        del query  # unused; order comes from existing scores
        return sorted(
            chunks,
            key=lambda chunk: chunk.score if chunk.score is not None else float("-inf"),
            reverse=True,
        )


class OllamaReranker:
    """Call Ollama-compatible ``/api/rerank`` (or ``/v1/rerank``) when available.

    If the endpoint is missing or fails, falls back to vector-score order.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str,
        timeout_sec: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._fallback = ScorePassthroughReranker()

    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> list[DocumentChunk]:
        items = list(chunks)
        if len(items) <= 1:
            return items

        documents = [chunk.text for chunk in items]
        try:
            scores = self._score_documents(query, documents)
        except Exception:
            logger.exception(
                "Rerank failed model=%s; falling back to vector scores",
                self._model,
            )
            return self._fallback.rerank(query, items)

        if scores is None or len(scores) != len(items):
            logger.warning(
                "Rerank returned unexpected scores model=%s; using vector scores",
                self._model,
            )
            return self._fallback.rerank(query, items)

        ranked = [
            replace(chunk, score=score)
            for chunk, score in sorted(
                zip(items, scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
        logger.info(
            "Reranked candidates model=%s count=%s top_score=%.4f",
            self._model,
            len(ranked),
            ranked[0].score if ranked and ranked[0].score is not None else -1.0,
        )
        return ranked

    def _score_documents(self, query: str, documents: list[str]) -> list[float] | None:
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        with httpx.Client(timeout=self._timeout) as client:
            for path in ("/api/rerank", "/v1/rerank"):
                response = client.post(f"{self._base_url}{path}", json=payload)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                return _parse_rerank_scores(response.json(), expected=len(documents))
        logger.warning(
            "No rerank endpoint on %s (/api/rerank, /v1/rerank); using vector scores",
            self._base_url,
        )
        return None


def _parse_rerank_scores(data: object, *, expected: int) -> list[float] | None:
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None

    scores = [0.0] * expected
    seen = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or index < 0 or index >= expected:
            continue
        raw = item.get("relevance_score", item.get("score"))
        if raw is None:
            continue
        scores[index] = float(raw)
        seen += 1
    if seen == 0:
        return None
    return scores
