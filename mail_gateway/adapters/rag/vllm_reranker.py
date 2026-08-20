"""Batch rerank via vLLM ``/v1/rerank``."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import replace

import httpx

from common.timing import record
from mail_gateway.adapters.rag.ollama_reranker import (
    _DEFAULT_INSTRUCT,
    ScorePassthroughReranker,
)
from mail_gateway.domain.models import DocumentChunk

logger = logging.getLogger(__name__)


class VllmReranker:
    """Score all candidates in one vLLM rerank request."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_sec: float = 60.0,
        instruct: str = _DEFAULT_INSTRUCT,
    ) -> None:
        self._base_url = _openai_base_url(base_url)
        self._model = model
        self._timeout = timeout_sec
        self._instruct = instruct
        self._fallback = ScorePassthroughReranker()

    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> list[DocumentChunk]:
        items = list(chunks)
        if len(items) <= 1:
            logger.debug(
                "Rerank skipped model=%s candidates=%s reason=not_enough_candidates",
                self._model,
                len(items),
            )
            return items

        started_at = time.perf_counter()
        logger.info(
            "Rerank started backend=vllm model=%s candidates=%s query_chars=%s",
            self._model,
            len(items),
            len(query.strip()),
        )
        try:
            scores = self._score_documents(query, items)
        except Exception:
            logger.exception(
                "Rerank failed model=%s elapsed=%.3fs; "
                "falling back to vector scores",
                self._model,
                time.perf_counter() - started_at,
            )
            return self._fallback.rerank(query, items)

        if scores is None or len(scores) != len(items):
            logger.warning(
                "Rerank returned unexpected scores model=%s elapsed=%.3fs; "
                "using vector scores",
                self._model,
                time.perf_counter() - started_at,
            )
            return self._fallback.rerank(query, items)

        scored = sorted(
            zip(items, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        ranked = [replace(chunk, score=score) for chunk, score in scored]
        for rank, (chunk, score) in enumerate(scored, start=1):
            logger.debug(
                "Rerank ranking rank=%s source=%r chunk_index=%s "
                "vector_score=%s rerank_score=%.4f headings=%r",
                rank,
                chunk.source_path,
                chunk.chunk_index,
                _format_score(chunk.score),
                score,
                chunk.headings,
            )
        logger.info(
            "Rerank completed backend=vllm model=%s candidates=%s "
            "http_calls=1 elapsed=%.3fs top_score=%.4f",
            self._model,
            len(ranked),
            time.perf_counter() - started_at,
            ranked[0].score if ranked and ranked[0].score is not None else -1.0,
        )
        return ranked

    def _score_documents(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> list[float] | None:
        documents = [chunk.text for chunk in chunks]
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "instruction": self._instruct,
        }
        request_started = time.perf_counter()
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._base_url}/rerank", json=payload)
            response.raise_for_status()
            data = response.json()
        record("rerank_batch", time.perf_counter() - request_started)
        return scores_from_vllm_rerank(data, expected=len(documents))


def scores_from_vllm_rerank(data: object, *, expected: int) -> list[float] | None:
    """Map vLLM/Cohere rerank payload onto the original document order."""
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None
    scores = [0.0] * expected
    seen = 0
    for item in results:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < expected:
            return None
        raw = item.get("relevance_score", item.get("score"))
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        scores[index] = score
        seen += 1
    if seen == 0:
        return None
    return scores


def _openai_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _format_score(score: float | None) -> str:
    return "none" if score is None else f"{score:.4f}"
