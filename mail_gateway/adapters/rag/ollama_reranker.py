"""Rerank adapters for RAG candidate reordering."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import replace

import httpx

from mail_gateway.domain.models import DocumentChunk
from mail_gateway.ports import Reranker

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_YES_NO_TOKEN_RE = re.compile(
    r"\b(yes|no|true|false|relevant|irrelevant|да|нет)\b",
    re.IGNORECASE,
)
_YES = frozenset({"yes", "true", "relevant", "да"})
_NO = frozenset({"no", "false", "irrelevant", "нет"})

_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_DEFAULT_INSTRUCT = (
    "Given a user question about 1C and company IT documentation, "
    "retrieve relevant passages that answer the query"
)
_QWEN_MODEL = "dengcao/Qwen3-Reranker-8B:Q8_0"


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
    """Score candidates with a Qwen3-Reranker via Ollama ``POST /api/chat``.

    Official scoring uses P(yes) vs P(no) at the next token. Ollama has no
    ``/api/rerank``; we send the Qwen3 Instruct/Query/Document chat turn and
    prefer logprobs when present, otherwise parse ``yes``/``no`` from content.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = _QWEN_MODEL,
        timeout_sec: float = 60.0,
        instruct: str = _DEFAULT_INSTRUCT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
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
        scores: list[float] = []
        parsed = 0
        with httpx.Client(timeout=self._timeout) as client:
            for document in documents:
                score = self._score_document(client, query, document)
                if score is None:
                    scores.append(0.0)
                    continue
                parsed += 1
                scores.append(score)
        if parsed == 0:
            return None
        return scores

    def _score_document(
        self,
        client: httpx.Client,
        query: str,
        document: str,
    ) -> float | None:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(self._instruct, query, document)},
            ],
            "stream": False,
            "logprobs": True,
            "top_logprobs": 8,
            "options": {
                "temperature": 0.0,
                "num_predict": 16,
            },
        }
        response = client.post(f"{self._base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return score_from_ollama_response(data)


def _user_content(instruct: str, query: str, document: str) -> str:
    return (
        f"<Instruct>: {instruct}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def score_from_ollama_response(data: object) -> float | None:
    """Prefer P(yes) from logprobs; otherwise parse yes/no (or a number) from text."""
    if not isinstance(data, dict):
        return None
    from_logprobs = score_from_logprobs(data)
    if from_logprobs is not None:
        return from_logprobs
    message = data.get("message")
    if isinstance(message, dict):
        from_logprobs = score_from_logprobs(message)
        if from_logprobs is not None:
            return from_logprobs
        content = str(message.get("content") or "")
        parsed = parse_relevance_score(content)
        if parsed is not None:
            return parsed
    return parse_relevance_score(str(data.get("response") or ""))


def score_from_logprobs(payload: object) -> float | None:
    """Softmax of yes vs no token logprobs at the first generated token."""
    if not isinstance(payload, dict):
        return None
    yes_lp: float | None = None
    no_lp: float | None = None
    for token, logprob in _iter_token_logprobs(payload):
        norm = _normalize_token(token)
        if norm in _YES:
            yes_lp = logprob if yes_lp is None else max(yes_lp, logprob)
        elif norm in _NO:
            no_lp = logprob if no_lp is None else max(no_lp, logprob)
        if yes_lp is not None and no_lp is not None:
            break
    if yes_lp is None and no_lp is None:
        return None
    if yes_lp is None:
        return 0.0
    if no_lp is None:
        return 1.0
    peak = max(yes_lp, no_lp)
    yes_p = math.exp(yes_lp - peak)
    no_p = math.exp(no_lp - peak)
    return yes_p / (yes_p + no_p)


def _iter_token_logprobs(payload: dict) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    raw = payload.get("logprobs")
    entries: list[object] = []
    if isinstance(raw, list):
        entries = list(raw)
    elif isinstance(raw, dict):
        content = raw.get("content", raw.get("tokens"))
        if isinstance(content, list):
            entries = list(content)
        else:
            entries = [raw]
    # First generated token is enough for Qwen3 yes/no.
    for entry in entries[:1]:
        _collect_logprob_pairs(entry, pairs)
        if pairs:
            break
    if not pairs and isinstance(raw, dict):
        _collect_logprob_pairs(raw, pairs)
    return pairs


def _collect_logprob_pairs(entry: object, pairs: list[tuple[str, float]]) -> None:
    if not isinstance(entry, dict):
        return
    token = entry.get("token", entry.get("text"))
    logprob = entry.get("logprob", entry.get("log_prob"))
    if isinstance(token, str) and isinstance(logprob, (int, float)):
        pairs.append((token, float(logprob)))
    top = entry.get("top_logprobs")
    if isinstance(top, list):
        for item in top:
            if not isinstance(item, dict):
                continue
            alt = item.get("token", item.get("text"))
            alt_lp = item.get("logprob", item.get("log_prob"))
            if isinstance(alt, str) and isinstance(alt_lp, (int, float)):
                pairs.append((alt, float(alt_lp)))
    elif isinstance(top, dict):
        for alt, alt_lp in top.items():
            if isinstance(alt, str) and isinstance(alt_lp, (int, float)):
                pairs.append((alt, float(alt_lp)))


def _normalize_token(token: str) -> str:
    cleaned = token.strip().lower()
    for prefix in ("▁", "Ġ"):
        cleaned = cleaned.removeprefix(prefix)
    return cleaned.strip(" .,;:\"'")


def parse_relevance_score(text: str) -> float | None:
    """Parse yes/no or a numeric score from model text."""
    cleaned = _THINK_RE.sub("", text).strip()
    if not cleaned:
        return None

    if cleaned[0] in "{[":
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            raw = data.get("score", data.get("relevance_score", data.get("relevance")))
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        elif isinstance(data, (int, float)):
            return float(data)

    matches = list(_YES_NO_TOKEN_RE.finditer(cleaned))
    if matches:
        token = matches[-1].group(1).lower()
        if token in _YES:
            return 1.0
        if token in _NO:
            return 0.0

    token = cleaned.split()[0].rstrip(",;:")
    try:
        return float(token)
    except ValueError:
        pass

    match = _SCORE_RE.search(cleaned)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None
