"""Qwen3 reranking through Ollama, including reusable scoring helpers."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import replace

import httpx

from knowledge.core.domain import DocumentChunk

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
    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> list[DocumentChunk]:
        del query
        return sorted(
            chunks,
            key=lambda chunk: chunk.score if chunk.score is not None else float("-inf"),
            reverse=True,
        )


class OllamaQwen3Reranker:
    """Prefer Qwen yes/no logprobs and fall back to vector scores."""

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
        try:
            scores = self._score_documents(query, [chunk.text for chunk in items])
        except Exception:
            logger.exception("Rerank failed; using vector scores")
            return self._fallback.rerank(query, items)
        if scores is None or len(scores) != len(items):
            return self._fallback.rerank(query, items)
        return [
            replace(chunk, score=score)
            for chunk, score in sorted(
                zip(items, scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]

    def _score_documents(self, query: str, documents: list[str]) -> list[float] | None:
        scores: list[float] = []
        parsed = 0
        with httpx.Client(timeout=self._timeout) as client:
            for document in documents:
                response = client.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": user_content(self._instruct, query, document),
                            },
                        ],
                        "stream": False,
                        "logprobs": True,
                        "top_logprobs": 8,
                        "options": {"temperature": 0.0, "num_predict": 16},
                    },
                )
                response.raise_for_status()
                score = score_from_ollama_response(response.json())
                if score is None:
                    scores.append(0.0)
                else:
                    parsed += 1
                    scores.append(score)
        return scores if parsed else None


# Compatibility name for callers migrating from the old adapter.
OllamaReranker = OllamaQwen3Reranker


def user_content(instruct: str, query: str, document: str) -> str:
    return (
        f"<Instruct>: {instruct}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def score_from_ollama_response(data: object) -> float | None:
    if not isinstance(data, dict):
        return None
    score = score_from_logprobs(data)
    if score is not None:
        return score
    message = data.get("message")
    if isinstance(message, dict):
        score = score_from_logprobs(message)
        if score is not None:
            return score
        parsed = parse_relevance_score(str(message.get("content") or ""))
        if parsed is not None:
            return parsed
    return parse_relevance_score(str(data.get("response") or ""))


def score_from_logprobs(payload: object) -> float | None:
    """Return softmax P(yes) from first-token yes/no log probabilities."""
    if not isinstance(payload, dict):
        return None
    yes_lp: float | None = None
    no_lp: float | None = None
    for token, logprob in iter_token_logprobs(payload):
        normalized = normalize_token(token)
        if normalized in _YES:
            yes_lp = logprob if yes_lp is None else max(yes_lp, logprob)
        elif normalized in _NO:
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
    yes_probability = math.exp(yes_lp - peak)
    no_probability = math.exp(no_lp - peak)
    return yes_probability / (yes_probability + no_probability)


def iter_token_logprobs(payload: dict) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    raw = payload.get("logprobs")
    entries: list[object] = []
    if isinstance(raw, list):
        entries = list(raw)
    elif isinstance(raw, dict):
        content = raw.get("content", raw.get("tokens"))
        entries = list(content) if isinstance(content, list) else [raw]
    for entry in entries[:1]:
        collect_logprob_pairs(entry, pairs)
        if pairs:
            break
    if not pairs and isinstance(raw, dict):
        collect_logprob_pairs(raw, pairs)
    return pairs


def collect_logprob_pairs(entry: object, pairs: list[tuple[str, float]]) -> None:
    if not isinstance(entry, dict):
        return
    token = entry.get("token", entry.get("text"))
    logprob = entry.get("logprob", entry.get("log_prob"))
    if isinstance(token, str) and isinstance(logprob, (int, float)):
        pairs.append((token, float(logprob)))
    top = entry.get("top_logprobs")
    if isinstance(top, list):
        for item in top:
            collect_logprob_pairs(item, pairs)
    elif isinstance(top, dict):
        for alternate, alternate_lp in top.items():
            if isinstance(alternate, str) and isinstance(alternate_lp, (int, float)):
                pairs.append((alternate, float(alternate_lp)))


def normalize_token(token: str) -> str:
    cleaned = token.strip().lower()
    for prefix in ("▁", "Ġ"):
        cleaned = cleaned.removeprefix(prefix)
    return cleaned.strip(" .,;:\"'")


def parse_relevance_score(text: str) -> float | None:
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
            try:
                return float(raw) if raw is not None else None
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
    try:
        return float(cleaned.split()[0].rstrip(",;:"))
    except ValueError:
        match = _SCORE_RE.search(cleaned)
        return float(match.group(0)) if match is not None else None
