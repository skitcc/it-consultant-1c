"""Rerank adapters for RAG candidate reordering."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx

from common.timing import record
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

_SYSTEM_PROMPT = """You are a strict document relevance evaluator.

Evaluate only whether the Document contains information that can answer the Query.
Match the requested entity, product, operation, version, date or period, metric,
units, and conditions exactly. Do not infer missing facts and do not reward a
document merely for sharing the same topic.

Use this scale:
- 0.90-1.00: complete, direct answer with all material constraints matching;
- 0.70-0.89: substantially answers the query, with only minor details missing;
- 0.40-0.69: useful partial answer, but important requested information is missing;
- 0.10-0.39: weak topical relation, wrong period/entity/metric, or no requested value;
- 0.00-0.09: irrelevant or contradictory.

The final answer must be exactly one number from 0.00 to 1.00. Do not put an
explanation, label, percent sign, or any other text in the final answer."""

_DEFAULT_INSTRUCT = (
    "Given a user question about 1C and company IT documentation, "
    "score how completely and precisely this passage can answer the query"
)
_QWEN_MODEL = "dengcao/Qwen3-Reranker-8B:Q8_0"
_SCORE_FORMAT = {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
}


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
    """Score candidates with a Qwen3-Reranker via Ollama ``POST /api/chat``."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = _QWEN_MODEL,
        timeout_sec: float = 60.0,
        num_predict: int = 16, 
        instruct: str = _DEFAULT_INSTRUCT,
        max_parallel_workers: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._num_predict = num_predict
        self._instruct = instruct
        self._max_workers = max_parallel_workers
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
            "Rerank started model=%s candidates=%s query_chars=%s workers=%s",
            self._model,
            len(items),
            len(query.strip()),
            self._max_workers,
        )
        try:
            scores = self._score_documents(query, items)
        except Exception:
            logger.exception(
                "Rerank failed model=%s elapsed=%.3fs; falling back to vector scores",
                self._model,
                time.perf_counter() - started_at,
            )
            return self._fallback.rerank(query, items)

        if scores is None or len(scores) != len(items):
            logger.warning(
                "Rerank returned unexpected scores model=%s elapsed=%.3fs; using vector scores",
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
            "Rerank completed model=%s candidates=%s elapsed=%.3fs top_score=%.4f",
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
        """Score candidates concurrently using ThreadPoolExecutor."""
        indexed_chunks = list(enumerate(chunks, start=1))
        workers = min(self._max_workers, len(chunks))

        def _evaluate_single(item: tuple[int, DocumentChunk]) -> tuple[int, float | None]:
            pos, chunk = item
            started = time.perf_counter()
            score = None
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    # 👈 3. Сразу запрашиваем быстрый скоринг БЕЗ thinking
                    score = self._score_document(client, query, chunk.text)
            except Exception:
                score = None
            finally:
                elapsed = time.perf_counter() - started
                record(f"rerank_{pos}/{len(chunks)}", elapsed)
                logger.info(
                    "Rerank candidate done candidate=%s/%s source=%r "
                    "chunk_index=%s parsed=%s rerank_score=%s elapsed=%.3fs",
                    pos,
                    len(chunks),
                    chunk.source_path,
                    chunk.chunk_index,
                    score is not None,
                    _format_score(score),
                    elapsed,
                )
            return pos, score

        # Запускаем параллельно в workers потоков
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_evaluate_single, indexed_chunks))

        scores_map = {pos: (score if score is not None else 0.0) for pos, score in results}
        parsed_count = sum(1 for _, s in results if s is not None)

        if parsed_count == 0:
            return None
        return [scores_map[i] for i in range(1, len(chunks) + 1)]

    def _score_document(
        self,
        client: httpx.Client,
        query: str,
        document: str,
    ) -> float | None:
        score = self._request_score(client, query, document, disable_thinking=True)
        if score is not None:
            return score

        logger.warning(
            "Reranker returned no valid score model=%s; retrying with thinking enabled",
            self._model,
        )
        return self._request_score(client, query, document, disable_thinking=False)

    def _request_score(
        self,
        client: httpx.Client,
        query: str,
        document: str,
        *,
        disable_thinking: bool,
    ) -> float | None:
        started_at = time.perf_counter()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(self._instruct, query, document)},
            ],
            "stream": False,
            "logprobs": True,
            "top_logprobs": 8,
            "format": _SCORE_FORMAT,
            "think": not disable_thinking, 
            "keep_alive": -1,
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 16 if disable_thinking else 256,
            },
        }
        if disable_thinking:
            payload["think"] = False

        response = client.post(f"{self._base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        score = score_from_ollama_response(data)
        
        logger.debug(
            "Rerank Ollama response model=%s thinking_disabled=%s "
            "elapsed=%.3fs done_reason=%s prompt_tokens=%s generated_tokens=%s score=%s",
            self._model,
            disable_thinking,
            time.perf_counter() - started_at,
            data.get("done_reason") if isinstance(data, dict) else None,
            data.get("prompt_eval_count") if isinstance(data, dict) else None,
            data.get("eval_count") if isinstance(data, dict) else None,
            _format_score(score),
        )
        return score


def _user_content(instruct: str, query: str, document: str) -> str:
    return (
        f"<Instruct>: {instruct}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def score_from_ollama_response(data: object) -> float | None:
    if not isinstance(data, dict):
        return None
    from_logprobs = score_from_logprobs(data)
    if from_logprobs is not None:
        return _valid_score(from_logprobs)
    message = data.get("message")
    if isinstance(message, dict):
        from_logprobs = score_from_logprobs(message)
        if from_logprobs is not None:
            return _valid_score(from_logprobs)
        content = str(message.get("content") or "")
        parsed = parse_relevance_score(content)
        if parsed is not None:
            return parsed
    return parse_relevance_score(str(data.get("response") or ""))


def _response_content(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    message = data.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(data.get("response") or "")


def _format_score(score: float | None) -> str:
    return "none" if score is None else f"{score:.4f}"


def score_from_logprobs(payload: object) -> float | None:
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
                    return _valid_score(float(raw))
                except (TypeError, ValueError):
                    pass
        elif isinstance(data, (int, float)):
            return _valid_score(float(data))

    matches = list(_YES_NO_TOKEN_RE.finditer(cleaned))
    if matches:
        token = matches[-1].group(1).lower()
        if token in _YES:
            return 1.0
        if token in _NO:
            return 0.0

    token = cleaned.split()[0].rstrip(",;:")
    try:
        return _valid_score(float(token))
    except ValueError:
        pass

    match = _SCORE_RE.search(cleaned)
    if match is None:
        return None
    try:
        return _valid_score(float(match.group(0)))
    except ValueError:
        return None


def _valid_score(value: float) -> float | None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value