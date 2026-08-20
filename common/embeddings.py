"""Embedding clients shared by reindex and mail_gateway."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)

_KEEP_ALIVE = -1
_NUM_CTX = 2048


@runtime_checkable
class Embedder(Protocol):
    """Turn text into a dense vector (and batches of texts into vectors)."""

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for ``text``."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per item in ``texts``."""
        ...


class OllamaEmbedder:
    """Calls Ollama ``/api/embed`` and returns a dense vector."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        timeout_sec: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if not cleaned:
            return []
        if any(not item for item in cleaned):
            raise ValueError("Cannot embed empty text")

        url = f"{self._base_url}/api/embed"
        logger.debug(
            "Ollama embed start model=%s batch=%s url=%s",
            self._model,
            len(cleaned),
            url,
        )
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                url,
                json={
                    "model": self._model,
                    "input": cleaned,
                    "keep_alive": _KEEP_ALIVE,
                    "options": {"num_ctx": _NUM_CTX},
                },
            )
            response.raise_for_status()
            data = response.json()

        vectors = _vectors_from_ollama(data, expected=len(cleaned))
        if vectors is None:
            keys = list(data) if isinstance(data, dict) else type(data).__name__
            raise RuntimeError(
                f"Ollama returned unexpected embeddings; "
                f"expected={len(cleaned)} keys={keys}"
            )
        logger.debug(
            "Ollama embed done model=%s batch=%s dim=%s",
            self._model,
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )
        return vectors


def _vectors_from_ollama(data: object, *, expected: int) -> list[list[float]] | None:
    if not isinstance(data, dict):
        return None
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        vectors: list[list[float]] = []
        for item in embeddings:
            parsed = _as_vector(item)
            if parsed is None:
                return None
            vectors.append(parsed)
        if len(vectors) == expected:
            return vectors
    if expected == 1:
        single = _vector_from_ollama(data)
        if single is not None:
            return [single]
    return None


def _vector_from_ollama(data: object) -> list[float] | None:
    """Accept both ``embedding`` (legacy) and ``embeddings`` (``/api/embed``)."""
    if not isinstance(data, dict):
        return None
    embedding = data.get("embedding")
    parsed = _as_vector(embedding)
    if parsed is not None:
        return parsed
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return _as_vector(embeddings[0])
    return None


def _as_vector(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    if isinstance(value[0], list):
        return _as_vector(value[0])
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


class OpenAIEmbedder:
    """Calls OpenAI-compatible ``/v1/embeddings`` (vLLM)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_sec: float = 120.0,
    ) -> None:
        self._base_url = _openai_base_url(base_url)
        self._model = model
        self._timeout = timeout_sec

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if not cleaned:
            return []
        if any(not item for item in cleaned):
            raise ValueError("Cannot embed empty text")

        url = f"{self._base_url}/embeddings"
        logger.debug(
            "OpenAI embed start model=%s batch=%s url=%s",
            self._model,
            len(cleaned),
            url,
        )
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                url,
                json={"model": self._model, "input": cleaned},
            )
            response.raise_for_status()
            data = response.json()

        vectors = _vectors_from_openai(data, expected=len(cleaned))
        if vectors is None:
            keys = list(data) if isinstance(data, dict) else type(data).__name__
            raise RuntimeError(
                f"Embeddings API returned unexpected payload; "
                f"expected={len(cleaned)} keys={keys}"
            )
        logger.debug(
            "OpenAI embed done model=%s batch=%s dim=%s",
            self._model,
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )
        return vectors


def _openai_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _vectors_from_openai(data: object, *, expected: int) -> list[list[float]] | None:
    if not isinstance(data, dict):
        return None
    raw = data.get("data")
    if not isinstance(raw, list) or len(raw) != expected:
        return None
    indexed: list[tuple[int, list[float]]] = []
    for offset, item in enumerate(raw):
        if not isinstance(item, dict):
            return None
        vector = _as_vector(item.get("embedding"))
        if vector is None:
            return None
        index = item.get("index", offset)
        if not isinstance(index, int):
            return None
        indexed.append((index, vector))
    indexed.sort(key=lambda pair: pair[0])
    return [vector for _index, vector in indexed]
