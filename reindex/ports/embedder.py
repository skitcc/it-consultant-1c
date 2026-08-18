"""Port: dense embeddings for document chunks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turn text into a dense vector (and batches of texts into vectors)."""

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for ``text``."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per item in ``texts``."""
        ...
