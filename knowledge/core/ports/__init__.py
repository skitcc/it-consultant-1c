"""Dependency-inversion ports for knowledge use cases."""

from knowledge.core.ports.protocols import (
    ChatModel,
    DocumentParser,
    DocumentRegistry,
    Embedder,
    Reranker,
    VectorIndex,
)

__all__ = [
    "ChatModel",
    "DocumentParser",
    "DocumentRegistry",
    "Embedder",
    "Reranker",
    "VectorIndex",
]
