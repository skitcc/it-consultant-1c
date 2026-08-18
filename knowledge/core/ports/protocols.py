"""Protocols implemented by outbound knowledge adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from knowledge.core.domain import ConversationMessage, DocumentChunk, DocumentRecord


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, raw_bytes: bytes, filename: str) -> Sequence[DocumentChunk]:
        """Parse the exact original bytes of one named document."""
        ...


@runtime_checkable
class DocumentRegistry(Protocol):
    def get(self, document_id: str, knowledge_id: str = "main") -> DocumentRecord | None:
        ...

    def list(self, knowledge_id: str = "main") -> Sequence[DocumentRecord]:
        ...

    def save(self, record: DocumentRecord) -> None:
        ...

    def delete(self, document_id: str, knowledge_id: str = "main") -> None:
        ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class VectorIndex(Protocol):
    def replace_document(
        self,
        *,
        document_id: str,
        knowledge_id: str,
        content_hash: str,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Commit a new version before removing older versions."""
        ...

    def remove_document(self, document_id: str, knowledge_id: str = "main") -> None:
        ...

    def update_document_metadata(
        self,
        document_id: str,
        knowledge_id: str = "main",
        *,
        filename: str,
        source_path: str | None = None,
    ) -> None:
        ...

    def search(
        self,
        vector: Sequence[float],
        *,
        knowledge_id: str = "main",
        limit: int = 20,
        score_threshold: float | None = None,
    ) -> Sequence[DocumentChunk]:
        ...

    def load_neighbors(
        self,
        seeds: Sequence[DocumentChunk],
        *,
        knowledge_id: str = "main",
        window: int = 1,
    ) -> Sequence[DocumentChunk]:
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> Sequence[DocumentChunk]:
        ...


@runtime_checkable
class ChatModel(Protocol):
    def complete(self, messages: Sequence[ConversationMessage]) -> str:
        ...
