"""Infrastructure-free knowledge domain models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One searchable part of a versioned source document."""

    text: str
    source_path: str = ""
    filename: str = ""
    document_id: str = ""
    knowledge_id: str = "main"
    chunk_index: int = 0
    score: float | None = None
    headings: tuple[str, ...] = ()
    content_hash: str = ""

    @property
    def document_version(self) -> str:
        return self.content_hash

    def with_document(
        self,
        *,
        source_path: str,
        filename: str,
        document_id: str,
        knowledge_id: str,
        chunk_index: int,
        content_hash: str,
    ) -> DocumentChunk:
        return replace(
            self,
            source_path=source_path,
            filename=filename,
            document_id=document_id,
            knowledge_id=knowledge_id,
            chunk_index=chunk_index,
            content_hash=content_hash,
        )


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    knowledge_id: str
    filename: str
    content_hash: str
    status: str
    chunk_count: int
    source_updated_at: str | None = None
    last_seen_at: str | None = None
    missing_count: int = 0
