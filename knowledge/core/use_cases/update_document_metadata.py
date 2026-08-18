"""Update mutable source metadata while preserving document identity/version."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePath

from knowledge.core.domain import DocumentRecord
from knowledge.core.ports import DocumentRegistry, VectorIndex


class UpdateDocumentMetadata:
    def __init__(self, *, registry: DocumentRegistry, vector_index: VectorIndex) -> None:
        self._registry = registry
        self._vector_index = vector_index

    def execute(
        self,
        document_id: str,
        *,
        knowledge_id: str = "main",
        filename: str | None = None,
        source_path: str | None = None,
        status: str | None = None,
        source_updated_at: str | None = None,
        last_seen_at: str | None = None,
        missing_count: int | None = None,
    ) -> DocumentRecord:
        current = self._registry.get(document_id, knowledge_id)
        if current is None:
            raise KeyError(f"Unknown document: {document_id}")
        new_filename = PurePath(filename).name if filename is not None else current.filename
        if not new_filename:
            raise ValueError("filename must not be empty")
        if missing_count is not None and missing_count < 0:
            raise ValueError("missing_count must be non-negative")

        if filename is not None or source_path is not None:
            self._vector_index.update_document_metadata(
                document_id,
                knowledge_id,
                filename=new_filename,
                source_path=source_path,
            )
        updated = replace(
            current,
            filename=new_filename,
            status=status if status is not None else current.status,
            source_updated_at=(
                source_updated_at
                if source_updated_at is not None
                else current.source_updated_at
            ),
            last_seen_at=last_seen_at if last_seen_at is not None else current.last_seen_at,
            missing_count=missing_count if missing_count is not None else current.missing_count,
        )
        self._registry.save(updated)
        return updated

    __call__ = execute
