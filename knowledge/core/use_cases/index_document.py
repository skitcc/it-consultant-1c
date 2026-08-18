"""Atomically index one version of one document."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import PurePath

from knowledge.core.domain import DocumentChunk, DocumentRecord
from knowledge.core.ports import DocumentParser, DocumentRegistry, Embedder, VectorIndex

_DOCUMENT_NAMESPACE = uuid.UUID("9d4b4754-b33f-4ccb-b8e7-ec3453f52bb7")


@dataclass(frozen=True, slots=True)
class IndexResult:
    document_id: str
    content_hash: str
    chunk_count: int
    status: str


class IndexDocument:
    """Parse, embed, and version-replace exactly one supplied byte stream."""

    def __init__(
        self,
        *,
        parser: DocumentParser,
        registry: DocumentRegistry,
        embedder: Embedder,
        vector_index: VectorIndex,
    ) -> None:
        self._parser = parser
        self._registry = registry
        self._embedder = embedder
        self._vector_index = vector_index

    def execute(
        self,
        raw_bytes: bytes,
        filename: str,
        *,
        document_id: str | None = None,
        knowledge_id: str = "main",
        source_path: str | None = None,
        source_updated_at: str | None = None,
        last_seen_at: str | None = None,
    ) -> IndexResult:
        if not isinstance(raw_bytes, bytes):
            raise TypeError("raw_bytes must be bytes")
        clean_filename = PurePath(filename).name
        if not clean_filename:
            raise ValueError("filename must not be empty")
        clean_knowledge_id = knowledge_id.strip()
        if not clean_knowledge_id:
            raise ValueError("knowledge_id must not be empty")

        resolved_id = document_id or stable_document_id(clean_knowledge_id, filename)
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        previous = self._registry.get(resolved_id, clean_knowledge_id)
        if previous is not None and previous.content_hash == content_hash:
            return IndexResult(
                document_id=resolved_id,
                content_hash=content_hash,
                chunk_count=previous.chunk_count,
                status="unchanged",
            )

        parsed = list(self._parser.parse(raw_bytes, clean_filename))
        chunks = _prepare_chunks(
            parsed,
            source_path=source_path or filename,
            filename=clean_filename,
            document_id=resolved_id,
            knowledge_id=clean_knowledge_id,
            content_hash=content_hash,
        )
        if not chunks:
            raise ValueError("Document parser returned no non-empty chunks")

        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        _validate_vectors(vectors, len(chunks))

        # The vector adapter must first commit the new version, then remove old
        # points. Registry state is intentionally written only after that succeeds.
        self._vector_index.replace_document(
            document_id=resolved_id,
            knowledge_id=clean_knowledge_id,
            content_hash=content_hash,
            chunks=chunks,
            vectors=vectors,
        )
        record = DocumentRecord(
            document_id=resolved_id,
            knowledge_id=clean_knowledge_id,
            filename=clean_filename,
            content_hash=content_hash,
            status="indexed",
            chunk_count=len(chunks),
            source_updated_at=source_updated_at,
            last_seen_at=last_seen_at,
            missing_count=0,
        )
        self._registry.save(record)
        return IndexResult(
            document_id=resolved_id,
            content_hash=content_hash,
            chunk_count=len(chunks),
            status="indexed",
        )

    __call__ = execute


def stable_document_id(knowledge_id: str, source_path: str) -> str:
    normalized = source_path.replace("\\", "/").strip("/")
    return str(uuid.uuid5(_DOCUMENT_NAMESPACE, f"{knowledge_id}:{normalized}"))


def _prepare_chunks(
    chunks: list[DocumentChunk],
    *,
    source_path: str,
    filename: str,
    document_id: str,
    knowledge_id: str,
    content_hash: str,
) -> list[DocumentChunk]:
    prepared: list[DocumentChunk] = []
    for chunk in chunks:
        if not isinstance(chunk, DocumentChunk):
            raise TypeError("Document parser must return DocumentChunk values")
        text = chunk.text.strip()
        if not text:
            continue
        clean = DocumentChunk(text=text, headings=tuple(chunk.headings)).with_document(
            source_path=source_path,
            filename=filename,
            document_id=document_id,
            knowledge_id=knowledge_id,
            chunk_index=len(prepared),
            content_hash=content_hash,
        )
        prepared.append(clean)
    return prepared


def _validate_vectors(vectors: object, expected_count: int) -> None:
    if not isinstance(vectors, (list, tuple)) or len(vectors) != expected_count:
        raise ValueError("Embedder returned an unexpected number of vectors")
    size: int | None = None
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("Embedder returned an empty vector")
        if size is None:
            size = len(vector)
        elif len(vector) != size:
            raise ValueError("Embedder returned vectors with inconsistent dimensions")
