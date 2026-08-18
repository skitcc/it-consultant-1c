"""Versioned, document-scoped Qdrant vector index."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from knowledge.core.domain import DocumentChunk

_POINT_NAMESPACE = uuid.UUID("36ce2f69-8f88-4e8a-b569-a415c86c83da")


class QdrantVectorIndex:
    def __init__(
        self,
        *,
        url: str,
        collection: str,
        client: Any | None = None,
    ) -> None:
        self._client = client or QdrantClient(url=url, check_compatibility=False)
        self._collection = collection

    def replace_document(
        self,
        *,
        document_id: str,
        knowledge_id: str,
        content_hash: str,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if not chunks:
            raise ValueError("Cannot index a document without chunks")
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must have equal lengths")
        vector_size = _validate_vector_dimensions(vectors)
        self._ensure_collection(vector_size)
        points = [
            qmodels.PointStruct(
                id=str(_point_id(document_id, content_hash, index)),
                vector=list(vector),
                payload=_payload_from_chunk(
                    chunk,
                    document_id=document_id,
                    knowledge_id=knowledge_id,
                    content_hash=content_hash,
                    chunk_index=index,
                ),
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]

        # This ordering is the central non-destructive guarantee: a failed
        # parse/embed never reaches here, and old points are deleted only after
        # Qdrant confirms the complete new-version upsert.
        self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=_document_conditions(document_id, knowledge_id),
                        must_not=[
                            qmodels.FieldCondition(
                                key="document_version",
                                match=qmodels.MatchValue(value=content_hash),
                            )
                        ],
                    )
                ),
                wait=True,
            )
        except Exception:
            # Registry still points to the previous version. Best-effort rollback
            # prevents the uncommitted new points from participating in search.
            try:
                conditions = _document_conditions(document_id, knowledge_id)
                conditions.append(
                    qmodels.FieldCondition(
                        key="document_version",
                        match=qmodels.MatchValue(value=content_hash),
                    )
                )
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(must=conditions),
                    ),
                    wait=True,
                )
            except Exception:
                pass
            raise

    def remove_document(self, document_id: str, knowledge_id: str = "main") -> None:
        if not self._client.collection_exists(self._collection):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=_document_conditions(document_id, knowledge_id),
                )
            ),
            wait=True,
        )

    def update_document_metadata(
        self,
        document_id: str,
        knowledge_id: str = "main",
        *,
        filename: str,
        source_path: str | None = None,
    ) -> None:
        if not self._client.collection_exists(self._collection):
            return
        payload = {"filename": filename}
        if source_path is not None:
            payload["source_path"] = source_path
        self._client.set_payload(
            collection_name=self._collection,
            payload=payload,
            points=qmodels.Filter(
                must=_document_conditions(document_id, knowledge_id),
            ),
            wait=True,
        )

    def search(
        self,
        vector: Sequence[float],
        *,
        knowledge_id: str = "main",
        limit: int = 20,
        score_threshold: float | None = None,
    ) -> list[DocumentChunk]:
        if not self._client.collection_exists(self._collection):
            return []
        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "query": list(vector),
            "query_filter": qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="knowledge_id",
                        match=qmodels.MatchValue(value=knowledge_id),
                    )
                ]
            ),
            "limit": limit,
            "with_payload": True,
        }
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        response = self._client.query_points(**kwargs)
        hits = getattr(response, "points", None) or []
        result: list[DocumentChunk] = []
        for hit in hits:
            chunk = chunk_from_payload(
                getattr(hit, "payload", None),
                score=getattr(hit, "score", None),
            )
            if chunk is not None:
                result.append(chunk)
        return result

    def load_neighbors(
        self,
        seeds: Sequence[DocumentChunk],
        *,
        knowledge_id: str = "main",
        window: int = 1,
    ) -> list[DocumentChunk]:
        if not seeds or window < 0 or not self._client.collection_exists(self._collection):
            return []
        found: dict[tuple[str, str, int], DocumentChunk] = {}
        for seed in seeds:
            version = seed.content_hash
            conditions = _document_conditions(seed.document_id, knowledge_id)
            conditions.append(
                qmodels.FieldCondition(
                    key="document_version",
                    match=qmodels.MatchValue(value=version),
                )
            )
            if seed.headings:
                document_chunks = self._scroll_chunks(conditions, all_pages=True)
                siblings = sorted(
                    (
                        chunk
                        for chunk in document_chunks
                        if chunk.headings == seed.headings
                    ),
                    key=lambda chunk: chunk.chunk_index,
                )
                indexes = [chunk.chunk_index for chunk in siblings]
                try:
                    position = indexes.index(seed.chunk_index)
                except ValueError:
                    selected = [seed]
                else:
                    selected = siblings[
                        max(0, position - window) : position + window + 1
                    ]
            else:
                indexes = {
                    index
                    for index in range(
                        seed.chunk_index - window,
                        seed.chunk_index + window + 1,
                    )
                    if index >= 0
                }
                numeric_conditions = [
                    *conditions,
                    qmodels.FieldCondition(
                        key="chunk_index",
                        match=qmodels.MatchAny(any=sorted(indexes)),
                    ),
                ]
                selected = self._scroll_chunks(numeric_conditions, all_pages=False)
            if not selected:
                selected = [seed]
            for chunk in selected:
                found[(chunk.document_id, chunk.content_hash, chunk.chunk_index)] = chunk
        return sorted(
            found.values(),
            key=lambda chunk: (chunk.source_path, chunk.chunk_index),
        )

    def _scroll_chunks(
        self,
        conditions: list[Any],
        *,
        all_pages: bool,
    ) -> list[DocumentChunk]:
        result: list[DocumentChunk] = []
        offset: object | None = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qmodels.Filter(must=conditions),
                with_payload=True,
                with_vectors=False,
                limit=256 if all_pages else 32,
                offset=offset,
            )
            for record in records:
                chunk = chunk_from_payload(
                    getattr(record, "payload", None),
                    score=None,
                )
                if chunk is not None:
                    result.append(chunk)
            if not all_pages or offset is None:
                return result

    def _ensure_collection(self, vector_size: int) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        fields = {
            "document_id": qmodels.PayloadSchemaType.KEYWORD,
            "knowledge_id": qmodels.PayloadSchemaType.KEYWORD,
            "filename": qmodels.PayloadSchemaType.KEYWORD,
            "source_path": qmodels.PayloadSchemaType.KEYWORD,
            "file_hash": qmodels.PayloadSchemaType.KEYWORD,
            "document_version": qmodels.PayloadSchemaType.KEYWORD,
            "headings": qmodels.PayloadSchemaType.KEYWORD,
            "chunk_index": qmodels.PayloadSchemaType.INTEGER,
        }
        for field_name, schema in fields.items():
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:
                # Existing indexes and servers without payload-index support are
                # both safe; filtering still remains correct.
                continue


def _payload_from_chunk(
    chunk: DocumentChunk,
    *,
    document_id: str,
    knowledge_id: str,
    content_hash: str,
    chunk_index: int,
) -> dict[str, object]:
    source_path = chunk.source_path or chunk.filename
    filename = chunk.filename or source_path.rsplit("/", 1)[-1]
    return {
        "document_id": document_id,
        "knowledge_id": knowledge_id,
        "filename": filename,
        "source_path": source_path,
        "chunk_index": chunk_index,
        "text": chunk.text,
        "file_hash": content_hash,
        "document_version": content_hash,
        "headings": list(chunk.headings),
    }


def chunk_from_payload(payload: object, *, score: object) -> DocumentChunk | None:
    if not isinstance(payload, dict):
        return None
    text = str(payload.get("text") or "").strip()
    if not text:
        return None
    raw_headings = payload.get("headings")
    if isinstance(raw_headings, (list, tuple)):
        headings = tuple(str(value) for value in raw_headings if value)
    elif isinstance(raw_headings, str) and raw_headings.strip():
        headings = (raw_headings.strip(),)
    else:
        headings = ()
    raw_score = float(score) if isinstance(score, (int, float)) else None
    return DocumentChunk(
        text=text,
        source_path=str(payload.get("source_path") or ""),
        filename=str(payload.get("filename") or ""),
        document_id=str(payload.get("document_id") or ""),
        knowledge_id=str(payload.get("knowledge_id") or "main"),
        chunk_index=int(payload.get("chunk_index") or 0),
        score=raw_score,
        headings=headings,
        content_hash=str(
            payload.get("document_version") or payload.get("file_hash") or ""
        ),
    )


def _document_conditions(document_id: str, knowledge_id: str) -> list[Any]:
    return [
        qmodels.FieldCondition(
            key="document_id",
            match=qmodels.MatchValue(value=document_id),
        ),
        qmodels.FieldCondition(
            key="knowledge_id",
            match=qmodels.MatchValue(value=knowledge_id),
        ),
    ]


def _point_id(document_id: str, content_hash: str, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(
        _POINT_NAMESPACE,
        f"{document_id}:{content_hash}:{chunk_index}",
    )


def _validate_vector_dimensions(vectors: Sequence[Sequence[float]]) -> int:
    if not vectors or not vectors[0]:
        raise ValueError("Vectors must not be empty")
    size = len(vectors[0])
    if any(len(vector) != size for vector in vectors):
        raise ValueError("Vectors must have consistent dimensions")
    return size
