"""Index watched documents into Qdrant via embeddings."""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from reindex.adapters.document_readers import (
    build_default_document_reader,
    is_useful_chunk_text,
)
from reindex.domain.changes import FsChange
from reindex.domain.documents import iter_document_files
from reindex.ports import DocumentReader, Embedder

logger = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_SCROLL_LIMIT = 100
_HASH_CHUNK_SIZE = 1024 * 1024
_INDEX_VERSION = "table-aware-v2"


class QdrantIndexer:
    """Index documents into Qdrant with content-hash skip and duplicate dedup."""

    def __init__(
        self,
        *,
        qdrant_url: str,
        collection: str,
        embedder: Embedder,
        document_reader: DocumentReader | None = None,
        allowed_extensions: frozenset[str] | set[str] | None = None,
        max_tokens: int = 512,
    ) -> None:
        self._client = QdrantClient(url=qdrant_url, check_compatibility=False)
        self._collection = collection
        self._embedder = embedder
        self._document_reader = document_reader or build_default_document_reader(
            max_tokens=max_tokens,
        )
        self._allowed_extensions = frozenset(allowed_extensions or ())

    def reindex(self, watch_path: str) -> None:
        root = Path(watch_path)
        files = iter_document_files(
            root,
            allowed_extensions=self._allowed_extensions or None,
        )
        logger.info(
            "Qdrant reindex start path=%s files=%s collection=%s extensions=%s",
            watch_path,
            len(files),
            self._collection,
            sorted(self._allowed_extensions) if self._allowed_extensions else ["*"],
        )

        disk_hashes = self._disk_file_hashes(watch_path)
        if not disk_hashes:
            indexed_paths = self._scroll_indexed_paths()
            removed = 0
            for relative in indexed_paths:
                self._delete_source(relative)
                removed += 1
            logger.info(
                "Qdrant reindex done collection=%s empty watch_path removed=%s",
                self._collection,
                removed,
            )
            return

        canonical = _canonical_paths(disk_hashes)
        indexed_paths = self._scroll_indexed_paths()

        skipped = upserted = dup_removed = stale_removed = 0
        for relative, content_hash in disk_hashes.items():
            if relative != canonical[content_hash]:
                if relative in indexed_paths:
                    self._delete_source(relative)
                    dup_removed += 1
                continue
            if indexed_paths.get(relative) == content_hash:
                skipped += 1
                continue
            self._upsert_file(watch_path, relative, force=True)
            upserted += 1

        for relative in indexed_paths:
            if relative not in disk_hashes:
                self._delete_source(relative)
                stale_removed += 1

        logger.info(
            "Qdrant reindex done collection=%s skipped=%s upserted=%s "
            "dup_removed=%s stale_removed=%s",
            self._collection,
            skipped,
            upserted,
            dup_removed,
            stale_removed,
        )

    def apply_changes(self, watch_path: str, changes: Sequence[FsChange]) -> None:
        if not changes:
            return
        logger.info(
            "Qdrant apply_changes path=%s ops=%s collection=%s",
            watch_path,
            [(item.op, item.path, item.is_prefix) for item in changes],
            self._collection,
        )
        deletes = [item for item in changes if item.op == "delete"]
        upserts = [item for item in changes if item.op == "upsert"]
        for item in deletes:
            if item.is_prefix:
                self._delete_prefix(item.path)
            else:
                self._delete_file(watch_path, item.path)
        for item in upserts:
            self._upsert_file(watch_path, item.path)

    def _delete_file(self, watch_path: str, relative: str) -> None:
        indexed_paths = self._scroll_indexed_paths()
        deleted_hash = indexed_paths.get(relative)
        disk_hashes = self._disk_file_hashes(watch_path)
        was_canonical = False
        if deleted_hash:
            peers = sorted(path for path, file_hash in disk_hashes.items() if file_hash == deleted_hash)
            was_canonical = not peers or relative < peers[0]

        self._delete_source(relative)

        if was_canonical and deleted_hash:
            next_path = _next_path_for_hash(disk_hashes, deleted_hash, exclude=set())
            if next_path:
                logger.info(
                    "Promoting duplicate source=%s after delete of canonical=%s",
                    next_path,
                    relative,
                )
                self._upsert_file(watch_path, next_path, force=True)

    def _upsert_file(self, watch_path: str, relative: str, *, force: bool = False) -> None:
        if not self._is_indexable_relative(relative):
            return
        path = Path(watch_path) / relative
        if not path.is_file():
            self._delete_source(relative)
            return

        content_hash = file_content_hash(path)
        if not force:
            disk_hashes = self._disk_file_hashes(watch_path)
            canonical = _canonical_paths(disk_hashes).get(content_hash)
            if canonical and canonical != relative:
                logger.info(
                    "Skip duplicate source=%s canonical=%s hash=%s",
                    relative,
                    canonical,
                    content_hash[:12],
                )
                if self._get_indexed_hash(relative) is not None:
                    self._delete_source(relative)
                return
            indexed_hash = self._get_indexed_hash(relative)
            if indexed_hash == content_hash:
                logger.info(
                    "Skip unchanged source=%s hash=%s",
                    relative,
                    content_hash[:12],
                )
                return

        embedded = self._embed_file(path, relative, content_hash)
        if embedded is None:
            logger.error(
                "Keeping existing Qdrant points after failed indexing source=%s",
                relative,
            )
            return
        points, vector_size = embedded
        if not points:
            self._delete_source(relative)
            return
        if vector_size is None:
            logger.error("Missing vector size source=%s; keeping existing points", relative)
            return
        self._ensure_collection(vector_size)
        self._delete_source(relative)
        if points:
            self._client.upsert(collection_name=self._collection, points=points)
            logger.info(
                "Qdrant upserted source=%s points=%s hash=%s",
                relative,
                len(points),
                content_hash[:12],
            )

    def _embed_file(
        self,
        path: Path,
        relative: str,
        content_hash: str,
    ) -> tuple[list[qmodels.PointStruct], int | None] | None:
        try:
            chunks = list(self._document_reader.read(path))
        except Exception:
            logger.exception("Failed to read document %s", relative)
            return None

        chunks = [
            chunk
            for chunk in chunks
            if chunk.text.strip() and is_useful_chunk_text(chunk.text)
        ]
        if not chunks:
            logger.info("Skip empty document %s", relative)
            return [], None

        parts_by_chunk = []
        for chunk in chunks:
            parts = tuple(
                part.strip()
                for part in (chunk.embedding_parts or (chunk.text,))
                if part.strip()
            )
            parts_by_chunk.append(parts or (chunk.text.strip(),))
        embedding_inputs = [
            part
            for parts in parts_by_chunk
            for part in parts
        ]
        logger.info(
            "Embed start path=%s chunks=%s embedding_parts=%s",
            relative,
            len(chunks),
            len(embedding_inputs),
        )
        started = time.perf_counter()
        try:
            part_vectors = self._embedder.embed_documents(embedding_inputs)
        except Exception:
            logger.exception("Failed to embed document %s chunks=%s", relative, len(chunks))
            return None
        logger.info(
            "Embed done path=%s chunks=%s embedding_parts=%s elapsed=%.2fs",
            relative,
            len(chunks),
            len(part_vectors),
            time.perf_counter() - started,
        )
        if len(part_vectors) != len(embedding_inputs):
            logger.error(
                "Embedding count mismatch path=%s got=%s expected_parts=%s",
                relative,
                len(part_vectors),
                len(embedding_inputs),
            )
            return None

        vectors: list[list[float]] = []
        cursor = 0
        for parts in parts_by_chunk:
            count = len(parts)
            grouped = part_vectors[cursor : cursor + count]
            cursor += count
            aggregated = _aggregate_vectors(grouped)
            if aggregated is None:
                logger.error(
                    "Cannot aggregate embeddings path=%s chunk=%s parts=%s",
                    relative,
                    len(vectors),
                    count,
                )
                return None
            vectors.append(aggregated)

        points: list[qmodels.PointStruct] = []
        vector_size: int | None = None
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if vector_size is None:
                vector_size = len(vector)
            elif len(vector) != vector_size:
                logger.error(
                    "Embedding size mismatch path=%s got=%s expected=%s",
                    relative,
                    len(vector),
                    vector_size,
                )
                continue
            payload: dict[str, object] = {
                "source_path": relative,
                "chunk_index": index,
                "text": chunk.text,
                "file_hash": content_hash,
                "index_version": _INDEX_VERSION,
                "chunk_type": chunk.chunk_type,
            }
            if chunk.headings:
                payload["headings"] = list(chunk.headings)
            if chunk.table_ref:
                payload["table_ref"] = chunk.table_ref
            if chunk.row_count:
                payload["row_count"] = chunk.row_count
            points.append(
                qmodels.PointStruct(
                    id=str(_point_id(relative, index)),
                    vector=vector,
                    payload=payload,
                )
            )
        if not points:
            return [], None
        return points, vector_size

    def _disk_file_hashes(self, watch_path: str) -> dict[str, str]:
        root = Path(watch_path)
        hashes: dict[str, str] = {}
        for path in iter_document_files(
            root,
            allowed_extensions=self._allowed_extensions or None,
        ):
            relative = path.relative_to(root).as_posix()
            hashes[relative] = file_content_hash(path)
        return hashes

    def _scroll_indexed_paths(self) -> dict[str, str]:
        if not self._client.collection_exists(self._collection):
            return {}
        indexed: dict[str, str] = {}
        offset: object | None = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                with_payload=["source_path", "file_hash"],
                with_vectors=False,
                limit=_SCROLL_LIMIT,
                offset=offset,
            )
            for record in records:
                payload = record.payload or {}
                source = str(payload.get("source_path") or "")
                if not source or source in indexed:
                    continue
                indexed[source] = str(payload.get("file_hash") or "")
            if offset is None:
                break
        return indexed

    def _get_indexed_hash(self, relative: str) -> str | None:
        if not self._client.collection_exists(self._collection):
            return None
        records, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_path",
                        match=qmodels.MatchValue(value=relative),
                    )
                ]
            ),
            with_payload=["file_hash"],
            with_vectors=False,
            limit=1,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        value = payload.get("file_hash")
        return str(value) if value else None

    def _is_indexable_relative(self, relative: str) -> bool:
        path = Path(relative)
        if path.name.startswith("."):
            return False
        suffix = path.suffix.lower()
        allowed = self._allowed_extensions
        if not allowed:
            from reindex.domain.formats import SUPPORTED_SUFFIXES

            allowed = frozenset(SUPPORTED_SUFFIXES)
        return suffix in allowed

    def _delete_source(self, relative: str) -> None:
        if not self._client.collection_exists(self._collection):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_path",
                            match=qmodels.MatchValue(value=relative),
                        )
                    ]
                )
            ),
        )
        logger.debug("Qdrant deleted source=%s", relative)

    def _delete_prefix(self, prefix: str) -> None:
        if not self._client.collection_exists(self._collection):
            return
        normalized = prefix.strip("/")
        if not normalized:
            return
        dir_prefix = f"{normalized}/"
        ids: list[object] = []
        offset: object | None = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                with_payload=["source_path"],
                with_vectors=False,
                limit=_SCROLL_LIMIT,
                offset=offset,
            )
            for record in records:
                payload = record.payload or {}
                source = str(payload.get("source_path") or "")
                if source == normalized or source.startswith(dir_prefix):
                    ids.append(record.id)
            if offset is None:
                break
        if ids:
            self._client.delete(
                collection_name=self._collection,
                points_selector=ids,
            )
            logger.info(
                "Qdrant deleted prefix=%s points=%s",
                normalized,
                len(ids),
            )

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._create_collection(vector_size)

    def _create_collection(self, vector_size: int) -> None:
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        for field in ("source_path", "file_hash", "chunk_type", "table_ref"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug(
                    "Could not create payload index on %s",
                    field,
                    exc_info=True,
                )


def file_content_hash(path: Path) -> str:
    """SHA-256 of file bytes and the indexing algorithm version."""
    digest = hashlib.sha256()
    digest.update(_INDEX_VERSION.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_vectors(vectors: list[list[float]]) -> list[float] | None:
    """Average unit vectors and normalize the result for cosine search."""
    if not vectors or not vectors[0]:
        return None
    if len(vectors) == 1:
        return [float(value) for value in vectors[0]]
    size = len(vectors[0])
    if any(len(vector) != size for vector in vectors):
        return None

    summed = [0.0] * size
    used = 0
    for vector in vectors:
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if norm <= 0:
            continue
        used += 1
        for index, value in enumerate(vector):
            summed[index] += float(value) / norm
    if used == 0:
        return None
    averaged = [value / used for value in summed]
    final_norm = math.sqrt(sum(value**2 for value in averaged))
    if final_norm <= 0:
        return None
    return [value / final_norm for value in averaged]


def _canonical_paths(path_hashes: dict[str, str]) -> dict[str, str]:
    """Map content hash to first sorted source_path on disk."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for relative, content_hash in path_hashes.items():
        by_hash[content_hash].append(relative)
    return {content_hash: sorted(paths)[0] for content_hash, paths in by_hash.items()}


def _next_path_for_hash(
    path_hashes: dict[str, str],
    content_hash: str,
    *,
    exclude: set[str],
) -> str | None:
    candidates = sorted(
        relative
        for relative, file_hash in path_hashes.items()
        if file_hash == content_hash and relative not in exclude
    )
    return candidates[0] if candidates else None


def _point_id(source_path: str, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(_POINT_NAMESPACE, f"{source_path}::{chunk_index}")
