"""Index watched documents into Qdrant via embeddings."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from reindex.adapters.document_readers import build_default_document_reader
from reindex.domain.documents import iter_document_files
from reindex.ports import DocumentReader, Embedder

logger = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class QdrantIndexer:
    """Full-directory reindex: read chunks → embed → recreate collection."""

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

        points: list[qmodels.PointStruct] = []
        vector_size: int | None = None

        for path in files:
            relative = path.relative_to(root).as_posix()
            try:
                chunks = list(self._document_reader.read(path))
            except Exception:
                logger.exception("Failed to read document %s", relative)
                continue

            chunks = [chunk for chunk in chunks if chunk.text.strip()]
            if not chunks:
                logger.info("Skip empty document %s", relative)
                continue

            file_hash = _file_fingerprint(path)
            for index, chunk in enumerate(chunks):
                try:
                    vector = self._embedder.embed(chunk.text)
                except Exception:
                    logger.exception(
                        "Failed to embed chunk path=%s index=%s",
                        relative,
                        index,
                    )
                    continue

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

                point_id = _point_id(relative, index)
                payload: dict[str, object] = {
                    "source_path": relative,
                    "chunk_index": index,
                    "text": chunk.text,
                    "file_hash": file_hash,
                }
                if chunk.headings:
                    payload["headings"] = list(chunk.headings)
                points.append(
                    qmodels.PointStruct(
                        id=str(point_id),
                        vector=vector,
                        payload=payload,
                    )
                )

        if vector_size is None:
            logger.warning(
                "No embeddable chunks under %s; leaving collection %s unchanged",
                watch_path,
                self._collection,
            )
            return

        self._recreate_collection(vector_size)
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

        logger.info(
            "Qdrant reindex done collection=%s points=%s vector_size=%s",
            self._collection,
            len(points),
            vector_size,
        )

    def _recreate_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )


def _point_id(source_path: str, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(_POINT_NAMESPACE, f"{source_path}::{chunk_index}")


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"
