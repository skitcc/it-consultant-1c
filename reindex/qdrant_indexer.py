"""Index watched documents into Qdrant via Ollama embeddings."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from common.chunking import chunk_text
from common.embeddings import OllamaEmbedder
from reindex.documents import iter_document_files, read_document_text
from reindex.indexer import Indexer

logger = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class QdrantIndexer(Indexer):
    """Full-directory reindex: parse → chunk → embed → recreate collection."""

    def __init__(
        self,
        *,
        qdrant_url: str,
        collection: str,
        embedder: OllamaEmbedder,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ) -> None:
        self._client = QdrantClient(url=qdrant_url, check_compatibility=False)
        self._collection = collection
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def reindex(self, watch_path: str) -> None:
        root = Path(watch_path)
        files = iter_document_files(root)
        logger.info(
            "Qdrant reindex start path=%s files=%s collection=%s",
            watch_path,
            len(files),
            self._collection,
        )

        points: list[qmodels.PointStruct] = []
        vector_size: int | None = None

        for path in files:
            relative = path.relative_to(root).as_posix()
            try:
                text = read_document_text(path)
            except Exception:
                logger.exception("Failed to read document %s", relative)
                continue

            chunks = chunk_text(
                text,
                chunk_size=self._chunk_size,
                overlap=self._chunk_overlap,
            )
            if not chunks:
                logger.info("Skip empty document %s", relative)
                continue

            file_hash = _file_fingerprint(path)
            for index, chunk in enumerate(chunks):
                try:
                    vector = self._embedder.embed(chunk)
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
                points.append(
                    qmodels.PointStruct(
                        id=str(point_id),
                        vector=vector,
                        payload={
                            "source_path": relative,
                            "chunk_index": index,
                            "text": chunk,
                            "file_hash": file_hash,
                        },
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
