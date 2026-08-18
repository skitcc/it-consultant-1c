"""Incrementally reconcile one OWUI Knowledge base with the document index."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from knowledge.core.ports import DocumentRegistry
from knowledge.core.use_cases import (
    IndexDocument,
    RemoveDocument,
    UpdateDocumentMetadata,
)
from knowledge_sync.open_webui_client import KnowledgeFile, OpenWebUIClient

logger = logging.getLogger(__name__)


class KnowledgeSynchronizer:
    def __init__(
        self,
        *,
        client: OpenWebUIClient,
        registry: DocumentRegistry,
        index_document: IndexDocument,
        remove_document: RemoveDocument,
        update_metadata: UpdateDocumentMetadata,
        knowledge_id: str = "main",
        source_knowledge_id: str | None = None,
        delete_grace_snapshots: int = 3,
    ) -> None:
        if delete_grace_snapshots < 1:
            raise ValueError("delete_grace_snapshots must be positive")
        self._client = client
        self._registry = registry
        self._index_document = index_document
        self._remove_document = remove_document
        self._update_metadata = update_metadata
        self._knowledge_id = knowledge_id
        self._source_knowledge_id = source_knowledge_id or knowledge_id
        self._delete_grace = delete_grace_snapshots

    def synchronize_once(self) -> None:
        """Apply one complete successful OWUI snapshot incrementally."""
        files = self._client.list_knowledge_files(self._source_knowledge_id)
        remote = {item.file_id: item for item in files}
        records = {
            item.document_id: item
            for item in self._registry.list(self._knowledge_id)
        }
        now = datetime.now(UTC).isoformat()

        for file_id, item in remote.items():
            current = records.get(file_id)
            if current is None or _content_changed(current, item):
                raw_bytes = self._client.download_file(file_id)
                self._index_document.execute(
                    raw_bytes,
                    item.filename,
                    document_id=file_id,
                    knowledge_id=self._knowledge_id,
                    source_path=item.filename,
                    source_updated_at=item.updated_at,
                    last_seen_at=now,
                )
                current = self._registry.get(file_id, self._knowledge_id)
                logger.info("Synchronized document id=%s filename=%s", file_id, item.filename)

            if current is None:
                continue
            self._update_metadata.execute(
                file_id,
                knowledge_id=self._knowledge_id,
                filename=item.filename if current.filename != item.filename else None,
                status="active",
                source_updated_at=item.updated_at,
                last_seen_at=now,
                missing_count=0,
            )

        for document_id, record in records.items():
            if document_id in remote:
                continue
            missing_count = record.missing_count + 1
            if missing_count >= self._delete_grace:
                self._remove_document.execute(
                    document_id,
                    knowledge_id=self._knowledge_id,
                )
                logger.info("Removed document absent from Knowledge id=%s", document_id)
                continue
            self._update_metadata.execute(
                document_id,
                knowledge_id=self._knowledge_id,
                missing_count=missing_count,
            )


def _content_changed(record, remote: KnowledgeFile) -> bool:
    if remote.content_hash:
        return remote.content_hash != record.content_hash
    if remote.updated_at is not None:
        return remote.updated_at != record.source_updated_at
    return False
