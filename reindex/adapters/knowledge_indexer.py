"""Index watched files through Knowledge Core use cases."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath

from knowledge.core.ports import DocumentRegistry
from knowledge.core.use_cases import IndexDocument, RemoveDocument, UpdateDocumentMetadata
from reindex.domain.changes import FsChange
from reindex.domain.documents import iter_document_files
from reindex.domain.duplicates import canonical_paths, next_path_for_hash
from reindex.domain.upload_names import parse_watched_upload

logger = logging.getLogger(__name__)

# OWUI stores in-app edits as UTF-8 text in webui.db, not as a new blob.
_BINARY_EDIT_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}


class KnowledgeIndexer:
    """Map filesystem upserts/deletes onto IndexDocument / RemoveDocument."""

    def __init__(
        self,
        *,
        index_document: IndexDocument,
        remove_document: RemoveDocument,
        update_metadata: UpdateDocumentMetadata,
        registry: DocumentRegistry,
        knowledge_id: str = "main",
        allowed_extensions: frozenset[str] | None = None,
        max_upload_bytes: int | None = None,
    ) -> None:
        self._index_document = index_document
        self._remove_document = remove_document
        self._update_metadata = update_metadata
        self._registry = registry
        self._knowledge_id = knowledge_id
        self._allowed_extensions = allowed_extensions
        self._max_upload_bytes = max_upload_bytes
        self._catalog_content: dict[str, bytes] = {}

    def set_catalog_content(self, content: Mapping[str, bytes] | None) -> None:
        """Prefer OWUI ``file.data.content`` over the upload blob when present."""
        self._catalog_content = dict(content or {})

    def reindex(self, watch_path: str) -> None:
        root = Path(watch_path)
        present_ids: set[str] = set()
        hashes = self._content_hashes(root)
        canonical = canonical_paths(hashes)
        for path in iter_document_files(
            root,
            allowed_extensions=self._allowed_extensions,
        ):
            relative = path.relative_to(root).as_posix()
            identity = parse_watched_upload(
                relative,
                knowledge_id=self._knowledge_id,
            )
            content_hash = hashes.get(relative)
            if content_hash is not None and canonical.get(content_hash) != relative:
                logger.info(
                    "Skip duplicate path=%s canonical=%s hash=%s",
                    relative,
                    canonical[content_hash],
                    content_hash[:12],
                )
                continue
            present_ids.add(identity.document_id)
            try:
                self._upsert_file(root, relative, hashes=hashes)
            except Exception:
                logger.exception("Failed to index %s", relative)
        for record in self._registry.list(self._knowledge_id):
            if record.document_id in present_ids:
                continue
            logger.info(
                "Removing missing document_id=%s filename=%s",
                record.document_id,
                record.filename,
            )
            self._remove_document.execute(
                record.document_id,
                knowledge_id=self._knowledge_id,
            )

    def apply_changes(self, watch_path: str, changes: Sequence[FsChange]) -> None:
        root = Path(watch_path)
        needs_reconcile = False
        deletes: list[str] = []
        upserts: list[str] = []
        for change in changes:
            if change.op == "delete" and change.is_prefix:
                needs_reconcile = True
            elif change.op == "delete":
                deletes.append(change.path)
            elif change.op == "upsert":
                upserts.append(change.path)
        for relative in deletes:
            self._delete_path(root, relative)
        if needs_reconcile:
            self.reindex(watch_path)
            return
        hashes = self._content_hashes(root)
        for relative in upserts:
            try:
                self._upsert_file(root, relative, hashes=hashes)
            except Exception:
                logger.exception("Failed to index %s", relative)

    def _delete_path(self, root: Path, relative: str) -> None:
        identity = parse_watched_upload(
            relative,
            knowledge_id=self._knowledge_id,
        )
        record = self._registry.get(identity.document_id, self._knowledge_id)
        removed = self._remove_document.execute(
            identity.document_id,
            knowledge_id=self._knowledge_id,
        )
        if removed:
            logger.info(
                "Deleted document_id=%s path=%s",
                identity.document_id,
                relative,
            )
        if record is None:
            return
        next_path = next_path_for_hash(
            self._content_hashes(root),
            record.content_hash,
        )
        if next_path is None:
            return
        logger.info(
            "Promoting duplicate path=%s after delete of canonical=%s",
            next_path,
            relative,
        )
        self._upsert_file(root, next_path)

    def _upsert_file(
        self,
        root: Path,
        relative: str,
        *,
        hashes: dict[str, str] | None = None,
    ) -> None:
        path = root / relative
        if not path.is_file() or path.name.startswith("."):
            return
        suffix = path.suffix.lower()
        if self._allowed_extensions is not None and suffix not in self._allowed_extensions:
            logger.debug("Skipping unsupported type %s", relative)
            return
        identity = parse_watched_upload(
            relative,
            knowledge_id=self._knowledge_id,
        )
        raw_bytes, from_catalog = self._load_bytes(root, relative)
        if not raw_bytes:
            logger.warning("Skipping empty file %s", relative)
            return
        if (
            self._max_upload_bytes is not None
            and len(raw_bytes) > self._max_upload_bytes
        ):
            logger.warning(
                "Skipping %s: size %s exceeds MAX_UPLOAD_BYTES",
                relative,
                len(raw_bytes),
            )
            return
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        path_hashes = hashes if hashes is not None else self._content_hashes(root)
        canonical = canonical_paths(path_hashes).get(content_hash)
        if canonical is not None and canonical != relative:
            logger.info(
                "Skip duplicate path=%s canonical=%s hash=%s",
                relative,
                canonical,
                content_hash[:12],
            )
            if self._registry.get(identity.document_id, self._knowledge_id) is not None:
                self._remove_document.execute(
                    identity.document_id,
                    knowledge_id=self._knowledge_id,
                )
            return
        parse_name = identity.filename
        if from_catalog and PurePath(identity.filename).suffix.lower() in _BINARY_EDIT_SUFFIXES:
            parse_name = str(PurePath(identity.filename).with_suffix(".md"))
        result = self._index_document.execute(
            raw_bytes,
            parse_name,
            document_id=identity.document_id,
            knowledge_id=self._knowledge_id,
            source_path=identity.source_path,
        )
        if parse_name != identity.filename or (
            result.status == "unchanged"
            and self._needs_filename_sync(identity.document_id, identity.filename)
        ):
            self._update_metadata.execute(
                identity.document_id,
                knowledge_id=self._knowledge_id,
                filename=identity.filename,
                source_path=identity.source_path,
            )
        if result.status == "unchanged":
            logger.info(
                "Unchanged document_id=%s path=%s hash=%s",
                identity.document_id,
                relative,
                result.content_hash,
            )
            return
        logger.info(
            "Indexed document_id=%s path=%s chunks=%s hash=%s",
            identity.document_id,
            relative,
            result.chunk_count,
            result.content_hash,
        )

    def _needs_filename_sync(self, document_id: str, filename: str) -> bool:
        current = self._registry.get(document_id, self._knowledge_id)
        return current is not None and current.filename != filename

    def _load_bytes(self, root: Path, relative: str) -> tuple[bytes, bool]:
        identity = parse_watched_upload(
            relative,
            knowledge_id=self._knowledge_id,
        )
        override = self._catalog_content.get(identity.document_id)
        if override:
            return override, True
        return (root / relative).read_bytes(), False

    def _content_hashes(self, root: Path) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path in iter_document_files(
            root,
            allowed_extensions=self._allowed_extensions,
        ):
            relative = path.relative_to(root).as_posix()
            raw_bytes, _from_catalog = self._load_bytes(root, relative)
            if not raw_bytes:
                continue
            hashes[relative] = hashlib.sha256(raw_bytes).hexdigest()
        return hashes
