from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from knowledge.core.domain import DocumentRecord
from reindex.adapters.knowledge_indexer import KnowledgeIndexer
from reindex.domain.changes import FsChange


@dataclass
class Registry:
    records: dict[str, DocumentRecord] = field(default_factory=dict)

    def get(self, document_id: str, knowledge_id: str = "main"):
        record = self.records.get(document_id)
        if record is None or record.knowledge_id != knowledge_id:
            return None
        return record

    def list(self, knowledge_id: str = "main"):
        return [
            record
            for record in self.records.values()
            if record.knowledge_id == knowledge_id
        ]

    def save(self, record: DocumentRecord) -> None:
        self.records[record.document_id] = record

    def delete(self, document_id: str, knowledge_id: str = "main") -> None:
        current = self.records.get(document_id)
        if current is not None and current.knowledge_id == knowledge_id:
            del self.records[document_id]


class Index:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.calls: list[tuple[bytes, str, dict]] = []
        self.status = "indexed"

    def execute(self, raw_bytes, filename, **kwargs):
        self.calls.append((raw_bytes, filename, kwargs))
        document_id = kwargs["document_id"]
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        if self.status != "unchanged":
            self.registry.save(
                DocumentRecord(
                    document_id=document_id,
                    knowledge_id=kwargs.get("knowledge_id", "main"),
                    filename=filename,
                    content_hash=content_hash,
                    status=self.status,
                    chunk_count=1,
                )
            )
        return SimpleNamespace(
            document_id=document_id,
            content_hash=content_hash,
            chunk_count=1,
            status=self.status,
        )


class Remove:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.calls: list[str] = []

    def execute(self, document_id: str, *, knowledge_id: str = "main") -> bool:
        self.calls.append(document_id)
        existed = self.registry.get(document_id, knowledge_id) is not None
        self.registry.delete(document_id, knowledge_id)
        return existed


class Update:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, document_id: str, **kwargs):
        self.calls.append({"document_id": document_id, **kwargs})
        return None


def _indexer(tmp_path: Path) -> tuple[KnowledgeIndexer, Index, Remove, Update, Registry]:
    registry = Registry()
    index = Index(registry)
    remove = Remove(registry)
    update = Update()
    indexer = KnowledgeIndexer(
        index_document=index,
        remove_document=remove,
        update_metadata=update,
        registry=registry,
        allowed_extensions=frozenset({".pdf", ".md"}),
        max_upload_bytes=1024,
    )
    return indexer, index, remove, update, registry


def test_upsert_parses_owui_filename_and_indexes_bytes(tmp_path: Path) -> None:
    indexer, index, _, _, _ = _indexer(tmp_path)
    name = "38ec13c1-3127-4a81-b301-f0e2b6f72baa_manual.pdf"
    (tmp_path / name).write_bytes(b"%PDF-bytes")

    indexer.apply_changes(str(tmp_path), [FsChange("upsert", name)])

    raw, filename, kwargs = index.calls[0]
    assert raw == b"%PDF-bytes"
    assert filename == "manual.pdf"
    assert kwargs["document_id"] == "38ec13c1-3127-4a81-b301-f0e2b6f72baa"
    assert kwargs["source_path"] == "manual.pdf"


def test_delete_uses_owui_file_id(tmp_path: Path) -> None:
    indexer, _, remove, _, registry = _indexer(tmp_path)
    document_id = "38ec13c1-3127-4a81-b301-f0e2b6f72baa"
    registry.save(
        DocumentRecord(
            document_id=document_id,
            knowledge_id="main",
            filename="manual.pdf",
            content_hash="abc",
            status="indexed",
            chunk_count=1,
        )
    )

    indexer.apply_changes(
        str(tmp_path),
        [FsChange("delete", f"{document_id}_manual.pdf")],
    )

    assert remove.calls == [document_id]
    assert registry.list() == []


def test_reindex_removes_registry_entries_missing_from_disk(tmp_path: Path) -> None:
    indexer, index, remove, _, registry = _indexer(tmp_path)
    keep_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    gone_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    (tmp_path / f"{keep_id}_keep.md").write_text("keep", encoding="utf-8")
    registry.save(
        DocumentRecord(
            document_id=gone_id,
            knowledge_id="main",
            filename="gone.md",
            content_hash="old",
            status="indexed",
            chunk_count=1,
        )
    )

    indexer.reindex(str(tmp_path))

    assert index.calls[0][1] == "keep.md"
    assert gone_id in remove.calls


def test_prefix_delete_reconciles_disk(tmp_path: Path) -> None:
    indexer, _, remove, _, registry = _indexer(tmp_path)
    gone_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    registry.save(
        DocumentRecord(
            document_id=gone_id,
            knowledge_id="main",
            filename="nested.md",
            content_hash="old",
            status="indexed",
            chunk_count=1,
        )
    )

    indexer.apply_changes(str(tmp_path), [FsChange("delete", "subdir", is_prefix=True)])

    assert gone_id in remove.calls


def test_unchanged_rename_updates_metadata(tmp_path: Path) -> None:
    indexer, index, _, update, registry = _indexer(tmp_path)
    document_id = "38ec13c1-3127-4a81-b301-f0e2b6f72baa"
    index.status = "unchanged"
    registry.save(
        DocumentRecord(
            document_id=document_id,
            knowledge_id="main",
            filename="old.pdf",
            content_hash="abc",
            status="indexed",
            chunk_count=1,
        )
    )
    name = f"{document_id}_new.pdf"
    (tmp_path / name).write_bytes(b"%PDF")

    indexer.apply_changes(str(tmp_path), [FsChange("upsert", name)])

    assert update.calls[0]["filename"] == "new.pdf"
    assert update.calls[0]["source_path"] == "new.pdf"


def test_skips_oversize_and_unknown_extension(tmp_path: Path) -> None:
    indexer, index, _, _, _ = _indexer(tmp_path)
    (tmp_path / "tiny.bin").write_bytes(b"nope")
    big = tmp_path / "38ec13c1-3127-4a81-b301-f0e2b6f72baa_huge.pdf"
    big.write_bytes(b"x" * 2048)

    indexer.apply_changes(
        str(tmp_path),
        [
            FsChange("upsert", "tiny.bin"),
            FsChange("upsert", big.name),
        ],
    )

    assert index.calls == []


def test_skips_duplicate_content_and_indexes_canonical(tmp_path: Path) -> None:
    indexer, index, _, _, _ = _indexer(tmp_path)
    first = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa_faq.md"
    second = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb_faq.md"
    (tmp_path / first).write_text("same", encoding="utf-8")
    (tmp_path / second).write_text("same", encoding="utf-8")

    indexer.apply_changes(
        str(tmp_path),
        [FsChange("upsert", first), FsChange("upsert", second)],
    )

    assert len(index.calls) == 1
    assert index.calls[0][2]["document_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def test_delete_canonical_promotes_duplicate(tmp_path: Path) -> None:
    indexer, index, remove, _, registry = _indexer(tmp_path)
    first = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa_faq.md"
    second = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb_faq.md"
    (tmp_path / first).write_text("same", encoding="utf-8")
    (tmp_path / second).write_text("same", encoding="utf-8")
    indexer.apply_changes(str(tmp_path), [FsChange("upsert", first)])
    (tmp_path / first).unlink()

    indexer.apply_changes(str(tmp_path), [FsChange("delete", first)])

    assert remove.calls[-1] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert index.calls[-1][2]["document_id"] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert registry.get("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb") is not None


def test_catalog_content_overrides_disk_bytes(tmp_path: Path) -> None:
    indexer, index, _, _, _ = _indexer(tmp_path)
    file_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    name = f"{file_id}_faq.md"
    (tmp_path / name).write_text("disk version", encoding="utf-8")
    indexer.set_catalog_content({file_id: b"edited in owui"})

    indexer.apply_changes(str(tmp_path), [FsChange("upsert", name)])

    assert index.calls[0][0] == b"edited in owui"
