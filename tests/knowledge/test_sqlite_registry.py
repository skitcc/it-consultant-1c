from __future__ import annotations

import sqlite3
from dataclasses import replace

from knowledge.adapters.outbound.sqlite_registry import SQLiteDocumentRegistry
from knowledge.core.domain import DocumentRecord


def record(document_id: str, knowledge_id: str = "main") -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        knowledge_id=knowledge_id,
        filename=f"{document_id}.pdf",
        content_hash=f"hash-{document_id}",
        status="indexed",
        chunk_count=2,
        source_updated_at="2026-08-18T12:00:00Z",
        last_seen_at="2026-08-18T12:01:00Z",
        missing_count=0,
    )


def test_registry_persists_and_scopes_records_between_instances(tmp_path):
    path = tmp_path / "state" / "registry.sqlite3"
    first = SQLiteDocumentRegistry(path)
    first.save(record("a"))
    first.save(record("b", "other"))

    second = SQLiteDocumentRegistry(path)

    assert second.get("a") == record("a")
    assert second.list() == [record("a")]
    assert second.list("other") == [record("b", "other")]
    second.delete("a")
    assert first.get("a") is None


def test_registry_uses_wal_and_upserts_in_place(tmp_path):
    path = tmp_path / "registry.sqlite3"
    registry = SQLiteDocumentRegistry(path, busy_timeout_ms=3210)
    registry.save(record("a"))
    changed = replace(record("a"), status="missing", missing_count=3)
    registry.save(changed)

    assert registry.get("a") == changed
    with sqlite3.connect(path) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
