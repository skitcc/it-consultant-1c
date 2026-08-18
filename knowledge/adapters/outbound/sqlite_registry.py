"""Persistent SQLite document registry for independent processes."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from knowledge.core.domain import DocumentRecord


class SQLiteDocumentRegistry:
    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self._path = Path(path)
        self._busy_timeout_ms = busy_timeout_ms
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT NOT NULL,
                    knowledge_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    source_updated_at TEXT,
                    last_seen_at TEXT,
                    missing_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (knowledge_id, document_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_knowledge_status "
                "ON documents (knowledge_id, status)"
            )
            connection.commit()

    def get(
        self,
        document_id: str,
        knowledge_id: str = "main",
    ) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT document_id, knowledge_id, filename, content_hash, status,
                       chunk_count, source_updated_at, last_seen_at, missing_count
                FROM documents
                WHERE knowledge_id = ? AND document_id = ?
                """,
                (knowledge_id, document_id),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list(self, knowledge_id: str = "main") -> list[DocumentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT document_id, knowledge_id, filename, content_hash, status,
                       chunk_count, source_updated_at, last_seen_at, missing_count
                FROM documents
                WHERE knowledge_id = ?
                ORDER BY filename, document_id
                """,
                (knowledge_id,),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def save(self, record: DocumentRecord) -> None:
        if record.chunk_count < 0 or record.missing_count < 0:
            raise ValueError("Document counts must be non-negative")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, knowledge_id, filename, content_hash, status,
                    chunk_count, source_updated_at, last_seen_at, missing_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (knowledge_id, document_id) DO UPDATE SET
                    filename = excluded.filename,
                    content_hash = excluded.content_hash,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count,
                    source_updated_at = excluded.source_updated_at,
                    last_seen_at = excluded.last_seen_at,
                    missing_count = excluded.missing_count
                """,
                (
                    record.document_id,
                    record.knowledge_id,
                    record.filename,
                    record.content_hash,
                    record.status,
                    record.chunk_count,
                    record.source_updated_at,
                    record.last_seen_at,
                    record.missing_count,
                ),
            )
            connection.commit()

    def delete(self, document_id: str, knowledge_id: str = "main") -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM documents WHERE knowledge_id = ? AND document_id = ?",
                (knowledge_id, document_id),
            )
            connection.commit()


def _record_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        document_id=str(row["document_id"]),
        knowledge_id=str(row["knowledge_id"]),
        filename=str(row["filename"]),
        content_hash=str(row["content_hash"]),
        status=str(row["status"]),
        chunk_count=int(row["chunk_count"]),
        source_updated_at=row["source_updated_at"],
        last_seen_at=row["last_seen_at"],
        missing_count=int(row["missing_count"]),
    )
