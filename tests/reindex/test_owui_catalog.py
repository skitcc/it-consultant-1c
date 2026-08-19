from __future__ import annotations

import sqlite3
from pathlib import Path

from reindex.adapters.owui_catalog import (
    catalog_content_overrides,
    catalog_snapshot,
    list_owui_file_ids,
    list_owui_files,
    owui_database_path,
    purge_orphaned_uploads,
)


def _write_catalog(path: Path, ids: list[str]) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE file (id TEXT PRIMARY KEY)")
    connection.executemany("INSERT INTO file (id) VALUES (?)", [(item,) for item in ids])
    connection.commit()
    connection.close()


def test_file_catalog_changed_ignores_unreadable_and_identical() -> None:
    from reindex.adapters.owui_catalog import file_catalog_changed

    assert file_catalog_changed(None, None) is False
    assert file_catalog_changed({"a": ("h", 1)}, None) is False
    assert file_catalog_changed({"a": ("h", 1)}, {"a": ("h", 1)}) is False
    assert file_catalog_changed(None, {"a": ("h", 1)}) is True
    assert file_catalog_changed({"a": ("h", 1)}, {"a": ("h2", 1)}) is True


def _write_full_catalog(
    path: Path,
    rows: list[tuple[str, str, str, int, str | None]],
) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE file (id TEXT PRIMARY KEY, filename TEXT, hash TEXT, "
        "updated_at INTEGER, data TEXT)"
    )
    connection.executemany(
        "INSERT INTO file (id, filename, hash, updated_at, data) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


def test_list_owui_files_reads_edited_content(tmp_path: Path) -> None:
    db_path = tmp_path / "webui.db"
    _write_full_catalog(
        db_path,
        [
            (
                "aaa",
                "faq.md",
                "hash-1",
                100,
                '{"content": "edited body", "status": "completed"}',
            ),
            ("bbb", "empty.md", "", 0, '{"status": "completed"}'),
        ],
    )

    files = list_owui_files(db_path)
    assert files is not None
    by_id = {item.file_id: item for item in files}
    assert by_id["aaa"].content == "edited body"
    assert by_id["aaa"].file_hash == "hash-1"
    assert by_id["bbb"].content is None
    assert catalog_snapshot(files) == {
        "aaa": ("hash-1", 100),
        "bbb": ("", 0),
    }
    assert catalog_content_overrides(files) == {"aaa": b"edited body"}


def test_owui_database_path_is_sibling_of_uploads(tmp_path: Path) -> None:
    uploads = tmp_path / "owui-data" / "uploads"
    uploads.mkdir(parents=True)
    assert owui_database_path(uploads) == tmp_path / "owui-data" / "webui.db"


def test_list_owui_file_ids_reads_file_table(tmp_path: Path) -> None:
    db_path = tmp_path / "webui.db"
    _write_catalog(db_path, ["aaa", "bbb"])
    assert list_owui_file_ids(db_path) == {"aaa", "bbb"}


def test_list_owui_file_ids_returns_none_when_missing(tmp_path: Path) -> None:
    assert list_owui_file_ids(tmp_path / "webui.db") is None


def test_purge_removes_uuid_blob_missing_from_catalog(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    live_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    gone_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    keep = uploads / f"{live_id}_keep.pdf"
    gone = uploads / f"{gone_id}_gone.pdf"
    manual = uploads / "faq.md"
    keep.write_bytes(b"%PDF")
    gone.write_bytes(b"%PDF")
    manual.write_text("manual", encoding="utf-8")
    catalog = tmp_path / "webui.db"
    _write_catalog(catalog, [live_id])
    later = gone.stat().st_mtime + 60

    removed = purge_orphaned_uploads(
        uploads,
        database_path=catalog,
        now=lambda: later,
        grace_sec=5.0,
    )

    assert [path.name for path in removed] == [gone.name]
    assert keep.exists()
    assert manual.exists()
    assert not gone.exists()


def test_purge_keeps_recent_upload_not_yet_in_catalog(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    file_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    inflight = uploads / f"{file_id}_new.pdf"
    inflight.write_bytes(b"%PDF")
    catalog = tmp_path / "webui.db"
    _write_catalog(catalog, [])
    mtime = inflight.stat().st_mtime

    removed = purge_orphaned_uploads(
        uploads,
        database_path=catalog,
        now=lambda: mtime + 1.0,
        grace_sec=5.0,
    )

    assert removed == []
    assert inflight.exists()


def test_purge_skips_when_database_missing(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    blob = uploads / "dddddddd-dddd-dddd-dddd-dddddddddddd_doc.pdf"
    blob.write_bytes(b"%PDF")

    removed = purge_orphaned_uploads(uploads, database_path=tmp_path / "missing.db")

    assert removed == []
    assert blob.exists()
