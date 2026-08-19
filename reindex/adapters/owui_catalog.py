"""Read Open WebUI's SQLite catalog and drop leftover upload blobs."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from reindex.domain.upload_names import owui_upload_file_id

logger = logging.getLogger(__name__)

OWUI_DATABASE_NAME = "webui.db"
_ORPHAN_GRACE_SEC = 5.0

OwuiCatalogSnapshot = dict[str, tuple[str, int]]


@dataclass(frozen=True, slots=True)
class OwuiFile:
    """One row from Open WebUI's ``file`` table."""

    file_id: str
    filename: str
    file_hash: str
    updated_at: int
    content: str | None


def owui_database_path(watch_path: str | Path) -> Path:
    """``webui.db`` lives in DATA_DIR, one level above ``uploads/``."""
    return Path(watch_path).resolve().parent / OWUI_DATABASE_NAME


T = TypeVar("T")


def file_catalog_changed(previous: T | None, current: T | None) -> bool:
    """True when the catalog snapshot is readable and differs from the last one."""
    return current is not None and current != previous


def catalog_snapshot(files: Sequence[OwuiFile]) -> OwuiCatalogSnapshot:
    """Fingerprint ids plus ``hash``/``updated_at`` so in-app edits are visible."""
    return {item.file_id: (item.file_hash, item.updated_at) for item in files}


def catalog_content_overrides(files: Sequence[OwuiFile]) -> dict[str, bytes]:
    """UTF-8 bodies from ``file.data.content`` when OWUI has in-app text."""
    return {
        item.file_id: item.content.encode("utf-8")
        for item in files
        if item.content
    }


def list_owui_files(database_path: str | Path) -> list[OwuiFile] | None:
    """Return ``file`` rows, or ``None`` if the database is unreadable."""
    path = Path(database_path)
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=5.0,
        )
    except sqlite3.Error:
        logger.exception("Could not open Open WebUI database %s", path)
        return None
    try:
        rows = connection.execute(
            "SELECT id, filename, hash, updated_at, data FROM file"
        ).fetchall()
    except sqlite3.Error:
        logger.exception("Could not read file catalog from %s", path)
        return None
    finally:
        connection.close()
    files: list[OwuiFile] = []
    for row in rows:
        file_id = str(row[0]) if row and row[0] else ""
        if not file_id:
            continue
        files.append(
            OwuiFile(
                file_id=file_id,
                filename=str(row[1] or ""),
                file_hash=str(row[2] or ""),
                updated_at=int(row[3] or 0),
                content=_content_from_data(row[4]),
            )
        )
    return files


def list_owui_file_ids(database_path: str | Path) -> set[str] | None:
    """Return file IDs from OWUI's ``file`` table, or ``None`` if unreadable."""
    path = Path(database_path)
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=5.0,
        )
    except sqlite3.Error:
        logger.exception("Could not open Open WebUI database %s", path)
        return None
    try:
        rows = connection.execute("SELECT id FROM file").fetchall()
    except sqlite3.Error:
        logger.exception("Could not read file ids from %s", path)
        return None
    finally:
        connection.close()
    return {str(row[0]) for row in rows if row and row[0]}


def _content_from_data(raw: object) -> str | None:
    payload = raw
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    if not isinstance(payload, dict):
        return None
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def purge_orphaned_uploads(
    watch_path: str | Path,
    *,
    database_path: str | Path | None = None,
    now: Callable[[], float] | None = None,
    grace_sec: float = _ORPHAN_GRACE_SEC,
) -> list[Path]:
    """Unlink OWUI blobs whose file id is no longer in ``webui.db``.

    Files without the ``{uuid}_`` prefix (manual copies) are left alone.
    Very new files are skipped so an in-flight upload is not deleted before
    OWUI inserts the ``file`` row.
    """
    root = Path(watch_path)
    if not root.is_dir():
        return []
    catalog = Path(database_path) if database_path is not None else owui_database_path(root)
    live_ids = list_owui_file_ids(catalog)
    if live_ids is None:
        logger.info("Skipping OWUI orphan purge; database unavailable at %s", catalog)
        return []

    clock = now or time.time
    removed: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        file_id = owui_upload_file_id(path.name)
        if file_id is None or file_id in live_ids:
            continue
        age = clock() - path.stat().st_mtime
        if age < grace_sec:
            logger.debug(
                "Keeping recent OWUI upload %s (age=%.2fs, not yet in webui.db)",
                path.name,
                age,
            )
            continue
        logger.info(
            "Removing OWUI upload missing from webui.db file_id=%s path=%s",
            file_id,
            path.relative_to(root).as_posix(),
        )
        try:
            path.unlink()
        except OSError:
            logger.exception("Could not delete leftover OWUI upload %s", path)
            continue
        removed.append(path)
    return removed
