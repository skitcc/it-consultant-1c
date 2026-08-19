"""Discover document files under the watch path."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from reindex.domain.formats import SUPPORTED_SUFFIXES


def parse_index_extensions(raw: str | None) -> frozenset[str]:
    """Parse comma/space-separated extensions into a normalized ``.ext`` set."""
    if raw is None:
        return frozenset()
    normalized: set[str] = set()
    for part in raw.replace(";", ",").replace(" ", ",").split(","):
        item = part.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        normalized.add(item)
    return frozenset(normalized)


def resolve_index_extensions(
    raw: str | None,
    *,
    readable: Iterable[str] = SUPPORTED_SUFFIXES,
) -> frozenset[str]:
    """Return configured extensions that the indexer can actually read."""
    readable_set = {suffix.lower() for suffix in readable}
    configured = parse_index_extensions(raw)
    if not configured:
        return frozenset(readable_set)
    return frozenset(ext for ext in configured if ext in readable_set)


def iter_document_files(
    watch_path: str | Path,
    *,
    allowed_extensions: Iterable[str] | None = None,
) -> list[Path]:
    """List files under ``watch_path`` whose suffix is in ``allowed_extensions``."""
    if allowed_extensions is None:
        allowed = frozenset(SUPPORTED_SUFFIXES)
    else:
        allowed = frozenset(
            ext if ext.startswith(".") else f".{ext}"
            for ext in (item.strip().lower() for item in allowed_extensions)
            if ext
        )

    root = Path(watch_path)
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in allowed:
            files.append(path)
    return files
