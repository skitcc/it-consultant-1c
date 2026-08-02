"""Discover document files under the watch path."""

from __future__ import annotations

from pathlib import Path

from reindex.adapters.document_readers import SUPPORTED_SUFFIXES


def iter_document_files(watch_path: str | Path) -> list[Path]:
    root = Path(watch_path)
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files
