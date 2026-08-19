"""Unit tests for reindex watch_path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge.settings import KnowledgeSettings
from reindex.service import run


def _minimal_settings(**overrides) -> KnowledgeSettings:
    base = {
        "WATCH_PATH": "/tmp",
        "DEBOUNCE_SECONDS": 0.5,
        "DOCUMENT_REGISTRY_PATH": "/tmp/registry.sqlite3",
    }
    base.update(overrides)
    return KnowledgeSettings(_env_file=None, **base)


def test_settings_loads_reindex_fields() -> None:
    settings = _minimal_settings(WATCH_PATH="/data/uploads", DEBOUNCE_SECONDS=2.0)
    assert settings.watch_path == "/data/uploads"
    assert settings.debounce_seconds == 2.0


def test_settings_default_watch_path() -> None:
    settings = KnowledgeSettings(_env_file=None)
    assert settings.watch_path == "/var/lib/it-consultant/owui-data/uploads"
    assert settings.debounce_seconds == 1.0


def test_run_once_reindexes_and_returns(tmp_path: Path) -> None:
    docs = tmp_path / "uploads"
    docs.mkdir()
    calls: list[str] = []

    class OnceIndexer:
        def reindex(self, watch_path: str) -> None:
            calls.append(watch_path)

        def apply_changes(self, watch_path: str, changes) -> None:
            raise AssertionError("apply_changes must not run in --once")

    settings = _minimal_settings(WATCH_PATH=str(docs))
    run(settings=settings, indexer=OnceIndexer(), once=True)
    assert calls == [str(docs)]


def test_run_fails_when_watch_path_missing(tmp_path: Path) -> None:
    settings = _minimal_settings(WATCH_PATH=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="watch_path"):
        run(settings=settings)
