"""Unit tests for shared Settings and reindex watch_path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from common import Settings
from reindex.service import run


def _minimal_settings(**overrides) -> Settings:
    base = {
        "EWS_SERVER": "mail.example.com",
        "EWS_EMAIL": "bot@example.com",
        "EWS_PASSWORD": "secret",
        "WATCH_PATH": "/tmp",
        "DEBOUNCE_SECONDS": 0.5,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_settings_loads_reindex_fields() -> None:
    settings = _minimal_settings(WATCH_PATH="/data/db", DEBOUNCE_SECONDS=2.0)
    assert settings.watch_path == "/data/db"
    assert settings.debounce_seconds == 2.0
    assert settings.ews_server == "mail.example.com"


def test_settings_default_watch_path() -> None:
    settings = Settings(
        _env_file=None,
        EWS_SERVER="s",
        EWS_EMAIL="e",
        EWS_PASSWORD="p",
    )
    assert settings.watch_path == "/var/lib/it-consultant/db"
    assert settings.debounce_seconds == 1.0


def test_run_fails_when_watch_path_missing(tmp_path: Path) -> None:
    settings = _minimal_settings(WATCH_PATH=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="watch_path"):
        run(settings=settings)
