"""Tests for shared Settings."""

from __future__ import annotations

from common import Settings


def test_settings_mail_and_reindex_fields() -> None:
    settings = Settings(
        _env_file=None,
        EWS_SERVER="mail.example.com",
        EWS_EMAIL="bot@example.com",
        EWS_PASSWORD="secret",
        AI_SERVICE_URL="http://ai/v1/ask",
        ASSISTANT_MODE="stub",
        WATCH_PATH="/var/db",
        DEBOUNCE_SECONDS=1.5,
        LOG_LEVEL="DEBUG",
    )
    assert settings.ews_server == "mail.example.com"
    assert settings.ai_service_url == "http://ai/v1/ask"
    assert settings.watch_path == "/var/db"
    assert settings.debounce_seconds == 1.5
    assert settings.log_level == "DEBUG"
