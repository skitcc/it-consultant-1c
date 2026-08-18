"""Tests for mail gateway Settings."""

from __future__ import annotations

from common import Settings


def test_settings_mail_and_gateway_fields() -> None:
    settings = Settings(
        _env_file=None,
        EWS_SERVER="mail.example.com",
        EWS_EMAIL="bot@example.com",
        EWS_PASSWORD="secret",
        API_GATEWAY_BASE_URL="http://gateway:8000/v1",
        API_GATEWAY_API_KEY="api-secret",
        API_GATEWAY_MODEL="it-consultant",
        LOG_LEVEL="DEBUG",
    )
    assert settings.ews_server == "mail.example.com"
    assert settings.api_gateway_base_url == "http://gateway:8000/v1"
    assert settings.api_gateway_api_key == "api-secret"
    assert settings.api_gateway_model == "it-consultant"
    assert settings.log_level == "DEBUG"
