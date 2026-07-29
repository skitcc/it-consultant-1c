"""Application settings loaded from environment / .env."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ews_server: str = Field(alias="EWS_SERVER")
    ews_email: str = Field(alias="EWS_EMAIL")
    ews_password: str = Field(alias="EWS_PASSWORD")
    # AD login for EWS (often DOMAIN\user). Defaults to EWS_EMAIL if empty.
    ews_username: str | None = Field(default=None, alias="EWS_USERNAME")
    ews_auth: str = Field(default="ntlm", alias="EWS_AUTH")
    ews_verify_ssl: bool = Field(default=True, alias="EWS_VERIFY_SSL")
    ews_streaming_timeout_minutes: int = Field(
        default=30,
        alias="EWS_STREAMING_TIMEOUT_MINUTES",
    )
    # Skip messages where sender == our mailbox (anti reply-loop).
    ews_ignore_own_mail: bool = Field(default=True, alias="EWS_IGNORE_OWN_MAIL")
    # On start, also process unread inbox mail from the last N minutes (0 = off).
    ews_catchup_minutes: int = Field(default=30, alias="EWS_CATCHUP_MINUTES")

    ai_service_url: str = Field(
        default="http://127.0.0.1:8000/v1/ask",
        alias="AI_SERVICE_URL",
    )
    ai_timeout_sec: float = Field(default=120.0, alias="AI_TIMEOUT_SEC")
    assistant_mode: Literal["http", "stub"] = Field(
        default="stub",
        alias="ASSISTANT_MODE",
    )

    reconnect_delay_sec: float = Field(default=5.0, alias="RECONNECT_DELAY_SEC")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("ews_username", mode="before")
    @classmethod
    def empty_username_as_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value
