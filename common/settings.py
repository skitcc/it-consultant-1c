"""Shared application settings for all services."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single settings object for mail_gateway and reindex (env / .env)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- mail_gateway ---
    ews_server: str = Field(alias="EWS_SERVER")
    ews_email: str = Field(alias="EWS_EMAIL")
    ews_password: str = Field(alias="EWS_PASSWORD")
    ews_username: str | None = Field(default=None, alias="EWS_USERNAME")
    ews_auth: str = Field(default="ntlm", alias="EWS_AUTH")
    ews_verify_ssl: bool = Field(default=True, alias="EWS_VERIFY_SSL")
    ews_streaming_timeout_minutes: int = Field(
        default=30,
        alias="EWS_STREAMING_TIMEOUT_MINUTES",
    )
    ews_session_pool_size: int = Field(default=2, alias="EWS_SESSION_POOL_SIZE")
    ews_ignore_own_mail: bool = Field(default=True, alias="EWS_IGNORE_OWN_MAIL")
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

    # --- reindex ---
    watch_path: str = Field(
        default="/var/lib/it-consultant/db",
        alias="WATCH_PATH",
    )
    debounce_seconds: float = Field(default=1.0, alias="DEBOUNCE_SECONDS")

    # --- shared ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("ews_username", mode="before")
    @classmethod
    def empty_username_as_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value
