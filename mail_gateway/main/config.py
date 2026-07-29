"""Application settings loaded from environment / .env."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
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
    ews_auth: str = Field(default="ntlm", alias="EWS_AUTH")
    ews_verify_ssl: bool = Field(default=True, alias="EWS_VERIFY_SSL")
    ews_streaming_timeout_minutes: int = Field(
        default=30,
        alias="EWS_STREAMING_TIMEOUT_MINUTES",
    )

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
