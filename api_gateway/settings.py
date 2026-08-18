from __future__ import annotations

from pydantic import Field

from knowledge.settings import KnowledgeSettings


class GatewaySettings(KnowledgeSettings):
    api_gateway_host: str = Field(default="127.0.0.1", alias="API_GATEWAY_HOST")
    api_gateway_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        alias="API_GATEWAY_PORT",
    )
    api_gateway_api_key: str = Field(min_length=1, alias="API_GATEWAY_API_KEY")
    api_gateway_model: str = Field(
        default="it-consultant",
        min_length=1,
        alias="API_GATEWAY_MODEL",
    )
    owui_loader_key: str = Field(min_length=1, alias="OWUI_LOADER_KEY")
    max_upload_bytes: int = Field(
        default=100 * 1024 * 1024,
        gt=0,
        alias="MAX_UPLOAD_BYTES",
    )
