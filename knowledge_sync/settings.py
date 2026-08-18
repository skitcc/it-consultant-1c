from __future__ import annotations

from pydantic import Field

from knowledge.settings import KnowledgeSettings


class SyncSettings(KnowledgeSettings):
    open_webui_base_url: str = Field(
        default="http://127.0.0.1:3000",
        alias="OPEN_WEBUI_BASE_URL",
    )
    open_webui_knowledge_id: str = Field(
        min_length=1,
        alias="OPEN_WEBUI_KNOWLEDGE_ID",
    )
    open_webui_sync_token: str = Field(
        min_length=1,
        alias="OPEN_WEBUI_SYNC_TOKEN",
    )
    knowledge_sync_interval_sec: float = Field(
        default=10.0,
        gt=0,
        alias="KNOWLEDGE_SYNC_INTERVAL_SEC",
    )
    knowledge_delete_grace_snapshots: int = Field(
        default=3,
        ge=1,
        alias="KNOWLEDGE_DELETE_GRACE_SNAPSHOTS",
    )
