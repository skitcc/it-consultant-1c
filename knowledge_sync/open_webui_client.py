"""Read-only client for one Open WebUI Knowledge base."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class KnowledgeFile:
    file_id: str
    filename: str
    content_hash: str
    updated_at: str | None = None


class OpenWebUIClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_sec: float = 60.0,
        page_size: int = 100,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_sec
        self._page_size = page_size

    def list_knowledge_files(self, knowledge_id: str) -> list[KnowledgeFile]:
        result: list[KnowledgeFile] = []
        page = 1
        with self._client() as client:
            while True:
                response = client.get(
                    f"/api/v1/knowledge/{knowledge_id}/files",
                    params={"page": page, "limit": self._page_size},
                )
                response.raise_for_status()
                data = response.json()
                items, has_more = _page_items(data, page, self._page_size)
                result.extend(_knowledge_file(item) for item in items)
                if not has_more:
                    break
                page += 1
        return result

    def download_file(self, file_id: str) -> bytes:
        with self._client() as client:
            response = client.get(f"/api/v1/files/{file_id}/content")
            response.raise_for_status()
            return response.content

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._token}"},
        )


def _page_items(
    data: object,
    page: int,
    page_size: int,
) -> tuple[list[dict], bool]:
    if isinstance(data, list):
        items = [item for item in data if isinstance(item, dict)]
        return items, len(items) >= page_size
    if not isinstance(data, dict):
        raise ValueError("Open WebUI returned an invalid Knowledge file list")
    raw = data.get("items", data.get("files", data.get("data", [])))
    if not isinstance(raw, list):
        raise ValueError("Open WebUI Knowledge response has no file list")
    items = [item for item in raw if isinstance(item, dict)]
    total = data.get("total")
    if isinstance(total, int):
        return items, page * page_size < total
    return items, len(items) >= page_size


def _knowledge_file(item: dict) -> KnowledgeFile:
    file_id = str(item.get("id", item.get("file_id", ""))).strip()
    filename = str(item.get("filename", item.get("name", ""))).strip()
    if not file_id or not filename:
        raise ValueError("Open WebUI returned a file without id or filename")
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    content_hash = str(
        item.get("hash")
        or item.get("file_hash")
        or meta.get("file_hash")
        or ""
    )
    updated = item.get("updated_at", meta.get("updated_at"))
    return KnowledgeFile(
        file_id=file_id,
        filename=filename,
        content_hash=content_hash,
        updated_at=str(updated) if updated is not None else None,
    )
