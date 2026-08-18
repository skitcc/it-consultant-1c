from __future__ import annotations

import hashlib
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_gateway.routes.owui import build_owui_router


class Index:
    def __init__(self) -> None:
        self.raw_bytes: bytes | None = None

    def execute(self, raw_bytes, filename, **kwargs):
        self.raw_bytes = raw_bytes
        return SimpleNamespace(
            document_id=kwargs["document_id"],
            content_hash=hashlib.sha256(raw_bytes).hexdigest(),
            chunk_count=2,
            status="indexed",
        )


class Update:
    def execute(self, *args, **kwargs):
        return None


def test_process_preserves_original_binary_body() -> None:
    index = Index()
    app = FastAPI()
    app.include_router(
        build_owui_router(
            index_document=index,
            update_metadata=Update(),
            loader_key="loader-secret",
            knowledge_id="main",
            max_upload_bytes=1024,
            allowed_extensions=frozenset({".pdf"}),
        )
    )
    raw = b"%PDF-1.7\x00\xff\r\nbinary-content"

    response = TestClient(app).put(
        "/process",
        content=raw,
        headers={
            "Authorization": "Bearer loader-secret",
            "Content-Type": "application/pdf",
            "X-OpenWebUI-File-Id": "file-123",
            "X-OpenWebUI-File-Name": "manual.pdf",
        },
    )

    assert response.status_code == 200
    assert index.raw_bytes == raw
    assert response.json()["metadata"]["content_hash"] == hashlib.sha256(raw).hexdigest()


def test_process_rejects_unsupported_extension_before_indexing() -> None:
    index = Index()
    app = FastAPI()
    app.include_router(
        build_owui_router(
            index_document=index,
            update_metadata=Update(),
            loader_key="secret",
            knowledge_id="main",
            max_upload_bytes=1024,
            allowed_extensions=frozenset({".pdf"}),
        )
    )

    response = TestClient(app).put(
        "/process",
        content=b"exe",
        headers={
            "Authorization": "Bearer secret",
            "X-OpenWebUI-File-Id": "id",
            "X-OpenWebUI-File-Name": "bad.exe",
        },
    )

    assert response.status_code == 415
    assert index.raw_bytes is None
