from __future__ import annotations

import httpx

from knowledge_sync.open_webui_client import OpenWebUIClient


def test_client_reads_paginated_metadata_and_original_bytes(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/files"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "file-1",
                            "filename": "manual.pdf",
                            "hash": "abc",
                            "updated_at": 123,
                        }
                    ],
                    "total": 1,
                },
            )
        return httpx.Response(200, content=b"\x00original\xff")

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    client = OpenWebUIClient(base_url="http://owui:8080", token="secret")

    files = client.list_knowledge_files("knowledge-1")
    raw = client.download_file("file-1")

    assert files[0].file_id == "file-1"
    assert files[0].filename == "manual.pdf"
    assert files[0].content_hash == "abc"
    assert raw == b"\x00original\xff"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert requests[0].url.path == "/api/v1/knowledge/knowledge-1/files"
