"""Open WebUI External Document Loader endpoint."""

from __future__ import annotations

from pathlib import PurePath
from urllib.parse import unquote

from fastapi import APIRouter, Header, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from api_gateway.auth import require_bearer
from knowledge.core.use_cases import IndexDocument, UpdateDocumentMetadata


def build_owui_router(
    *,
    index_document: IndexDocument,
    update_metadata: UpdateDocumentMetadata,
    loader_key: str,
    knowledge_id: str,
    max_upload_bytes: int,
    allowed_extensions: frozenset[str],
) -> APIRouter:
    router = APIRouter()
    authenticate = require_bearer(loader_key)

    @router.put("/process", dependencies=[])
    async def process_document(
        request: Request,
        authorization: str | None = Header(default=None),
        file_id: str | None = Header(default=None, alias="X-OpenWebUI-File-Id"),
        file_name: str | None = Header(default=None, alias="X-OpenWebUI-File-Name"),
        fallback_name: str | None = Header(default=None, alias="X-Filename"),
    ) -> dict:
        await authenticate(authorization)
        document_id = (file_id or "").strip()
        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-OpenWebUI-File-Id header is required",
            )
        filename = PurePath(unquote(file_name or fallback_name or "")).name
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document filename header is required",
            )
        if PurePath(filename).suffix.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported document type: {PurePath(filename).suffix}",
            )

        declared_size = _content_length(request.headers.get("content-length"))
        if declared_size is not None and declared_size > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Document exceeds MAX_UPLOAD_BYTES",
            )
        raw_bytes = await request.body()
        if not raw_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document body is empty",
            )
        if len(raw_bytes) > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Document exceeds MAX_UPLOAD_BYTES",
            )
        if declared_size is not None and declared_size != len(raw_bytes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incomplete document body",
            )

        result = await run_in_threadpool(
            index_document.execute,
            raw_bytes,
            filename,
            document_id=document_id,
            knowledge_id=knowledge_id,
            source_path=filename,
        )
        await run_in_threadpool(
            update_metadata.execute,
            document_id,
            knowledge_id=knowledge_id,
            filename=filename,
            source_path=filename,
            status="pending_confirmation",
            missing_count=0,
        )
        return {
            "page_content": "Документ проиндексирован внешним сервисом.",
            "metadata": {
                "document_id": result.document_id,
                "content_hash": result.content_hash,
                "chunk_count": result.chunk_count,
                "status": result.status,
            },
        }

    return router


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Content-Length",
        ) from exc
    if parsed < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Content-Length",
        )
    return parsed
