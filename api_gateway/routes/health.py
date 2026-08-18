"""Liveness and dependency readiness routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool


def build_health_router(readiness: Callable[[], None]) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/ready")
    async def ready() -> dict[str, str]:
        try:
            await run_in_threadpool(readiness)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Knowledge dependencies are unavailable",
            ) from exc
        return {"status": "ready"}

    return router
