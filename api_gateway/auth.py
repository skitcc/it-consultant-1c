"""Bearer-key authentication dependencies."""

from __future__ import annotations

import hmac
from collections.abc import Callable

from fastapi import Header, HTTPException, status


def require_bearer(expected_key: str) -> Callable:
    async def dependency(authorization: str | None = Header(default=None)) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        valid = (
            scheme.lower() == "bearer"
            and bool(supplied)
            and hmac.compare_digest(supplied, expected_key)
        )
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return dependency
