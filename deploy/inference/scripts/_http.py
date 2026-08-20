"""Minimal JSON HTTP helper (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> tuple[int, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
        try:
            return status, json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError):
            return status, {"error": raw.decode("utf-8", errors="replace")}
    except urllib.error.URLError as exc:
        raise ConnectionError(f"{url}: {exc}") from exc
    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError:
        return status, {"error": "non-json", "text": raw.decode("utf-8", errors="replace")}


def ns_to_sec(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value >= 1_000_000:
        return float(value) / 1_000_000_000.0
    return float(value)
