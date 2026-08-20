"""Per-request step timing for mail_gateway logs."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_TIMER: ContextVar[RequestTimer | None] = ContextVar("request_timer", default=None)


@dataclass
class RequestTimer:
    conversation_id: str
    item_id: str = ""
    started_at: float = field(default_factory=time.perf_counter)
    steps: list[tuple[str, float]] = field(default_factory=list)
    _token: Token[RequestTimer | None] | None = field(default=None, repr=False)

    def record(self, name: str, elapsed: float) -> None:
        self.steps.append((name, elapsed))
        logger.info(
            "Timing step=%s elapsed=%.3fs conversation_id=%s",
            name,
            elapsed,
            self.conversation_id,
        )

    def span(self, name: str) -> _Span:
        return _Span(self, name)

    def log_summary(self) -> None:
        total = time.perf_counter() - self.started_at
        parts = " ".join(f"{name}={elapsed:.3f}s" for name, elapsed in self.steps)
        extra = f"{parts} " if parts else ""
        logger.info(
            "Timing summary conversation_id=%s item_id=%s %stotal=%.3fs",
            self.conversation_id,
            self.item_id or "-",
            extra,
            total,
        )


class _Span:
    def __init__(self, timer: RequestTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._started = 0.0

    def __enter__(self) -> _Span:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._timer.record(self._name, time.perf_counter() - self._started)


class _NullSpan:
    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def begin_request(*, conversation_id: str, item_id: str = "") -> RequestTimer:
    timer = RequestTimer(conversation_id=conversation_id, item_id=item_id)
    timer._token = _TIMER.set(timer)
    return timer


def end_request(timer: RequestTimer) -> None:
    try:
        timer.log_summary()
    finally:
        if timer._token is not None:
            _TIMER.reset(timer._token)
            timer._token = None


def span(name: str) -> _Span | _NullSpan:
    timer = _TIMER.get()
    if timer is None:
        return _NullSpan()
    return timer.span(name)


def record(name: str, elapsed: float) -> None:
    timer = _TIMER.get()
    if timer is not None:
        timer.record(name, elapsed)
