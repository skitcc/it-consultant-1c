"""Filesystem watcher that triggers debounced reindexing."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from reindex.ports import Indexer

logger = logging.getLogger(__name__)


class DebouncedReindex:
    """Coalesce filesystem events and call ``indexer.reindex`` after quiet period."""

    def __init__(
        self,
        indexer: Indexer,
        watch_path: str,
        debounce_seconds: float,
    ) -> None:
        self._indexer = indexer
        self._watch_path = watch_path
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def notify(self) -> None:
        """Schedule (or reschedule) a reindex after the debounce window."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self) -> None:
        logger.info("Debounce elapsed; starting reindex for %s", self._watch_path)
        try:
            self._indexer.reindex(self._watch_path)
        except Exception:
            logger.exception("Indexer.reindex failed for %s", self._watch_path)

    def cancel(self) -> None:
        """Cancel any pending reindex timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class ChangeHandler(FileSystemEventHandler):
    """Forward relevant FS events to a debounced reindex trigger."""

    def __init__(self, on_change: Callable[[], None]) -> None:
        super().__init__()
        self._on_change = on_change

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory and event.event_type == "modified":
            return
        logger.debug(
            "FS event: type=%s path=%s is_directory=%s",
            event.event_type,
            event.src_path,
            event.is_directory,
        )
        self._on_change()


def create_observer(
    watch_path: str,
    handler: FileSystemEventHandler,
) -> Observer:
    """Create and schedule a recursive observer for ``watch_path``."""
    observer = Observer()
    observer.schedule(handler, path=watch_path, recursive=True)
    return observer
