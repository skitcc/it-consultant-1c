"""Filesystem watcher that triggers debounced incremental indexing."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from reindex.domain.changes import FsChange
from reindex.ports import Indexer

logger = logging.getLogger(__name__)

# inotify also emits opened/closed when reindex *reads* files. Those must not
# retrigger indexing, or the watcher loops forever.
_TRIGGER_EVENTS = frozenset({"created", "deleted", "modified", "moved"})
# Commits land in the WAL; webui.db itself may stay untouched until checkpoint.
# Do not watch -shm: our own SQLite readers update it and would retrigger.
_OWUI_DB_NAMES = frozenset({"webui.db", "webui.db-wal"})


class DebouncedReindex:
    """Coalesce filesystem events and apply them after a quiet period."""

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
        self._running = False
        self._pending: dict[tuple[str, bool], FsChange] = {}

    def notify(self, change: FsChange) -> None:
        """Record a change and (re)schedule apply after the debounce window."""
        with self._lock:
            self._pending[(change.path, change.is_prefix)] = change
            if self._running:
                return
            self._arm_timer_locked()

    def _arm_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._timer = None
            changes = list(self._pending.values())
            self._pending.clear()
        if changes:
            logger.info(
                "Debounce elapsed; applying %s change(s) under %s",
                len(changes),
                self._watch_path,
            )
            try:
                self._indexer.apply_changes(self._watch_path, changes)
            except Exception:
                logger.exception(
                    "Indexer.apply_changes failed for %s",
                    self._watch_path,
                )
        with self._lock:
            self._running = False
            has_more = bool(self._pending)
            if has_more:
                self._arm_timer_locked()

    def cancel(self) -> None:
        """Cancel any pending apply timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()


class DebouncedCallback:
    """Coalesce bursts of events into a single callback after a quiet period."""

    def __init__(self, callback: Callable[[], None], debounce_seconds: float) -> None:
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False
        self._pending = False

    def notify(self) -> None:
        with self._lock:
            self._pending = True
            if self._running:
                return
            self._arm_timer_locked()

    def _arm_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._timer = None
            self._pending = False
        try:
            self._callback()
        except Exception:
            logger.exception("Debounced callback failed")
        with self._lock:
            self._running = False
            if self._pending:
                self._arm_timer_locked()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending = False


class ChangeHandler(FileSystemEventHandler):
    """Forward relevant FS events as incremental ``FsChange`` records."""

    def __init__(
        self,
        watch_path: str,
        on_change: Callable[[FsChange], None],
    ) -> None:
        super().__init__()
        self._watch_path = watch_path
        self._on_change = on_change

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in _TRIGGER_EVENTS:
            return
        if event.is_directory and event.event_type in {"modified", "created"}:
            return
        logger.debug(
            "FS event: type=%s path=%s dest=%s is_directory=%s",
            event.event_type,
            event.src_path,
            getattr(event, "dest_path", ""),
            event.is_directory,
        )
        for change in self._changes_from_event(event):
            self._on_change(change)

    def _changes_from_event(self, event: FileSystemEvent) -> list[FsChange]:
        if event.event_type == "moved":
            return self._changes_from_move(event)
        relative = self._relative(str(event.src_path))
        if relative is None:
            return []
        if event.is_directory:
            if event.event_type == "deleted":
                return [FsChange("delete", relative, is_prefix=True)]
            return []
        if event.event_type == "deleted":
            return [FsChange("delete", relative)]
        return [FsChange("upsert", relative)]

    def _changes_from_move(self, event: FileSystemEvent) -> list[FsChange]:
        src = self._relative(str(event.src_path))
        dest = self._relative(str(getattr(event, "dest_path", "") or ""))
        changes: list[FsChange] = []
        if event.is_directory:
            if src:
                changes.append(FsChange("delete", src, is_prefix=True))
            if dest:
                dest_dir = Path(self._watch_path) / dest
                if dest_dir.is_dir():
                    for path in dest_dir.rglob("*"):
                        if not path.is_file():
                            continue
                        relative = self._relative(str(path))
                        if relative is not None:
                            changes.append(FsChange("upsert", relative))
            return changes
        if src:
            changes.append(FsChange("delete", src))
        if dest:
            changes.append(FsChange("upsert", dest))
        return changes

    def _relative(self, src: str) -> str | None:
        if not src:
            return None
        try:
            relative = Path(src).relative_to(self._watch_path)
        except ValueError:
            return None
        posix = relative.as_posix()
        if posix == ".":
            return None
        return posix


class OpenWebUIDatabaseHandler(FileSystemEventHandler):
    """Notify when Open WebUI's SQLite catalog files change."""

    def __init__(self, on_change: Callable[[], None]) -> None:
        super().__init__()
        self._on_change = on_change

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in _TRIGGER_EVENTS:
            return
        names = {Path(str(event.src_path)).name}
        dest = str(getattr(event, "dest_path", "") or "")
        if dest:
            names.add(Path(dest).name)
        if names & _OWUI_DB_NAMES:
            logger.debug(
                "OWUI database event: type=%s path=%s",
                event.event_type,
                event.src_path,
            )
            self._on_change()


def create_observer(
    watch_path: str,
    handler: FileSystemEventHandler,
    *,
    recursive: bool = True,
) -> Observer:
    """Create and schedule an observer for ``watch_path``."""
    observer = Observer()
    observer.schedule(handler, path=watch_path, recursive=recursive)
    return observer
