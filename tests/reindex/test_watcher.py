"""Integration tests for the reindex watcher service."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from reindex.domain.changes import FsChange
from reindex.ports import Indexer
from reindex.watcher import ChangeHandler, DebouncedReindex, create_observer


class RecordingIndexer(Indexer):
    """Test double that records incremental apply_changes / reindex calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.applied: list[list[FsChange]] = []
        self.fail_next = False

    def reindex(self, watch_path: str) -> None:
        self.calls.append(watch_path)

    def apply_changes(self, watch_path: str, changes: Sequence[FsChange]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated indexer failure")
        self.calls.append(watch_path)
        self.applied.append(list(changes))


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"Condition not met within {timeout}s")


def _ops(indexer: RecordingIndexer) -> list[tuple[str, str, bool]]:
    return [
        (change.op, change.path, change.is_prefix)
        for batch in indexer.applied
        for change in batch
    ]


@pytest.fixture
def watch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "db"
    d.mkdir()
    return d


@pytest.fixture
def running_watcher(watch_dir: Path):
    indexer = RecordingIndexer()
    debouncer = DebouncedReindex(
        indexer=indexer,
        watch_path=str(watch_dir),
        debounce_seconds=0.2,
    )
    handler = ChangeHandler(str(watch_dir), debouncer.notify)
    observer = create_observer(str(watch_dir), handler)
    observer.start()
    time.sleep(0.15)
    try:
        yield indexer, watch_dir, debouncer
    finally:
        debouncer.cancel()
        observer.stop()
        observer.join(timeout=5)


def test_file_create_upserts_that_path(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    (watch_dir / "doc.txt").write_text("hello", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    assert ("upsert", "doc.txt", False) in _ops(indexer)
    assert indexer.calls == [str(watch_dir)]


def test_file_modify_upserts_that_path(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    path = watch_dir / "doc.txt"
    path.write_text("v1", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    indexer.applied.clear()
    indexer.calls.clear()

    path.write_text("v2", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    assert ("upsert", "doc.txt", False) in _ops(indexer)


def test_file_delete_deletes_that_path(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    path = watch_dir / "doc.txt"
    path.write_text("x", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    indexer.applied.clear()
    indexer.calls.clear()

    path.unlink()
    _wait_until(lambda: indexer.applied)
    assert ("delete", "doc.txt", False) in _ops(indexer)


def test_subdirectory_create_does_not_index_until_file(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    sub = watch_dir / "subdir"
    sub.mkdir()
    time.sleep(0.4)
    assert indexer.applied == []

    (sub / "nested.txt").write_text("nested", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    assert ("upsert", "subdir/nested.txt", False) in _ops(indexer)


def test_subdirectory_delete_removes_nested_file(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    sub = watch_dir / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    indexer.applied.clear()
    indexer.calls.clear()

    (sub / "nested.txt").unlink()
    sub.rmdir()
    _wait_until(lambda: indexer.applied)
    ops = _ops(indexer)
    assert ("delete", "subdir/nested.txt", False) in ops or (
        "delete",
        "subdir",
        True,
    ) in ops


def test_move_rename_deletes_old_and_upserts_new(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    src = watch_dir / "old.txt"
    src.write_text("data", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    indexer.applied.clear()
    indexer.calls.clear()

    src.rename(watch_dir / "new.txt")
    _wait_until(lambda: indexer.applied)
    ops = _ops(indexer)
    assert ("delete", "old.txt", False) in ops
    assert ("upsert", "new.txt", False) in ops


def test_burst_of_files_is_one_apply_batch(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    for i in range(10):
        (watch_dir / f"f{i}.txt").write_text(str(i), encoding="utf-8")
        time.sleep(0.02)
    _wait_until(lambda: indexer.applied, timeout=5.0)
    time.sleep(0.5)
    assert len(indexer.applied) == 1
    upserted = {change.path for change in indexer.applied[0] if change.op == "upsert"}
    assert upserted == {f"f{i}.txt" for i in range(10)}


def test_indexer_exception_does_not_crash_watcher(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    indexer.fail_next = True
    (watch_dir / "boom.txt").write_text("x", encoding="utf-8")
    _wait_until(lambda: not indexer.fail_next, timeout=5.0)
    (watch_dir / "ok.txt").write_text("y", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    assert any(change.path == "ok.txt" for change in indexer.applied[-1])


def test_opened_and_closed_events_do_not_trigger() -> None:
    calls: list[FsChange] = []
    handler = ChangeHandler("/tmp", lambda change: calls.append(change))
    for event in _open_close_events("/tmp/doc.md"):
        handler.on_any_event(event)
    assert calls == []


def test_created_event_upserts_relative_path() -> None:
    from watchdog.events import FileCreatedEvent

    calls: list[FsChange] = []
    handler = ChangeHandler("/tmp", lambda change: calls.append(change))
    handler.on_any_event(FileCreatedEvent("/tmp/doc.md"))
    assert calls == [FsChange("upsert", "doc.md")]


def test_deleted_dir_emits_prefix_delete() -> None:
    from watchdog.events import DirDeletedEvent

    calls: list[FsChange] = []
    handler = ChangeHandler("/tmp", lambda change: calls.append(change))
    handler.on_any_event(DirDeletedEvent("/tmp/subdir"))
    assert calls == [FsChange("delete", "subdir", is_prefix=True)]


def test_reading_watched_file_does_not_reindex(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    path = watch_dir / "doc.txt"
    path.write_text("v1", encoding="utf-8")
    _wait_until(lambda: indexer.applied)
    indexer.applied.clear()
    indexer.calls.clear()

    path.read_text(encoding="utf-8")
    time.sleep(0.6)
    assert indexer.applied == []


def test_events_during_apply_coalesce_to_one_followup() -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowIndexer(RecordingIndexer):
        def apply_changes(self, watch_path: str, changes: Sequence[FsChange]) -> None:
            started.set()
            if not release.wait(timeout=5.0):
                raise TimeoutError("test gate was not released")
            super().apply_changes(watch_path, changes)

    indexer = SlowIndexer()
    debouncer = DebouncedReindex(
        indexer=indexer,
        watch_path="/tmp/db",
        debounce_seconds=0.05,
    )
    first = FsChange("upsert", "a.txt")
    extra = FsChange("upsert", "b.txt")
    try:
        debouncer.notify(first)
        _wait_until(started.is_set)
        for _ in range(5):
            debouncer.notify(extra)
        release.set()
        _wait_until(lambda: len(indexer.applied) >= 2)
        time.sleep(0.2)
        assert len(indexer.applied) == 2
        assert indexer.applied[0] == [first]
        assert indexer.applied[1] == [extra]
    finally:
        release.set()
        debouncer.cancel()


def test_last_write_wins_per_path() -> None:
    indexer = RecordingIndexer()
    debouncer = DebouncedReindex(
        indexer=indexer,
        watch_path="/tmp/db",
        debounce_seconds=0.05,
    )
    try:
        debouncer.notify(FsChange("upsert", "a.txt"))
        debouncer.notify(FsChange("delete", "a.txt"))
        _wait_until(lambda: indexer.applied)
        time.sleep(0.15)
        assert indexer.applied == [[FsChange("delete", "a.txt")]]
    finally:
        debouncer.cancel()


def _open_close_events(src: str) -> list:
    events = []
    try:
        from watchdog.events import FileOpenedEvent

        events.append(FileOpenedEvent(src))
    except ImportError:
        pass
    try:
        from watchdog.events import FileClosedEvent

        events.append(FileClosedEvent(src))
    except ImportError:
        pass
    if not events:
        from watchdog.events import FileSystemEvent

        events.append(FileSystemEvent(src))
        events[-1].event_type = "opened"  # type: ignore[misc]
        events.append(FileSystemEvent(src))
        events[-1].event_type = "closed"  # type: ignore[misc]
    return events
