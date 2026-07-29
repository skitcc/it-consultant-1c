"""Integration tests for the reindex watcher service."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from reindex.indexer import Indexer
from reindex.watcher import ChangeHandler, DebouncedReindex, create_observer


class RecordingIndexer(Indexer):
    """Test double that records reindex calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_next = False

    def reindex(self, watch_path: str) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated indexer failure")
        self.calls.append(watch_path)


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"Condition not met within {timeout}s")


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
    handler = ChangeHandler(on_change=debouncer.notify)
    observer = create_observer(str(watch_dir), handler)
    observer.start()
    # Let the observer thread settle before generating events.
    time.sleep(0.15)
    try:
        yield indexer, watch_dir, debouncer
    finally:
        debouncer.cancel()
        observer.stop()
        observer.join(timeout=5)


def test_file_create_triggers_reindex(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    (watch_dir / "doc.txt").write_text("hello", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    assert indexer.calls == [str(watch_dir)]


def test_file_modify_triggers_reindex(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    path = watch_dir / "doc.txt"
    path.write_text("v1", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    indexer.calls.clear()

    path.write_text("v2", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    assert indexer.calls == [str(watch_dir)]


def test_file_delete_triggers_reindex(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    path = watch_dir / "doc.txt"
    path.write_text("x", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    indexer.calls.clear()

    path.unlink()
    _wait_until(lambda: len(indexer.calls) >= 1)
    assert indexer.calls == [str(watch_dir)]


def test_subdirectory_and_nested_file(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    sub = watch_dir / "subdir"
    sub.mkdir()
    _wait_until(lambda: len(indexer.calls) >= 1)
    indexer.calls.clear()

    (sub / "nested.txt").write_text("nested", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    assert indexer.calls[-1] == str(watch_dir)


def test_subdirectory_delete(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    sub = watch_dir / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    indexer.calls.clear()

    (sub / "nested.txt").unlink()
    sub.rmdir()
    _wait_until(lambda: len(indexer.calls) >= 1)
    assert indexer.calls[-1] == str(watch_dir)


def test_move_rename_triggers_reindex(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    src = watch_dir / "old.txt"
    src.write_text("data", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    indexer.calls.clear()

    src.rename(watch_dir / "new.txt")
    _wait_until(lambda: len(indexer.calls) >= 1)
    assert indexer.calls == [str(watch_dir)]


def test_burst_of_events_debounces_to_single_reindex(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    for i in range(10):
        (watch_dir / f"f{i}.txt").write_text(str(i), encoding="utf-8")
        time.sleep(0.02)
    _wait_until(lambda: len(indexer.calls) >= 1, timeout=5.0)
    # Allow a short window for any extra delayed callbacks.
    time.sleep(0.5)
    assert len(indexer.calls) == 1
    assert indexer.calls[0] == str(watch_dir)


def test_indexer_exception_does_not_crash_watcher(running_watcher) -> None:
    indexer, watch_dir, _ = running_watcher
    indexer.fail_next = True
    (watch_dir / "boom.txt").write_text("x", encoding="utf-8")
    _wait_until(lambda: not indexer.fail_next, timeout=5.0)
    # Watcher still alive: a subsequent change should trigger another reindex.
    (watch_dir / "ok.txt").write_text("y", encoding="utf-8")
    _wait_until(lambda: len(indexer.calls) >= 1)
    assert indexer.calls == [str(watch_dir)]
