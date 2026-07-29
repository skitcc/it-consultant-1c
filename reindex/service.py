"""Composition root and run loop for the reindex service."""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from common import Settings
from common.logging_config import configure_logging
from reindex.indexer import Indexer, LoggingIndexer
from reindex.watcher import ChangeHandler, DebouncedReindex, create_observer

logger = logging.getLogger(__name__)


def run(
    settings: Settings | None = None,
    indexer: Indexer | None = None,
) -> None:
    """Start watching ``settings.watch_path`` and reindex on changes.

    Blocks until SIGINT/SIGTERM or KeyboardInterrupt.
    """
    cfg = settings if settings is not None else Settings()
    configure_logging(cfg.log_level)

    watch = Path(cfg.watch_path)
    if not watch.exists():
        raise FileNotFoundError(f"watch_path does not exist: {watch}")

    idx = indexer if indexer is not None else LoggingIndexer()

    debouncer = DebouncedReindex(
        indexer=idx,
        watch_path=str(watch),
        debounce_seconds=cfg.debounce_seconds,
    )
    handler = ChangeHandler(on_change=debouncer.notify)
    observer = create_observer(str(watch), handler)

    stop_event = threading.Event()

    def _shutdown(signum: int | None = None, _frame: object = None) -> None:
        if signum is not None:
            logger.info("Received signal %s; shutting down", signum)
        else:
            logger.info("Shutting down")
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(
        "Starting reindex watcher on %s (debounce=%ss)",
        watch,
        cfg.debounce_seconds,
    )
    observer.start()
    try:
        while not stop_event.wait(timeout=1.0):
            pass
    except KeyboardInterrupt:
        _shutdown()
    finally:
        debouncer.cancel()
        observer.stop()
        observer.join()
        logger.info("Reindex watcher stopped")
