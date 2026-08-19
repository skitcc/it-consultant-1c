"""Composition root and run loop for the reindex service."""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from common.logging_config import configure_logging
from knowledge.bootstrap import KnowledgeContainer, build_container
from knowledge.settings import KnowledgeSettings
from reindex.adapters.knowledge_indexer import KnowledgeIndexer
from reindex.adapters.owui_catalog import (
    catalog_content_overrides,
    catalog_snapshot,
    file_catalog_changed,
    list_owui_files,
    owui_database_path,
    purge_orphaned_uploads,
)
from reindex.domain.documents import parse_index_extensions, resolve_index_extensions
from reindex.ports import Indexer
from reindex.watcher import (
    ChangeHandler,
    DebouncedCallback,
    DebouncedReindex,
    OpenWebUIDatabaseHandler,
    create_observer,
)

logger = logging.getLogger(__name__)


def build_indexer(
    settings: KnowledgeSettings,
    container: KnowledgeContainer | None = None,
) -> Indexer:
    configured = parse_index_extensions(settings.index_extensions)
    allowed = resolve_index_extensions(settings.index_extensions)
    unknown = configured - allowed
    if unknown:
        logger.warning(
            "INDEX_EXTENSIONS has unsupported types (no reader): %s",
            sorted(unknown),
        )
    logger.info("Index extensions enabled: %s", sorted(allowed))
    services = container or build_container(settings)
    return KnowledgeIndexer(
        index_document=services.index_document,
        remove_document=services.remove_document,
        update_metadata=services.update_metadata,
        registry=services.registry,
        knowledge_id=settings.default_knowledge_id,
        allowed_extensions=allowed,
        max_upload_bytes=settings.max_upload_bytes,
    )


def _apply_catalog_content(indexer: Indexer, files: list) -> None:
    setter = getattr(indexer, "set_catalog_content", None)
    if setter is None:
        return
    setter(catalog_content_overrides(files))


def run(
    settings: KnowledgeSettings | None = None,
    indexer: Indexer | None = None,
    *,
    once: bool = False,
) -> None:
    """Start watching ``settings.watch_path`` and reindex on changes.

    Blocks until SIGINT/SIGTERM or KeyboardInterrupt, unless ``once`` is set.
    """
    cfg = settings if settings is not None else KnowledgeSettings()
    configure_logging(cfg.log_level)

    watch = Path(cfg.watch_path)
    if not watch.exists():
        raise FileNotFoundError(f"watch_path does not exist: {watch}")
    data_dir = watch.parent
    database = owui_database_path(watch)

    idx = indexer if indexer is not None else build_indexer(cfg)

    logger.info(
        "Starting reindex on %s (once=%s debounce=%ss qdrant=%s "
        "collection=%s owui_db=%s extensions=%s)",
        watch,
        once,
        cfg.debounce_seconds,
        cfg.qdrant_url,
        cfg.qdrant_collection,
        database,
        sorted(resolve_index_extensions(cfg.index_extensions)),
    )
    last_snapshot = None
    files = list_owui_files(database)
    if files is not None:
        last_snapshot = catalog_snapshot(files)
        _apply_catalog_content(idx, files)
    try:
        purge_orphaned_uploads(watch, database_path=database)
        idx.reindex(str(watch))
    except Exception:
        logger.exception("Initial reindex failed for %s", watch)
        if once:
            raise

    if once:
        logger.info("Reindex --once complete")
        return

    debouncer = DebouncedReindex(
        indexer=idx,
        watch_path=str(watch),
        debounce_seconds=cfg.debounce_seconds,
    )
    handler = ChangeHandler(str(watch), debouncer.notify)
    observer = create_observer(str(watch), handler)

    def _on_owui_catalog() -> None:
        nonlocal last_snapshot
        current_files = list_owui_files(database)
        if current_files is None:
            return
        snapshot = catalog_snapshot(current_files)
        if not file_catalog_changed(last_snapshot, snapshot):
            return
        last_snapshot = snapshot
        _apply_catalog_content(idx, current_files)
        logger.debug("Open WebUI file catalog changed; reconciling uploads")
        removed = purge_orphaned_uploads(watch, database_path=database)
        if removed:
            logger.info("Purged %s leftover OWUI upload(s)", len(removed))
        try:
            idx.reindex(str(watch))
        except Exception:
            logger.exception("Catalog reindex failed for %s", watch)

    db_debouncer = DebouncedCallback(_on_owui_catalog, cfg.debounce_seconds)
    db_observer = create_observer(
        str(data_dir),
        OpenWebUIDatabaseHandler(db_debouncer.notify),
        recursive=False,
    )

    stop_event = threading.Event()

    def _shutdown(signum: int | None = None, _frame: object = None) -> None:
        if signum is not None:
            logger.info("Received signal %s; shutting down", signum)
        else:
            logger.info("Shutting down")
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    observer.start()
    db_observer.start()
    logger.info("Watching Open WebUI database %s", database)
    try:
        while not stop_event.wait(timeout=1.0):
            pass
    except KeyboardInterrupt:
        _shutdown()
    finally:
        debouncer.cancel()
        db_debouncer.cancel()
        observer.stop()
        db_observer.stop()
        observer.join()
        db_observer.join()
        logger.info("Reindex watcher stopped")
