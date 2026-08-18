"""Composition root and run loop for the reindex service."""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from common import Settings
from common.embeddings import OllamaEmbedder
from common.logging_config import configure_logging
from reindex.adapters.document_readers import (
    PictureDescriptionConfig,
    build_default_document_reader,
)
from reindex.adapters.qdrant_indexer import QdrantIndexer
from reindex.domain.documents import parse_index_extensions, resolve_index_extensions
from reindex.ports import Indexer
from reindex.watcher import ChangeHandler, DebouncedReindex, create_observer

logger = logging.getLogger(__name__)


def build_indexer(settings: Settings) -> Indexer:
    configured = parse_index_extensions(settings.index_extensions)
    allowed = resolve_index_extensions(settings.index_extensions)
    unknown = configured - allowed
    if unknown:
        logger.warning(
            "INDEX_EXTENSIONS has unsupported types (no reader): %s",
            sorted(unknown),
        )
    logger.info("Index extensions enabled: %s", sorted(allowed))
    if settings.picture_description_enabled:
        logger.info(
            "Picture description enabled model=%s ollama=%s",
            settings.vlm_model,
            settings.ollama_base_url,
        )
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        timeout_sec=settings.embedding_timeout_sec,
    )
    reader = build_default_document_reader(
        max_tokens=settings.chunk_size,
        picture=PictureDescriptionConfig(
            enabled=settings.picture_description_enabled,
            ollama_base_url=settings.ollama_base_url,
            model=settings.vlm_model,
            timeout_sec=settings.vlm_timeout_sec,
            area_threshold=settings.picture_area_threshold,
        ),
    )
    return QdrantIndexer(
        qdrant_url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        embedder=embedder,
        document_reader=reader,
        allowed_extensions=allowed,
    )


def run(
    settings: Settings | None = None,
    indexer: Indexer | None = None,
    *,
    once: bool = False,
) -> None:
    """Start watching ``settings.watch_path`` and reindex on changes.

    Blocks until SIGINT/SIGTERM or KeyboardInterrupt, unless ``once`` is set.
    """
    cfg = settings if settings is not None else Settings()
    configure_logging(cfg.log_level)

    watch = Path(cfg.watch_path)
    if not watch.exists():
        raise FileNotFoundError(f"watch_path does not exist: {watch}")

    idx = indexer if indexer is not None else build_indexer(cfg)

    logger.info(
        "Starting reindex on %s (once=%s debounce=%ss qdrant=%s collection=%s extensions=%s)",
        watch,
        once,
        cfg.debounce_seconds,
        cfg.qdrant_url,
        cfg.qdrant_collection,
        sorted(resolve_index_extensions(cfg.index_extensions)),
    )
    try:
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
