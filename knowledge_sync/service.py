"""Run the standalone incremental Open WebUI Knowledge synchronizer."""

from __future__ import annotations

import logging
import signal
import threading

from common.logging_config import configure_logging
from knowledge.bootstrap import build_container
from knowledge_sync.open_webui_client import OpenWebUIClient
from knowledge_sync.settings import SyncSettings
from knowledge_sync.synchronizer import KnowledgeSynchronizer

logger = logging.getLogger(__name__)


def run(settings: SyncSettings | None = None) -> None:
    cfg = settings or SyncSettings()
    configure_logging(cfg.log_level)
    container = build_container(cfg)
    synchronizer = KnowledgeSynchronizer(
        client=OpenWebUIClient(
            base_url=cfg.open_webui_base_url,
            token=cfg.open_webui_sync_token,
        ),
        registry=container.registry,
        index_document=container.index_document,
        remove_document=container.remove_document,
        update_metadata=container.update_metadata,
        knowledge_id=cfg.default_knowledge_id,
        source_knowledge_id=cfg.open_webui_knowledge_id,
        delete_grace_snapshots=cfg.knowledge_delete_grace_snapshots,
    )
    stop_event = threading.Event()

    def stop(signum: int, _frame: object) -> None:
        logger.info("Knowledge sync received signal=%s", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info(
        "Knowledge sync started owui=%s knowledge_id=%s interval=%ss",
        cfg.open_webui_base_url,
        cfg.open_webui_knowledge_id,
        cfg.knowledge_sync_interval_sec,
    )
    while not stop_event.is_set():
        try:
            synchronizer.synchronize_once()
        except Exception:
            logger.exception("Knowledge synchronization failed; index left unchanged")
        stop_event.wait(cfg.knowledge_sync_interval_sec)

