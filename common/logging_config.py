"""Shared utilities used by mail_gateway and reindex."""

from __future__ import annotations

import logging

# Chatty HTTP / HF clients. At INFO they drown journald; at DEBUG they are useful.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "filelock",
    "qdrant_client",
)


def configure_logging(level: str = "INFO") -> None:
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logging.getLogger().setLevel(resolved)
    third_party_level = logging.DEBUG if resolved <= logging.DEBUG else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)
