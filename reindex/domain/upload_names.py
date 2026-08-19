"""Map Open WebUI upload filenames to stable document identities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath

from knowledge.core.use_cases.index_document import stable_document_id

# OWUI stores blobs as ``{file_uuid}_{original_filename}`` under uploads/.
_OWUI_UPLOAD = re.compile(
    r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_(.+)$"
)


@dataclass(frozen=True, slots=True)
class WatchedUpload:
    document_id: str
    filename: str
    source_path: str


def owui_upload_file_id(filename: str) -> str | None:
    """Return the OWUI file UUID if ``filename`` is ``{uuid}_{original}``."""
    match = _OWUI_UPLOAD.match(PurePath(filename).name)
    if match is None:
        return None
    return match.group(1)


def parse_watched_upload(
    relative_path: str,
    *,
    knowledge_id: str = "main",
) -> WatchedUpload:
    """Derive ``document_id`` and citation names from a path under WATCH_PATH."""
    posix = relative_path.replace("\\", "/").strip("/")
    if not posix:
        raise ValueError("relative_path must not be empty")
    name = PurePath(posix).name
    parent = posix[: -len(name)].rstrip("/") if posix != name else ""
    match = _OWUI_UPLOAD.match(name)
    if match is not None:
        filename = match.group(2)
        source_path = f"{parent}/{filename}" if parent else filename
        return WatchedUpload(
            document_id=match.group(1),
            filename=filename,
            source_path=source_path,
        )
    return WatchedUpload(
        document_id=stable_document_id(knowledge_id, posix),
        filename=name,
        source_path=posix,
    )
