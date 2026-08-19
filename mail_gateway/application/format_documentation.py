"""Format retrieved documentation chunks for the system prompt."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mail_gateway.domain.models import DocumentChunk


def document_display_name(source_path: str) -> str:
    name = Path(source_path or "").name.strip()
    return name or (source_path or "").strip()


def unique_source_names(chunks: Sequence[DocumentChunk]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        name = document_display_name(chunk.source_path)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def format_documentation_context(chunks: Sequence[DocumentChunk]) -> str:
    if not chunks:
        return ""

    lines = ["Релевантные фрагменты документации:"]
    for chunk in chunks:
        name = document_display_name(chunk.source_path) or chunk.source_path
        header = f"Документ: {name}"
        if chunk.headings:
            header = f"{header}. Раздел: {' / '.join(chunk.headings)}"
        lines.append(header)
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()
