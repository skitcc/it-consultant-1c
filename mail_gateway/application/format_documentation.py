"""Format retrieved documentation chunks for the system prompt."""

from __future__ import annotations

from collections.abc import Sequence

from mail_gateway.domain.models import DocumentChunk


def format_documentation_context(chunks: Sequence[DocumentChunk]) -> str:
    if not chunks:
        return ""

    lines = ["Релевантные фрагменты документации:"]
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"[{index}] source={chunk.source_path}")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()
