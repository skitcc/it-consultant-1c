"""Format retrieved documentation chunks for the system prompt."""

from __future__ import annotations

from collections.abc import Sequence

from mail_gateway.domain.models import DocumentChunk


def format_documentation_context(chunks: Sequence[DocumentChunk]) -> str:
    if not chunks:
        return ""

    lines = ["Релевантные фрагменты документации:"]
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{index}] source={chunk.source_path} chunk={chunk.chunk_index}"
        if chunk.headings:
            header = f"{header} headings={' / '.join(chunk.headings)}"
        if chunk.score is not None:
            header = f"{header} score={chunk.score:.4f}"
        lines.append(header)
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()
