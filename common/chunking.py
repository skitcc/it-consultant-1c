"""Split documents into overlapping text chunks for embedding."""

from __future__ import annotations


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[str]:
    """Split ``text`` into character windows with overlap.

    ``chunk_size`` / ``overlap`` are in characters (approx. tokens × 3–4).
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = end - overlap
    return chunks
