"""Expand selected chunks with neighboring chunks from the same source."""

from __future__ import annotations

from collections.abc import Sequence

from mail_gateway.domain.models import DocumentChunk


def expand_neighbor_chunks(
    selected: Sequence[DocumentChunk],
    pool: Sequence[DocumentChunk],
    *,
    window: int = 1,
) -> list[DocumentChunk]:
    """Include pool chunks with the same source and nearby chunk_index.

    Result is sorted by (source_path, chunk_index) for coherent prompt context.
    """
    if window < 0:
        raise ValueError("window must be non-negative")
    if not selected:
        return []

    pool_by_key = {
        (chunk.source_path, chunk.chunk_index): chunk for chunk in pool
    }
    wanted: set[tuple[str, int]] = set()
    for chunk in selected:
        for delta in range(-window, window + 1):
            wanted.add((chunk.source_path, chunk.chunk_index + delta))

    expanded = [
        pool_by_key[key]
        for key in sorted(wanted, key=lambda item: (item[0], item[1]))
        if key in pool_by_key
    ]
    return expanded
