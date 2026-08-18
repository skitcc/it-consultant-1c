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
    """Include nearby chunks from ``pool`` for each selected seed.

    When a seed has ``headings``, neighbors are other chunks from the same
    file with the same heading path (same section), not a different H2 that
    happens to sit at ``chunk_index ± 1``. The window still caps how many
    siblings to keep around the seed.

    Without headings (plain ``.txt`` / ``.log``), fall back to ``chunk_index ± window``.

    Result is sorted by ``(source_path, chunk_index)`` for coherent prompt context.
    """
    if window < 0:
        raise ValueError("window must be non-negative")
    if not selected:
        return []

    pool_by_key = {
        (chunk.source_path, chunk.chunk_index): chunk for chunk in pool
    }
    for chunk in selected:
        pool_by_key[(chunk.source_path, chunk.chunk_index)] = chunk

    wanted: set[tuple[str, int]] = set()
    for chunk in selected:
        if chunk.headings:
            siblings = [
                item
                for item in pool_by_key.values()
                if item.source_path == chunk.source_path
                and item.headings == chunk.headings
            ]
            siblings.sort(key=lambda item: item.chunk_index)
            indexes = [item.chunk_index for item in siblings]
            try:
                position = indexes.index(chunk.chunk_index)
            except ValueError:
                wanted.add((chunk.source_path, chunk.chunk_index))
                continue
            start = max(0, position - window)
            stop = min(len(siblings), position + window + 1)
            for sibling in siblings[start:stop]:
                wanted.add((sibling.source_path, sibling.chunk_index))
        else:
            for delta in range(-window, window + 1):
                index = chunk.chunk_index + delta
                if index >= 0:
                    wanted.add((chunk.source_path, index))

    return [
        pool_by_key[key]
        for key in sorted(wanted, key=lambda item: (item[0], item[1]))
        if key in pool_by_key
    ]
