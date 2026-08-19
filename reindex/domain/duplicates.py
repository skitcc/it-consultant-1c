"""Pick one filesystem path when several files share the same content hash."""

from __future__ import annotations


def canonical_paths(path_hashes: dict[str, str]) -> dict[str, str]:
    """Map each content hash to the lexicographically first relative path."""
    by_hash: dict[str, list[str]] = {}
    for relative, content_hash in path_hashes.items():
        by_hash.setdefault(content_hash, []).append(relative)
    return {content_hash: sorted(paths)[0] for content_hash, paths in by_hash.items()}


def next_path_for_hash(
    path_hashes: dict[str, str],
    content_hash: str,
    *,
    exclude: set[str] | None = None,
) -> str | None:
    """Return the next remaining path for ``content_hash``, if any."""
    skipped = exclude or set()
    peers = sorted(
        relative
        for relative, file_hash in path_hashes.items()
        if file_hash == content_hash and relative not in skipped
    )
    return peers[0] if peers else None
