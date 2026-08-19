from reindex.domain.duplicates import canonical_paths, next_path_for_hash


def test_canonical_paths_picks_sorted_first() -> None:
    hashes = {
        "b.md": "same",
        "a.md": "same",
        "c.md": "other",
    }
    assert canonical_paths(hashes) == {"same": "a.md", "other": "c.md"}


def test_next_path_for_hash_skips_excluded() -> None:
    hashes = {"a.md": "h", "b.md": "h", "c.md": "other"}
    assert next_path_for_hash(hashes, "h", exclude={"a.md"}) == "b.md"
    assert next_path_for_hash(hashes, "missing") is None
