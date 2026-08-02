from common.chunking import chunk_text


def test_chunk_text_short_returns_single() -> None:
    assert chunk_text("hello", chunk_size=100, overlap=10) == ["hello"]


def test_chunk_text_splits_with_overlap() -> None:
    text = "a" * 50
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 3
    assert chunks[0] == "a" * 20
    # Next window starts 15 chars in (20 - 5 overlap).
    assert chunks[1].startswith("a" * 15) or chunks[1] == "a" * 20


def test_chunk_text_empty() -> None:
    assert chunk_text("   ") == []
