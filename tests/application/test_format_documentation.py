from mail_gateway.application.format_documentation import format_documentation_context
from mail_gateway.domain.models import DocumentChunk


def test_format_documentation_context_empty() -> None:
    assert format_documentation_context([]) == ""


def test_format_documentation_context_lists_sources() -> None:
    chunks = [
        DocumentChunk(text="first chunk", source_path="a.md", chunk_index=0, score=0.9, headings=("Установка",)),
        DocumentChunk(text="second chunk", source_path="b.txt", chunk_index=1),
    ]
    rendered = format_documentation_context(chunks)
    assert "Релевантные фрагменты документации:" in rendered
    assert "[1] source=a.md chunk=0 headings=Установка score=0.9000" in rendered
    assert "first chunk" in rendered
    assert "[2] source=b.txt chunk=1" in rendered
    assert "second chunk" in rendered
