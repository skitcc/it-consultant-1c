from mail_gateway.application.format_documentation import (
    document_display_name,
    format_documentation_context,
    unique_source_names,
)
from mail_gateway.domain.models import DocumentChunk


def test_format_documentation_context_empty() -> None:
    assert format_documentation_context([]) == ""


def test_format_documentation_context_lists_sources_without_numeric_refs() -> None:
    chunks = [
        DocumentChunk(
            text="first chunk",
            source_path="docs/a.md",
            chunk_index=0,
            score=0.9,
            headings=("Установка",),
        ),
        DocumentChunk(text="second chunk", source_path="b.txt", chunk_index=1),
    ]
    rendered = format_documentation_context(chunks)
    assert "Релевантные фрагменты документации:" in rendered
    assert "Документ: a.md. Раздел: Установка" in rendered
    assert "first chunk" in rendered
    assert "Документ: b.txt" in rendered
    assert "second chunk" in rendered
    assert "[1]" not in rendered
    assert "source=" not in rendered
    assert "score=" not in rendered


def test_unique_source_names_dedupes_basenames() -> None:
    chunks = [
        DocumentChunk(text="a", source_path="hr/grades.pdf", chunk_index=0),
        DocumentChunk(text="b", source_path="hr/grades.pdf", chunk_index=1),
        DocumentChunk(text="c", source_path="it/vpn.pdf", chunk_index=0),
    ]
    assert unique_source_names(chunks) == ["grades.pdf", "vpn.pdf"]
    assert document_display_name("folder/Outlook (v5).pdf") == "Outlook (v5).pdf"
