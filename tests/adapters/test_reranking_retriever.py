from mail_gateway.adapters.rag.neighbors import expand_neighbor_chunks
from mail_gateway.adapters.rag.reranking_retriever import RerankingRetriever
from mail_gateway.domain.models import DocumentChunk


def _chunk(
    path: str,
    index: int,
    text: str,
    score: float | None = None,
    headings: tuple[str, ...] = (),
) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        source_path=path,
        chunk_index=index,
        score=score,
        headings=headings,
    )


def test_expand_neighbor_chunks_includes_window() -> None:
    selected = [_chunk("faq.md", 2, "center", 0.9)]
    pool = [
        _chunk("faq.md", 1, "left"),
        _chunk("faq.md", 2, "center", 0.9),
        _chunk("faq.md", 3, "right"),
        _chunk("other.md", 0, "noise"),
    ]
    expanded = expand_neighbor_chunks(selected, pool, window=1)
    assert [c.chunk_index for c in expanded] == [1, 2, 3]
    assert [c.text for c in expanded] == ["left", "center", "right"]


def test_expand_neighbor_chunks_stays_in_same_section() -> None:
    selected = [_chunk("faq.md", 2, "center", 0.9, headings=("Принтеры", "Чеки"))]
    pool = [
        _chunk("faq.md", 1, "same section prev", headings=("Принтеры", "Чеки")),
        _chunk("faq.md", 2, "center", 0.9, headings=("Принтеры", "Чеки")),
        _chunk("faq.md", 3, "next section", headings=("Сеть", "VPN")),
    ]
    expanded = expand_neighbor_chunks(selected, pool, window=1)
    assert [c.text for c in expanded] == ["same section prev", "center"]


def test_expand_neighbor_chunks_caps_section_siblings() -> None:
    selected = [_chunk("guide.md", 3, "c", headings=("Установка",))]
    pool = [
        _chunk("guide.md", 1, "a", headings=("Установка",)),
        _chunk("guide.md", 2, "b", headings=("Установка",)),
        _chunk("guide.md", 3, "c", headings=("Установка",)),
        _chunk("guide.md", 4, "d", headings=("Установка",)),
        _chunk("guide.md", 5, "e", headings=("Установка",)),
    ]
    expanded = expand_neighbor_chunks(selected, pool, window=1)
    assert [c.text for c in expanded] == ["b", "c", "d"]


def test_reranking_retriever_top_k_and_neighbors() -> None:
    candidates = [
        _chunk("faq.md", 0, "labels", 0.55),
        _chunk("faq.md", 1, "receipt printer", 0.50),
        _chunk("faq.md", 2, "receipt steps", 0.40),
        _chunk("other.md", 0, "vpn", 0.30),
    ]

    class FakeBase:
        def retrieve(self, query: str) -> list[DocumentChunk]:
            assert "принтер" in query
            return list(candidates)

        def load_neighbors(
            self,
            seeds: list[DocumentChunk],
            *,
            window: int = 1,
        ) -> list[DocumentChunk]:
            assert window == 1
            assert seeds[0].text == "receipt printer"
            return [
                _chunk("faq.md", 0, "labels"),
                _chunk("faq.md", 1, "receipt printer"),
                _chunk("faq.md", 2, "receipt steps"),
            ]

    class FakeReranker:
        def rerank(self, query: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
            del query
            # Put receipt chapter first regardless of vector score.
            order = {"receipt printer": 0, "receipt steps": 1, "labels": 2, "vpn": 3}
            return sorted(chunks, key=lambda c: order[c.text])

    retriever = RerankingRetriever(
        base=FakeBase(),
        reranker=FakeReranker(),
        top_k=1,
        neighbor_window=1,
    )
    result = retriever.retrieve("принтер чеков не печатает")
    assert [c.text for c in result] == ["labels", "receipt printer", "receipt steps"]
