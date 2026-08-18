from __future__ import annotations

from dataclasses import replace

from knowledge.core.domain import ConversationMessage, DocumentChunk
from knowledge.core.use_cases import AnswerQuestion, RetrieveKnowledge


class Embedder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed(self, text: str):
        self.events.append(f"embed:{text}")
        return [0.1, 0.2]

    def embed_documents(self, texts: list[str]):
        return [[0.1, 0.2] for _ in texts]


class Index:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.candidates = [
            DocumentChunk(
                text=f"candidate-{index}",
                source_path="manual.md",
                filename="manual.md",
                document_id="doc",
                chunk_index=index,
                score=float(index),
                content_hash="v1",
            )
            for index in range(3)
        ]

    def search(self, vector, **kwargs):
        self.events.append(f"search:{kwargs['knowledge_id']}:{kwargs['limit']}")
        assert vector == [0.1, 0.2]
        return self.candidates

    def load_neighbors(self, seeds, **kwargs):
        self.events.append(
            "neighbors:" + ",".join(str(seed.chunk_index) for seed in seeds)
        )
        return [
            replace(self.candidates[0], text="neighbor context"),
            *seeds,
        ]


class Reranker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def rerank(self, query, chunks):
        self.events.append(f"rerank:{query}:{len(chunks)}")
        return list(reversed(chunks))


class Chat:
    def __init__(self) -> None:
        self.messages = []

    def complete(self, messages):
        self.messages = list(messages)
        return "  Готовый ответ  "


def test_retrieval_pipeline_selects_top_k_before_loading_neighbors():
    events: list[str] = []
    retriever = RetrieveKnowledge(
        embedder=Embedder(events),
        vector_index=Index(events),
        reranker=Reranker(events),
        candidate_limit=12,
    )

    result = retriever.execute(
        "  Как настроить? ",
        knowledge_id="team",
        top_k=2,
        neighbor_window=1,
    )

    assert events == [
        "embed:Как настроить?",
        "search:team:12",
        "rerank:Как настроить?:3",
        "neighbors:2,1",
    ]
    assert [chunk.text for chunk in result] == [
        "neighbor context",
        "candidate-2",
        "candidate-1",
    ]


def test_answer_uses_shared_russian_prompt_context_and_history():
    events: list[str] = []
    retriever = RetrieveKnowledge(
        embedder=Embedder(events),
        vector_index=Index(events),
        reranker=Reranker(events),
    )
    chat = Chat()
    answer = AnswerQuestion(
        retriever=retriever,
        chat_model=chat,
        top_k=1,
        neighbor_window=0,
    )

    result = answer.execute(
        "Что делать?",
        history=(ConversationMessage(role="assistant", content="Предыдущий ответ"),),
    )

    assert result == "Готовый ответ"
    assert [message.role for message in chat.messages] == [
        "system",
        "assistant",
        "user",
    ]
    assert "ИТ-консультант" in chat.messages[0].content
    assert "Контекст базы знаний" in chat.messages[0].content
    assert "neighbor context" in chat.messages[0].content
