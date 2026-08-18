"""Answer a question using retrieved knowledge and conversation history."""

from __future__ import annotations

from collections.abc import Sequence

from knowledge.core.domain import ConversationMessage
from knowledge.core.ports import ChatModel
from knowledge.core.use_cases.prompts import (
    IT_CONSULTANT_SYSTEM_PROMPT,
    format_knowledge_context,
)
from knowledge.core.use_cases.retrieve_knowledge import RetrieveKnowledge


class AnswerQuestion:
    def __init__(
        self,
        *,
        retriever: RetrieveKnowledge,
        chat_model: ChatModel,
        top_k: int = 8,
        neighbor_window: int = 1,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")
        self._retriever = retriever
        self._chat_model = chat_model
        self._top_k = top_k
        self._neighbor_window = neighbor_window

    def execute(
        self,
        question: str,
        *,
        history: Sequence[ConversationMessage] = (),
        knowledge_id: str = "main",
    ) -> str:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        chunks = self._retriever.execute(
            cleaned,
            knowledge_id=knowledge_id,
            top_k=self._top_k,
            neighbor_window=self._neighbor_window,
        )
        system = ConversationMessage(
            role="system",
            content=f"{IT_CONSULTANT_SYSTEM_PROMPT.strip()}\n\n{format_knowledge_context(chunks)}",
        )
        conversation = [
            message
            for message in history
            if message.role in {"user", "assistant"} and message.content.strip()
        ]
        messages = [system, *conversation, ConversationMessage(role="user", content=cleaned)]
        return self._chat_model.complete(messages).strip()

    __call__ = execute
