"""Shared prompts for knowledge-backed IT consulting."""

IT_CONSULTANT_SYSTEM_PROMPT = """\
Ты — внутренний ИТ-консультант компании, специализирующийся на 1С и корпоративных \
системах. Отвечай по-русски, точно и практично. Используй прежде всего приведённый \
контекст из базы знаний. Не выдумывай факты, которых нет в контексте. Если данных \
недостаточно, прямо сообщи об этом и задай уточняющий вопрос. Сохраняй важные \
предупреждения, ограничения и последовательность действий из документации.
"""


def format_knowledge_context(chunks: object) -> str:
    from knowledge.core.domain import DocumentChunk

    if not isinstance(chunks, (list, tuple)):
        chunks = list(chunks)  # type: ignore[arg-type]
    if not chunks:
        return "Контекст базы знаний отсутствует."
    sections: list[str] = ["Контекст базы знаний:"]
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, DocumentChunk):
            raise TypeError("Context must contain DocumentChunk values")
        location = chunk.source_path or chunk.filename or chunk.document_id
        heading = " > ".join(chunk.headings)
        label = f"[{index}] {location}"
        if heading:
            label += f" — {heading}"
        sections.extend((label, chunk.text))
    return "\n\n".join(sections)
