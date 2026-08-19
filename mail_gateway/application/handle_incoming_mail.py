import logging

from mail_gateway.application.clean_email_body import clean_email_body
from mail_gateway.application.format_documentation import (
    format_documentation_context,
    unique_source_names,
)
from mail_gateway.application.render_answer import render_answer
from mail_gateway.domain.models import (
    IncomingMessage,
    Reply,
    turn_from_incoming,
    with_messages,
    with_rag_chunks,
)
from mail_gateway.ports import (
    Assistant,
    ConversationHistoryLoader,
    DocumentRetriever,
    MailSender,
)

logger = logging.getLogger(__name__)

ADMIN_FALLBACK_TEXT = "Обратитесь с этим вопросом к администратору."
INSUFFICIENT_DOCS_TEXT = (
    "В предоставленной документации недостаточно подтверждённых данных "
    "для ответа. Уточните вопрос или обратитесь к администратору."
)
UNVERIFIED_DRAFT_TEXT = (
    "Не удалось подтвердить ответ по документации. "
    "Повторите запрос позже или обратитесь к администратору."
)


def _normalize_email(address: str) -> str:
    cleaned = (address or "").strip().lower()
    if "<" in cleaned and ">" in cleaned:
        start = cleaned.rfind("<")
        end = cleaned.rfind(">")
        if start < end:
            cleaned = cleaned[start + 1 : end].strip()
    return cleaned


class HandleIncomingMail:
    def __init__(
        self,
        assistant: Assistant,
        mail_sender: MailSender,
        history_loader: ConversationHistoryLoader | None = None,
        document_retriever: DocumentRetriever | None = None,
        bot_email: str | None = None,
    ) -> None:
        self._assistant = assistant
        self._mail_sender = mail_sender
        self._history_loader = history_loader
        self._document_retriever = document_retriever
        self._bot_email = _normalize_email(bot_email or "")

    def __call__(self, message: IncomingMessage) -> None:
        logger.info(
            "Handling mail conversation_id=%s from=%s subject=%r",
            message.conversation_id,
            message.from_address,
            message.subject,
        )
        sender = _normalize_email(message.from_address)
        if self._bot_email and sender == self._bot_email:
            logger.info(
                "Ignoring own bot message conversation_id=%s from=%s",
                message.conversation_id,
                message.from_address,
            )
            return

        enriched = self._with_history(message)
        enriched = self._with_documentation(enriched)

        if self._document_retriever is not None and not enriched.rag_chunks:
            logger.info(
                "No documentation chunks conversation_id=%s; refusing ungrounded answer",
                message.conversation_id,
            )
            self._send(enriched, INSUFFICIENT_DOCS_TEXT)
            return

        reply_text = self._assistant.ask(enriched)
        if reply_text is not None:
            reply_text = reply_text.strip()
        if not reply_text:
            reply_text = (
                UNVERIFIED_DRAFT_TEXT
                if enriched.rag_chunks
                else ADMIN_FALLBACK_TEXT
            )
            logger.info(
                "No verified answer for conversation_id=%s; using fallback",
                message.conversation_id,
            )

        self._send(enriched, reply_text)

    def _send(self, message: IncomingMessage, reply_text: str) -> None:
        sources = unique_source_names(message.rag_chunks)
        body = render_answer(reply_text, source_names=sources)
        self._mail_sender.send_reply(
            Reply(
                conversation_id=message.conversation_id,
                in_reply_to_item_id=message.item_id,
                in_reply_to_change_key=message.change_key,
                body=body,
                html=True,
            )
        )
        logger.info(
            "Reply sent conversation_id=%s sources=%s html_chars=%s",
            message.conversation_id,
            sources,
            len(body),
        )

    def _with_history(self, message: IncomingMessage) -> IncomingMessage:
        if self._history_loader is None:
            turns = (turn_from_incoming(message),)
            logger.info(
                "No history loader; sending single-message thread conversation_id=%s",
                message.conversation_id,
            )
            return with_messages(message, turns)

        turns = list(self._history_loader.load(message.conversation_id))
        if not any(turn.item_id == message.item_id for turn in turns):
            turns.append(turn_from_incoming(message))
            logger.info(
                "Current message missing from EWS history; appended conversation_id=%s",
                message.conversation_id,
            )
        if not turns:
            turns = [turn_from_incoming(message)]

        logger.info(
            "Prepared thread for assistant conversation_id=%s turns=%s",
            message.conversation_id,
            len(turns),
        )
        return with_messages(message, turns)

    def _with_documentation(self, message: IncomingMessage) -> IncomingMessage:
        if self._document_retriever is None:
            return message

        query = _retrieval_query(message)
        if not query:
            logger.info(
                "Empty retrieval query conversation_id=%s; skipping RAG",
                message.conversation_id,
            )
            return message

        try:
            chunks = list(self._document_retriever.retrieve(query))
        except Exception:
            logger.exception(
                "Document retrieval failed conversation_id=%s",
                message.conversation_id,
            )
            return message

        context = format_documentation_context(chunks)
        logger.info(
            "RAG context conversation_id=%s chunks=%s sources=%s chars=%s",
            message.conversation_id,
            len(chunks),
            unique_source_names(chunks),
            len(context),
        )
        return with_rag_chunks(message, chunks, rag_context=context or None)


def _retrieval_query(message: IncomingMessage) -> str:
    """Prefer the latest user turn; fall back to the current message body."""
    for turn in reversed(message.messages):
        if turn.role == "user":
            body = clean_email_body(turn.body) or turn.body.strip()
            if body:
                return body
    return clean_email_body(message.body) or message.body.strip()
