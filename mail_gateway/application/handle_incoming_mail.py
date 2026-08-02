import logging

from mail_gateway.application.clean_email_body import clean_email_body
from mail_gateway.application.format_documentation import format_documentation_context
from mail_gateway.domain.models import (
    IncomingMessage,
    Reply,
    turn_from_incoming,
    with_messages,
    with_rag_context,
)
from mail_gateway.ports import (
    Assistant,
    ConversationHistoryLoader,
    DocumentRetriever,
    MailSender,
)

logger = logging.getLogger(__name__)

ADMIN_FALLBACK_TEXT = "Обратитесь с этим вопросом к администратору."


class HandleIncomingMail:
    def __init__(
        self,
        assistant: Assistant,
        mail_sender: MailSender,
        history_loader: ConversationHistoryLoader | None = None,
        document_retriever: DocumentRetriever | None = None,
    ) -> None:
        self._assistant = assistant
        self._mail_sender = mail_sender
        self._history_loader = history_loader
        self._document_retriever = document_retriever

    def __call__(self, message: IncomingMessage) -> None:
        logger.info(
            "Handling mail conversation_id=%s from=%s subject=%r",
            message.conversation_id,
            message.from_address,
            message.subject,
        )
        enriched = self._with_history(message)
        enriched = self._with_documentation(enriched)
        reply_text = self._assistant.ask(enriched)
        if reply_text is not None:
            reply_text = reply_text.strip()
        if not reply_text:
            reply_text = ADMIN_FALLBACK_TEXT
            logger.info(
                "No relevant answer for conversation_id=%s; using admin fallback",
                message.conversation_id,
            )

        self._mail_sender.send_reply(
            Reply(
                conversation_id=message.conversation_id,
                in_reply_to_item_id=message.item_id,
                in_reply_to_change_key=message.change_key,
                body=reply_text,
            )
        )
        logger.info("Reply sent for conversation_id=%s", message.conversation_id)

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
            chunks = self._document_retriever.retrieve(query)
        except Exception:
            logger.exception(
                "Document retrieval failed conversation_id=%s",
                message.conversation_id,
            )
            return message

        context = format_documentation_context(chunks)
        logger.info(
            "RAG context conversation_id=%s chunks=%s chars=%s",
            message.conversation_id,
            len(chunks),
            len(context),
        )
        return with_rag_context(message, context or None)


def _retrieval_query(message: IncomingMessage) -> str:
    """Prefer the latest user turn; fall back to the current message body."""
    for turn in reversed(message.messages):
        if turn.role == "user":
            body = clean_email_body(turn.body) or turn.body.strip()
            if body:
                return body
    return clean_email_body(message.body) or message.body.strip()
