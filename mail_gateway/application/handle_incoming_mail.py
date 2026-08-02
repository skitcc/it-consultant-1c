import logging

from mail_gateway.domain.models import (
    IncomingMessage,
    Reply,
    turn_from_incoming,
    with_messages,
)
from mail_gateway.ports import Assistant, ConversationHistoryLoader, MailSender

logger = logging.getLogger(__name__)

ADMIN_FALLBACK_TEXT = "Обратитесь с этим вопросом к администратору."


class HandleIncomingMail:
    def __init__(
        self,
        assistant: Assistant,
        mail_sender: MailSender,
        history_loader: ConversationHistoryLoader | None = None,
    ) -> None:
        self._assistant = assistant
        self._mail_sender = mail_sender
        self._history_loader = history_loader

    def __call__(self, message: IncomingMessage) -> None:
        logger.info(
            "Handling mail conversation_id=%s from=%s subject=%r",
            message.conversation_id,
            message.from_address,
            message.subject,
        )
        enriched = self._with_history(message)
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
