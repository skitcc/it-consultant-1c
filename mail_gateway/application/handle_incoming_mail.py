import logging

from mail_gateway.domain.models import IncomingMessage, Reply
from mail_gateway.ports import Assistant, MailSender

logger = logging.getLogger(__name__)

ADMIN_FALLBACK_TEXT = "Обратитесь с этим вопросом к администратору."


class HandleIncomingMail:
    def __init__(self, assistant: Assistant, mail_sender: MailSender) -> None:
        self._assistant = assistant
        self._mail_sender = mail_sender

    def __call__(self, message: IncomingMessage) -> None:
        logger.info(
            "Handling mail conversation_id=%s from=%s subject=%r",
            message.conversation_id,
            message.from_address,
            message.subject,
        )
        reply_text = self._assistant.ask(message)
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
