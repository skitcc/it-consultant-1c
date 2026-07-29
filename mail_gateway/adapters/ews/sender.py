"""Send replies through Exchange EWS."""

from __future__ import annotations

import logging

from exchangelib import Account, Message

from mail_gateway.domain.models import Reply
from mail_gateway.ports import MailSender

logger = logging.getLogger(__name__)


class EwsMailSender(MailSender):
    def __init__(self, account: Account) -> None:
        self._account = account

    def send_reply(self, reply: Reply) -> None:
        item = self._account.inbox.get(
            id=reply.in_reply_to_item_id,
            changekey=reply.in_reply_to_change_key,
        )
        if not isinstance(item, Message):
            raise TypeError(
                f"Cannot reply to non-message item {reply.in_reply_to_item_id!r}"
            )

        subject = item.subject or ""
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        item.reply(subject=subject, body=reply.body)
        logger.info(
            "EWS reply sent conversation_id=%s in_reply_to=%s",
            reply.conversation_id,
            reply.in_reply_to_item_id,
        )
