"""Send replies through Exchange EWS."""

from __future__ import annotations

import logging
import time

from exchangelib import Account, HTMLBody, Message

from mail_gateway.domain.models import Reply
from mail_gateway.ports import MailSender

logger = logging.getLogger(__name__)


class EwsMailSender(MailSender):
    def __init__(self, account: Account) -> None:
        self._account = account

    def send_reply(self, reply: Reply) -> None:
        started_at = time.perf_counter()
        logger.info(
            "Loading source message for reply item_id=%s pool_size=%s/%s",
            reply.in_reply_to_item_id,
            self._account.protocol.session_pool_size,
            self._account.protocol._session_pool_maxsize,
        )
        kwargs: dict[str, object] = {"id": reply.in_reply_to_item_id}
        if reply.in_reply_to_change_key:
            kwargs["changekey"] = reply.in_reply_to_change_key
        item = self._account.inbox.get(**kwargs)
        logger.info(
            "Source message loaded for reply item_id=%s elapsed=%.3fs type=%s",
            reply.in_reply_to_item_id,
            time.perf_counter() - started_at,
            type(item).__name__,
        )
        if not isinstance(item, Message):
            raise TypeError(
                f"Cannot reply to non-message item {reply.in_reply_to_item_id!r}"
            )

        subject = item.subject or ""
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        logger.info(
            "Sending EWS reply conversation_id=%s in_reply_to=%s html=%s",
            reply.conversation_id,
            reply.in_reply_to_item_id,
            reply.html,
        )
        # create_reply + explicit body avoids dumping the whole quoted thread
        # into Sent Items (which later pollutes model history).
        body = HTMLBody(reply.body) if reply.html else reply.body
        draft = item.create_reply(subject=subject, body=body)
        draft.body = body
        draft.send()
        logger.info(
            "EWS reply sent conversation_id=%s in_reply_to=%s elapsed=%.3fs",
            reply.conversation_id,
            reply.in_reply_to_item_id,
            time.perf_counter() - started_at,
        )
