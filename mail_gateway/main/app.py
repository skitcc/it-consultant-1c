"""Composition root and reconnect loop."""

from __future__ import annotations

import logging
import time

from mail_gateway.adapters.assistant.http_assistant import HttpAssistant
from mail_gateway.adapters.assistant.stub_assistant import StubAssistant
from mail_gateway.adapters.ews.account import build_account
from mail_gateway.adapters.ews.listener import EwsMailListener
from mail_gateway.adapters.ews.sender import EwsMailSender
from mail_gateway.application.handle_incoming_mail import HandleIncomingMail
from mail_gateway.main.config import Settings
from mail_gateway.ports import Assistant

logger = logging.getLogger(__name__)


def build_assistant(settings: Settings) -> Assistant:
    if settings.assistant_mode == "stub":
        logger.warning("ASSISTANT_MODE=stub — using StubAssistant")
        return StubAssistant()
    return HttpAssistant(
        settings.ai_service_url,
        timeout_sec=settings.ai_timeout_sec,
    )


def run(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    login = settings.ews_username or settings.ews_email
    logger.info(
        "Connecting to EWS server=%s auth=%s login=%s mailbox=%s",
        settings.ews_server,
        settings.ews_auth,
        login,
        settings.ews_email,
    )
    account = build_account(
        server=settings.ews_server,
        email=settings.ews_email,
        password=settings.ews_password,
        auth=settings.ews_auth,
        verify_ssl=settings.ews_verify_ssl,
        username=settings.ews_username,
    )
    listener = EwsMailListener(
        account,
        connection_timeout_minutes=settings.ews_streaming_timeout_minutes,
        ignore_own_mail=settings.ews_ignore_own_mail,
        catchup_minutes=settings.ews_catchup_minutes,
    )
    sender = EwsMailSender(account)
    assistant = build_assistant(settings)
    handle = HandleIncomingMail(assistant=assistant, mail_sender=sender)

    logger.info(
        "Mail gateway started mailbox=%s assistant=%s",
        settings.ews_email,
        settings.assistant_mode,
    )

    while True:
        try:
            for message in listener.listen():
                try:
                    handle(message)
                except Exception:
                    logger.exception(
                        "Failed to handle message conversation_id=%s item_id=%s",
                        message.conversation_id,
                        message.item_id,
                    )
            logger.warning(
                "EWS streaming ended; reconnecting in %s sec",
                settings.reconnect_delay_sec,
            )
        except Exception:
            logger.exception(
                "EWS listener error; reconnecting in %s sec",
                settings.reconnect_delay_sec,
            )
        time.sleep(settings.reconnect_delay_sec)
