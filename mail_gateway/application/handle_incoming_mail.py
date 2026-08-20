import logging

from collections.abc import Sequence

from common.timing import begin_request, end_request, span

from mail_gateway.application.clean_email_body import clean_email_body
from mail_gateway.application.format_documentation import (
    format_documentation_context,
    unique_source_names,
)
from mail_gateway.application.render_answer import UnsafeAnswerError, render_answer
from mail_gateway.domain.models import (
    ConversationTurn,
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
TECHNICAL_FAILURE_TEXT = (
    "Сервис временно не смог обработать запрос. "
    "Повторите запрос позже или обратитесь к администратору."
)


class _DocumentRetrievalError(RuntimeError):
    pass


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
        admin_email: str | None = None,
    ) -> None:
        self._assistant = assistant
        self._mail_sender = mail_sender
        self._history_loader = history_loader
        self._document_retriever = document_retriever
        self._bot_email = _normalize_email(bot_email or "")
        self._admin_email = _normalize_email(admin_email or "") or None

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

        timer = begin_request(
            conversation_id=message.conversation_id,
            item_id=message.item_id,
        )
        delivered = False
        error: str | None = None
        ahead = 0
        enriched = message
        try:
            with span("history"):
                enriched = self._with_history(message)
            ahead = pending_user_requests_before(enriched.messages, message.item_id)
            logger.log(
                logging.WARNING if ahead else logging.INFO,
                "Pending user requests before this one conversation_id=%s ahead=%s",
                message.conversation_id,
                ahead,
            )
            try:
                with span("rag"):
                    enriched = self._with_documentation(enriched)
            except _DocumentRetrievalError as exc:
                error = _error_text(exc)
                delivered, send_error = self._try_send(enriched, TECHNICAL_FAILURE_TEXT)
                error = error or send_error
            else:
                if self._document_retriever is not None and not enriched.rag_chunks:
                    logger.info(
                        "No documentation chunks conversation_id=%s; "
                        "refusing ungrounded answer",
                        message.conversation_id,
                    )
                    delivered, send_error = self._try_send(
                        enriched, INSUFFICIENT_DOCS_TEXT
                    )
                    error = send_error
                else:
                    try:
                        reply_text = self._assistant.ask(enriched)
                    except Exception as exc:
                        logger.exception(
                            "Assistant failed conversation_id=%s; using technical fallback",
                            message.conversation_id,
                        )
                        error = _error_text(exc)
                        delivered, send_error = self._try_send(
                            enriched, TECHNICAL_FAILURE_TEXT
                        )
                        error = error or send_error
                    else:
                        if reply_text is not None:
                            reply_text = reply_text.strip()
                        if not reply_text:
                            error = "empty_answer"
                            reply_text = (
                                UNVERIFIED_DRAFT_TEXT
                                if enriched.rag_chunks
                                else ADMIN_FALLBACK_TEXT
                            )
                            logger.info(
                                "No verified answer for conversation_id=%s; using fallback",
                                message.conversation_id,
                            )
                        delivered, send_error = self._try_send(enriched, reply_text)
                        error = error or send_error
        except Exception as exc:
            logger.exception(
                "Failed to handle message conversation_id=%s item_id=%s",
                message.conversation_id,
                message.item_id,
            )
            error = error or _error_text(exc)
            if not delivered:
                delivered, send_error = self._try_send(enriched, TECHNICAL_FAILURE_TEXT)
                error = error or send_error

        try:
            if error or not delivered or ahead:
                with span("admin_notify"):
                    self._notify_admin(
                        message=enriched,
                        error=error,
                        delivered=delivered,
                        ahead=ahead,
                    )
        finally:
            end_request(timer)

    def _try_send(
        self,
        message: IncomingMessage,
        reply_text: str,
    ) -> tuple[bool, str | None]:
        with span("ews_reply"):
            return self._send_reply(message, reply_text)

    def _send_reply(
        self,
        message: IncomingMessage,
        reply_text: str,
    ) -> tuple[bool, str | None]:
        sources = unique_source_names(message.rag_chunks)
        send_error: str | None = None
        try:
            body = render_answer(reply_text, source_names=sources)
        except UnsafeAnswerError as exc:
            logger.exception(
                "Unsafe internal reasoning blocked conversation_id=%s",
                message.conversation_id,
            )
            send_error = _error_text(exc)
            body = render_answer(UNVERIFIED_DRAFT_TEXT, source_names=sources)
        try:
            self._mail_sender.send_reply(
                Reply(
                    conversation_id=message.conversation_id,
                    in_reply_to_item_id=message.item_id,
                    in_reply_to_change_key=message.change_key,
                    body=body,
                    html=True,
                )
            )
        except Exception as exc:
            logger.exception(
                "Reply not delivered conversation_id=%s item_id=%s",
                message.conversation_id,
                message.item_id,
            )
            return False, send_error or _error_text(exc)
        logger.info(
            "Reply sent conversation_id=%s sources=%s html_chars=%s",
            message.conversation_id,
            sources,
            len(body),
        )
        return True, send_error

    def _notify_admin(
        self,
        *,
        message: IncomingMessage,
        error: str | None,
        delivered: bool,
        ahead: int,
    ) -> None:
        body = format_admin_alert(
            conversation_id=message.conversation_id,
            item_id=message.item_id,
            from_address=message.from_address,
            subject=message.subject,
            question=message.body,
            error=error,
            delivered=delivered,
            ahead=ahead,
        )
        subject = _admin_subject(error=error, delivered=delivered, ahead=ahead)
        logger.warning(
            "Admin alert conversation_id=%s item_id=%s delivered=%s ahead=%s "
            "error=%s admin=%s subject=%s",
            message.conversation_id,
            message.item_id,
            delivered,
            ahead,
            error,
            self._admin_email or "-",
            subject,
        )
        logger.warning("Admin alert body conversation_id=%s\n%s", message.conversation_id, body)
        if not self._admin_email:
            logger.warning(
                "ADMIN_EMAIL is not set; admin mail skipped conversation_id=%s",
                message.conversation_id,
            )
            return
        try:
            self._mail_sender.send_mail(to=self._admin_email, subject=subject, body=body)
        except Exception:
            logger.exception(
                "Failed to notify admin conversation_id=%s to=%s",
                message.conversation_id,
                self._admin_email,
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
        except Exception as exc:
            logger.exception(
                "Document retrieval failed conversation_id=%s",
                message.conversation_id,
            )
            raise _DocumentRetrievalError from exc

        context = format_documentation_context(chunks)
        logger.info(
            "RAG context conversation_id=%s chunks=%s sources=%s chars=%s",
            message.conversation_id,
            len(chunks),
            unique_source_names(chunks),
            len(context),
        )
        return with_rag_chunks(message, chunks, rag_context=context or None)


def pending_user_requests_before(
    turns: Sequence[ConversationTurn],
    current_item_id: str,
) -> int:
    """Count unanswered user turns after the last bot reply, excluding current."""
    pending = 0
    for turn in turns:
        if turn.role == "assistant":
            pending = 0
            continue
        if turn.role != "user":
            continue
        if turn.item_id and turn.item_id == current_item_id:
            continue
        pending += 1
    return pending


def format_admin_alert(
    *,
    conversation_id: str,
    item_id: str,
    from_address: str,
    subject: str,
    question: str,
    error: str | None,
    delivered: bool,
    ahead: int,
) -> str:
    snippet = (question or "").strip()
    if len(snippet) > 500:
        snippet = snippet[:500] + "…"
    return "\n".join(
        [
            f"conversation_id: {conversation_id}",
            f"item_id: {item_id}",
            f"от: {from_address}",
            f"тема: {subject}",
            "",
            f"ошибка: {error or 'нет'}",
            f"ответ пользователю: {'отправлен' if delivered else 'не отправлен'}",
            f"запросов пользователя перед этим: {ahead}",
            "",
            "вопрос:",
            snippet or "(пусто)",
        ]
    )


def _admin_subject(*, error: str | None, delivered: bool, ahead: int) -> str:
    if not delivered:
        return "IT-консультант: ответ не отправлен пользователю"
    if error:
        return "IT-консультант: ошибка обработки запроса"
    if ahead:
        return f"IT-консультант: {_requests_word(ahead)} пользователя перед этим"
    return "IT-консультант: предупреждение"


def _requests_word(count: int) -> str:
    mod10 = count % 10
    mod100 = count % 100
    if mod10 == 1 and mod100 != 11:
        word = "запрос"
    elif mod10 in {2, 3, 4} and mod100 not in {12, 13, 14}:
        word = "запроса"
    else:
        word = "запросов"
    return f"{count} {word}"


def _error_text(exc: BaseException) -> str:
    cause = exc.__cause__ if isinstance(exc, _DocumentRetrievalError) else exc
    if cause is None:
        cause = exc
    return f"{type(cause).__name__}: {cause}"


def _retrieval_query(message: IncomingMessage) -> str:
    """Prefer the latest user turn; fall back to the current message body."""
    for turn in reversed(message.messages):
        if turn.role == "user":
            body = clean_email_body(turn.body) or turn.body.strip()
            if body:
                return body
    return clean_email_body(message.body) or message.body.strip()
