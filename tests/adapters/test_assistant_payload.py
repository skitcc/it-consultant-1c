from mail_gateway.adapters.assistant.payload import (
    DEFAULT_SYSTEM_PROMPT,
    build_assistant_payload,
)
from mail_gateway.domain.models import ConversationTurn, IncomingMessage, with_messages


def test_build_assistant_payload_is_minimal_and_cleaned() -> None:
    message = IncomingMessage(
        conversation_id="conv-1",
        item_id="item-2",
        change_key="ck",
        from_address="user@company.ru",
        subject="Help",
        body="second\nSent from my iPhone\n\n> quoted",
    )
    message = with_messages(
        message,
        [
            ConversationTurn(
                role="user",
                body="first\nSent from my iPhone",
                from_address="user@company.ru",
                subject="Help",
                item_id="item-1",
            ),
            ConversationTurn(
                role="assistant",
                body=(
                    "answer\n\nconversation_id=x\nsubject=y\n"
                    "________________________________________\nFrom: u"
                ),
                from_address="bot@company.ru",
                subject="Re: Help",
                item_id="item-bot",
            ),
            ConversationTurn(
                role="user",
                body="second\nSent from my iPhone\n\n> quoted",
                from_address="user@company.ru",
                subject="Help",
                item_id="item-2",
            ),
        ],
    )

    payload = build_assistant_payload(message)

    assert payload["conversation_id"] == "conv-1"
    assert payload["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    assert payload["messages"] == [
        {"role": "user", "body": "first"},
        {"role": "assistant", "body": "answer"},
        {"role": "user", "body": "second"},
    ]


def test_custom_system_prompt_in_payload() -> None:
    message = IncomingMessage(
        conversation_id="c",
        item_id="i",
        change_key="k",
        from_address="u@x.ru",
        subject="s",
        body="hi",
    )
    payload = build_assistant_payload(message, system_prompt="Отвечай кратко.")
    assert payload["system_prompt"] == "Отвечай кратко."
