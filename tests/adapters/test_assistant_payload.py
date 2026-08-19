from mail_gateway.adapters.assistant.payload import (
    DEFAULT_SYSTEM_PROMPT,
    OUTPUT_CONTRACT,
    VERIFIER_SYSTEM_PROMPT,
    build_assistant_payload,
    build_verifier_payload,
)
from mail_gateway.domain.models import (
    ConversationTurn,
    IncomingMessage,
    with_messages,
    with_rag_context,
)


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


def test_custom_system_prompt_keeps_output_contract() -> None:
    message = IncomingMessage(
        conversation_id="c",
        item_id="i",
        change_key="k",
        from_address="u@x.ru",
        subject="s",
        body="hi",
    )
    payload = build_assistant_payload(message, system_prompt="Отвечай кратко.")
    assert payload["system_prompt"].startswith("Отвечай кратко.")
    assert OUTPUT_CONTRACT in payload["system_prompt"]
    assert "Запрещены маркеры" in payload["system_prompt"]


def test_rag_context_appended_to_system_prompt() -> None:
    message = IncomingMessage(
        conversation_id="c",
        item_id="i",
        change_key="k",
        from_address="u@x.ru",
        subject="s",
        body="hi",
    )
    message = with_rag_context(
        message,
        "Релевантные фрагменты документации:\nДокумент: a.md\ntext",
    )
    payload = build_assistant_payload(message, system_prompt="Base.")
    assert payload["system_prompt"].startswith("Base.")
    assert "source=a.md" not in payload["system_prompt"]
    assert "Документ: a.md" in payload["system_prompt"]
    assert "<documentation_context>" in payload["system_prompt"]
    assert "</documentation_context>" in payload["system_prompt"]


def test_default_system_prompt_requires_grounded_formal_answers() -> None:
    assert "Не выдумывай" in DEFAULT_SYSTEM_PROMPT
    assert "thinking" in DEFAULT_SYSTEM_PROMPT
    assert "без Markdown" in DEFAULT_SYSTEM_PROMPT
    assert "K0 и K1" in DEFAULT_SYSTEM_PROMPT
    assert OUTPUT_CONTRACT in DEFAULT_SYSTEM_PROMPT


def test_verifier_payload_includes_draft_and_same_chunks() -> None:
    payload = build_verifier_payload(
        conversation_id="c1",
        draft="<p>K0-2: PM готов покупать по K1</p>",
        rag_context="Документ: grades.pdf\nK0-2: PM готов покупать по K0.",
    )
    assert payload["conversation_id"] == "c1"
    assert payload["system_prompt"].startswith(VERIFIER_SYSTEM_PROMPT[:40])
    assert "покупать по K0" in payload["system_prompt"]
    assert payload["messages"][0]["role"] == "user"
    assert "покупать по K1" in payload["messages"][0]["body"]
    assert "нельзя писать K1" in VERIFIER_SYSTEM_PROMPT
