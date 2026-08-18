from mail_gateway.application.clean_email_body import clean_email_body


def test_strips_iphone_signature_and_quotes() -> None:
    raw = (
        "проверка истории\r\n"
        "Sent from my iPhone\r\n"
        "\r\n"
        "> On 29 Jul 2026, at 17:21, AI Assistant wrote:\r\n"
        "> old quote\r\n"
    )
    assert clean_email_body(raw) == "проверка истории"


def test_strips_outlook_forward_block() -> None:
    raw = (
        "Hello\n"
        "\n"
        "________________________________________\n"
        "From: User <u@x.ru>\n"
        "Sent: yesterday\n"
        "Subject: old\n"
        "\n"
        "quoted\n"
    )
    assert clean_email_body(raw) == "Hello"


def test_strips_stub_meta_lines() -> None:
    raw = (
        "Это тестовый ответ почтового шлюза (stub).\n"
        "\n"
        "conversation_id=AAQk...\n"
        "subject=Re: Поддержка 2\n"
        "________________________________________\n"
        "From: Иван\n"
    )
    assert clean_email_body(raw) == "Это тестовый ответ почтового шлюза (stub)."
