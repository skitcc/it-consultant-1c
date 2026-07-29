from mail_gateway.adapters.ews.account import normalize_ews_server


def test_normalize_bare_host() -> None:
    assert normalize_ews_server("mail.1c-perspective.ru") == "mail.1c-perspective.ru"


def test_normalize_full_url() -> None:
    assert (
        normalize_ews_server("https://mail.1c-perspective.ru/EWS/Exchange.asmx")
        == "mail.1c-perspective.ru"
    )


def test_normalize_host_with_path() -> None:
    assert (
        normalize_ews_server("mail.1c-perspective.ru/EWS/Exchange.asmx")
        == "mail.1c-perspective.ru"
    )
