import pytest

from mail_gateway.adapters.ews.account import build_account, normalize_ews_server


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


def test_rejects_single_session_pool() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_account(
            server="mail.example.com",
            email="bot@example.com",
            username="DOMAIN\\bot",
            password="secret",
            session_pool_size=1,
        )
