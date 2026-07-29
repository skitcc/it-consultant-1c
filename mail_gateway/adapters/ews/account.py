"""Shared Exchange account factory."""

from __future__ import annotations

from urllib.parse import urlparse

from exchangelib import BASIC, DELEGATE, NTLM, Account, Configuration, Credentials
from exchangelib.protocol import BaseProtocol, NoVerifyHTTPAdapter

_AUTH_TYPES = {
    "ntlm": NTLM,
    "basic": BASIC,
}


def normalize_ews_server(server: str) -> str:
    """Return hostname for exchangelib Configuration.server.

    Accepts either a bare host (mail.example.com) or a full EWS URL
    (https://mail.example.com/EWS/Exchange.asmx).
    """
    value = server.strip()
    if "://" not in value:
        return value.split("/", 1)[0]

    parsed = urlparse(value)
    if not parsed.hostname:
        raise ValueError(
            f"Invalid EWS_SERVER={server!r}. Use host only, e.g. mail.company.ru"
        )
    return parsed.hostname


def build_account(
    *,
    server: str,
    email: str,
    password: str,
    auth: str = "ntlm",
    verify_ssl: bool = True,
    username: str | None = None,
    session_pool_size: int = 2,
) -> Account:
    if session_pool_size < 2:
        raise ValueError(
            "EWS_SESSION_POOL_SIZE must be at least 2: streaming holds one "
            "session while message fetch/reply needs another"
        )

    # exchangelib caches Protocol instances per endpoint. Separate Account
    # objects therefore still share this pool. Its default size is 1, which
    # deadlocks when we fetch an item while GetStreamingEvents is active.
    BaseProtocol.SESSION_POOLSIZE = session_pool_size

    if not verify_ssl:
        BaseProtocol.HTTP_ADAPTER_CLS = NoVerifyHTTPAdapter

    auth_type = _AUTH_TYPES.get(auth.lower())
    if auth_type is None:
        raise ValueError(f"Unsupported EWS auth type: {auth!r}. Use ntlm or basic.")

    host = normalize_ews_server(server)
    login = (username or email).strip()
    credentials = Credentials(username=login, password=password)
    config = Configuration(server=host, credentials=credentials, auth_type=auth_type)
    return Account(
        primary_smtp_address=email,
        config=config,
        autodiscover=False,
        access_type=DELEGATE,
    )
