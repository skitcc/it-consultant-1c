"""Shared Exchange account factory."""

from __future__ import annotations

from exchangelib import BASIC, DELEGATE, NTLM, Account, Configuration, Credentials
from exchangelib.protocol import BaseProtocol, NoVerifyHTTPAdapter

_AUTH_TYPES = {
    "ntlm": NTLM,
    "basic": BASIC,
}


def build_account(
    *,
    server: str,
    email: str,
    password: str,
    auth: str = "ntlm",
    verify_ssl: bool = True,
) -> Account:
    if not verify_ssl:
        BaseProtocol.HTTP_ADAPTER_CLS = NoVerifyHTTPAdapter

    auth_type = _AUTH_TYPES.get(auth.lower())
    if auth_type is None:
        raise ValueError(f"Unsupported EWS auth type: {auth!r}. Use ntlm or basic.")

    credentials = Credentials(username=email, password=password)
    config = Configuration(server=server, credentials=credentials, auth_type=auth_type)
    return Account(
        primary_smtp_address=email,
        config=config,
        autodiscover=False,
        access_type=DELEGATE,
    )
