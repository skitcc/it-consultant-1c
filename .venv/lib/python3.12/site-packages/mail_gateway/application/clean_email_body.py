"""Strip quoted history and mobile signatures from email bodies."""

from __future__ import annotations

import re

_SIGNATURE_PATTERNS = (
    re.compile(r"^sent from my iphone\s*$", re.I),
    re.compile(r"^sent from my ipad\s*$", re.I),
    re.compile(r"^sent from my android\s*$", re.I),
    re.compile(r"^sent from mail for windows\s*$", re.I),
    re.compile(r"^get outlook for .*$", re.I),
    re.compile(r"^отправлено с (моего )?iphone\s*$", re.I),
    re.compile(r"^отправлено с (моего )?ipad\s*$", re.I),
    re.compile(r"^отправлено с android\s*$", re.I),
)

_QUOTE_START_PATTERNS = (
    re.compile(r"^_{5,}\s*$"),
    re.compile(r"^-{5,}\s*original message\s*-{5,}\s*$", re.I),
    re.compile(r"^-{5,}\s*пересылаемое сообщение\s*-{5,}\s*$", re.I),
    re.compile(r"^from:\s+.+$", re.I),
    re.compile(r"^от кого:\s+.+$", re.I),
    re.compile(r"^on .+ wrote:\s*$", re.I),
    re.compile(r"^.*написал\(а\):\s*$", re.I),
)

_STUB_META_PATTERNS = (
    re.compile(r"^conversation_id\s*=.*$", re.I),
    re.compile(r"^subject\s*=.*$", re.I),
    re.compile(r"^messages_count\s*=.*$", re.I),
    re.compile(r"^last_body\s*=.*$", re.I),
)


def clean_email_body(text: str) -> str:
    """Keep only the new message text: drop quotes, separators, mobile signatures."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    lines = normalized.split("\n")
    kept: list[str] = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if _is_quote_start(stripped):
            break
        if stripped.startswith(">"):
            break
        if any(pattern.match(stripped) for pattern in _STUB_META_PATTERNS):
            continue
        if any(pattern.match(stripped) for pattern in _SIGNATURE_PATTERNS):
            continue

        kept.append(line)

    # Trim trailing empty lines produced after removals.
    while kept and not kept[-1].strip():
        kept.pop()
    while kept and not kept[0].strip():
        kept.pop(0)

    return "\n".join(kept).strip()


def _is_quote_start(stripped: str) -> bool:
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in _QUOTE_START_PATTERNS)
