"""Turn model output into a sanitized HTML email body."""

from __future__ import annotations

import re
from collections.abc import Sequence
from html import escape, unescape
from html.parser import HTMLParser

SOURCES_HEADING = "Документы, использованные при подготовке ответа"
NO_SOURCES_TEXT = "Документы при подготовке ответа не использовались."

_ALLOWED_TAGS = frozenset(
    {
        "p",
        "h2",
        "h3",
        "ul",
        "ol",
        "li",
        "strong",
        "em",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "br",
    }
)
_VOID_TAGS = frozenset({"br"})
_SKIP_TAGS = frozenset({"script", "style"})
_TAG_ALIASES = {"b": "strong", "i": "em"}

_TABLE_STYLE = "border-collapse:collapse;width:100%;margin:12px 0;"
_CELL_STYLE = (
    "border:1px solid #8a8a8a;padding:6px 10px;vertical-align:top;"
    "word-break:break-word;"
)
_HEADER_CELL_STYLE = f"{_CELL_STYLE}background:#f2f2f2;font-weight:bold;"
_TAG_STYLES = {
    "table": _TABLE_STYLE,
    "th": _HEADER_CELL_STYLE,
    "td": _CELL_STYLE,
    "p": "margin:0 0 10px 0;",
    "h2": "margin:16px 0 8px 0;font-size:16px;",
    "h3": "margin:14px 0 8px 0;font-size:14px;",
    "ul": "margin:0 0 10px 20px;",
    "ol": "margin:0 0 10px 20px;",
}

_CITATION_RE = re.compile(
    r"(?:фрагмент(?:ы|а|ов)?\s*)?\[\s*\d+(?:\s*[-–—,]\s*\d+)?\s*\]",
    re.IGNORECASE,
)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_URL_RE = re.compile(r"(https?://[^\s<>]+|www\.[^\s<>]+)", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_UL_RE = re.compile(r"^[-*]\s+(.*)$")
_OL_RE = re.compile(r"^(\d+)[.)]\s+(.*)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_HTML_BLOCK_RE = re.compile(r"<(p|table|ul|ol|h2|h3)\b", re.IGNORECASE)
_INTERNAL_REASONING_PREFIX = re.compile(
    r"^\s*(?:<[^>]+>\s*)*(?:thinking|analysis|reasoning|content)\b",
    re.IGNORECASE,
)


class UnsafeAnswerError(ValueError):
    """Model output contains an internal reasoning marker."""


def render_answer(text: str, *, source_names: Sequence[str] = ()) -> str:
    if _INTERNAL_REASONING_PREFIX.match(text or ""):
        raise UnsafeAnswerError("internal reasoning must not be rendered")
    cleaned = strip_disallowed_markup(text or "")
    html = (
        cleaned
        if _HTML_BLOCK_RE.search(cleaned)
        else markdownish_to_html(cleaned)
    )
    body = sanitize_html(html)
    if not body.strip():
        body = f"<p>{escape(cleaned) if cleaned else ''}</p>"
    return wrap_email_document(body, source_names)


def strip_disallowed_markup(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    without_links = _MD_LINK_RE.sub(r"\1", normalized)
    without_citations = _CITATION_RE.sub("", without_links)
    without_urls = _URL_RE.sub("", without_citations)
    without_urls = re.sub(r"[ \t]{2,}", " ", without_urls)
    without_urls = re.sub(r" +\n", "\n", without_urls)
    return without_urls.strip()


def strip_sources_footer(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cut = None
    for marker in (SOURCES_HEADING, NO_SOURCES_TEXT):
        index = normalized.find(marker)
        if index >= 0 and (cut is None or index < cut):
            cut = index
    if cut is None:
        return normalized.strip()
    return normalized[:cut].rstrip()


def markdownish_to_html(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").split("\n")]
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or _FENCE_RE.match(line.strip()):
            i += 1
            continue
        if _is_table_start(lines, i):
            html, i = _consume_table(lines, i)
            blocks.append(html)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            level = 2 if len(heading.group(1)) <= 2 else 3
            tag = f"h{level}"
            blocks.append(f"<{tag}>{_inline(heading.group(2))}</{tag}>")
            i += 1
            continue
        if _UL_RE.match(line):
            html, i = _consume_list(lines, i, ordered=False)
            blocks.append(html)
            continue
        if _OL_RE.match(line):
            html, i = _consume_list(lines, i, ordered=True)
            blocks.append(html)
            continue
        para, i = _consume_paragraph(lines, i)
        if para:
            blocks.append(f"<p>{para}</p>")
    return "".join(blocks) or f"<p>{_inline(text)}</p>"


def sanitize_html(html: str) -> str:
    parser = _HtmlSanitizer()
    parser.feed(html or "")
    parser.close()
    return parser.result()


def wrap_email_document(body_html: str, source_names: Sequence[str]) -> str:
    footer_items: list[str] = []
    seen: set[str] = set()
    for name in source_names:
        cleaned = (name or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        footer_items.append(escape(cleaned))
    if footer_items:
        items = "".join(f"<li>{name}</li>" for name in footer_items)
        footer = (
            f'<p style="{_TAG_STYLES["p"]}"><strong>{SOURCES_HEADING}:</strong></p>'
            f'<ul style="{_TAG_STYLES["ul"]}">{items}</ul>'
        )
    else:
        footer = f'<p style="{_TAG_STYLES["p"]}">{escape(NO_SOURCES_TEXT)}</p>'
    inner = (
        f'<div style="font-family:Segoe UI,Arial,sans-serif;font-size:14px;'
        f'line-height:1.45;color:#222;">{body_html}{footer}</div>'
    )
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head>"
        f"<body>{inner}</body></html>"
    )


def _inline(text: str) -> str:
    escaped = escape(text.strip())
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC_RE.sub(r"<em>\1</em>", escaped)
    return escaped


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(_TABLE_ROW_RE.match(stripped)) and stripped.count("|") >= 2


def _is_sep_row(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line.strip()))


def _is_table_start(lines: list[str], index: int) -> bool:
    if index >= len(lines) or not _is_table_row(lines[index]):
        return False
    if index + 1 < len(lines) and _is_sep_row(lines[index + 1]):
        return True
    return index + 1 < len(lines) and _is_table_row(lines[index + 1])


def _split_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _consume_table(lines: list[str], index: int) -> tuple[str, int]:
    header = _split_cells(lines[index])
    cursor = index + 1
    if cursor < len(lines) and _is_sep_row(lines[cursor]):
        cursor += 1
    rows: list[list[str]] = []
    while cursor < len(lines) and _is_table_row(lines[cursor]) and not _is_sep_row(
        lines[cursor]
    ):
        rows.append(_split_cells(lines[cursor]))
        cursor += 1

    head_cells = "".join(f"<th>{_inline(cell)}</th>" for cell in header)
    body_rows = []
    for row in rows:
        padded = row + [""] * max(0, len(header) - len(row))
        cells = "".join(f"<td>{_inline(cell)}</td>" for cell in padded[: len(header) or len(padded)])
        body_rows.append(f"<tr>{cells}</tr>")
    html = (
        f"<table><thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )
    return html, cursor


def _consume_list(lines: list[str], index: int, *, ordered: bool) -> tuple[str, int]:
    pattern = _OL_RE if ordered else _UL_RE
    items: list[str] = []
    cursor = index
    while cursor < len(lines):
        match = pattern.match(lines[cursor])
        if not match:
            break
        text = match.group(2) if ordered else match.group(1)
        items.append(f"<li>{_inline(text)}</li>")
        cursor += 1
    tag = "ol" if ordered else "ul"
    return f"<{tag}>{''.join(items)}</{tag}>", cursor


def _consume_paragraph(lines: list[str], index: int) -> tuple[str, int]:
    parts: list[str] = []
    cursor = index
    while cursor < len(lines):
        line = lines[cursor]
        if not line.strip():
            break
        if _FENCE_RE.match(line.strip()) or _HEADING_RE.match(line):
            break
        if _UL_RE.match(line) or _OL_RE.match(line) or _is_table_start(lines, cursor):
            break
        parts.append(line.strip())
        cursor += 1
    return _inline(" ".join(parts)), cursor


class _HtmlSanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def result(self) -> str:
        return "".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = _TAG_ALIASES.get(tag.lower(), tag.lower())
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip or tag not in _ALLOWED_TAGS:
            return
        style = _TAG_STYLES.get(tag)
        if style:
            self._parts.append(f'<{tag} style="{style}">')
        elif tag in _VOID_TAGS:
            self._parts.append(f"<{tag}>")
        else:
            self._parts.append(f"<{tag}>")

    def handle_endtag(self, tag: str) -> None:
        tag = _TAG_ALIASES.get(tag.lower(), tag.lower())
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
            return
        if self._skip or tag not in _ALLOWED_TAGS or tag in _VOID_TAGS:
            return
        self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._skip or not data:
            return
        self._parts.append(escape(unescape(data)))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = _TAG_ALIASES.get(tag.lower(), tag.lower())
        if self._skip or tag not in _ALLOWED_TAGS:
            return
        if tag in _VOID_TAGS:
            self._parts.append(f"<{tag}>")
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)
