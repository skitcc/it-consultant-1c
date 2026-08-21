import pytest

from mail_gateway.application.render_answer import (
    NO_SOURCES_TEXT,
    SOURCES_HEADING,
    UnsafeAnswerError,
    render_answer,
    strip_sources_footer,
)


def test_render_answer_strips_numeric_citations_and_markdown() -> None:
    raw = """
**Ответ:**
Перейти в **Профиль**. [1][3]

### Создать подпись
- Нажать «Создать». фрагменты [8]-[9]
См. https://example.com и [док](https://intranet/doc)
"""
    html = render_answer(raw, source_names=["Памятка.pdf"])
    assert "[1]" not in html
    assert "[3]" not in html
    assert "фрагмент" not in html.lower()
    assert 'href="https://example.com"' in html
    assert 'href="https://intranet/doc"' in html
    assert ">док</a>" in html
    assert "**" not in html
    assert "###" not in html
    assert "<strong>Ответ:</strong>" in html or "Ответ:" in html
    assert "<ul" in html
    assert SOURCES_HEADING in html
    assert "Памятка.pdf" in html
    assert html.index(SOURCES_HEADING) > html.index("Профиль")


def test_render_answer_converts_pipe_table() -> None:
    raw = """
| Этап | Условие |
|------|---------|
| K0-2 | PM готов покупать по K0 |
| K1 | после выполнения K0-2 |
"""
    html = render_answer(raw, source_names=["грейд.pdf", "грейд.pdf"])
    assert "<table" in html
    assert "<th" in html
    assert "K0-2" in html
    assert "покупать по K0" in html
    assert "|" not in html.split(SOURCES_HEADING)[0]
    assert html.count("грейд.pdf") == 1
    assert "border:1px solid" in html


def test_render_answer_keeps_safe_links_and_strips_unsafe_html() -> None:
    html = render_answer(
        '<p>ok <a href="https://ovpn.1c-perspective.ru:943">VPN</a></p>'
        '<script>alert(1)</script>'
        '<a href="javascript:alert(1)">xss</a>'
        '<img src="http://x">',
        source_names=[],
    )
    assert "<script" not in html
    assert "<img" not in html
    assert "src=" not in html
    assert "javascript:" not in html
    assert 'href="https://ovpn.1c-perspective.ru:943"' in html
    assert ">VPN</a>" in html
    assert "xss" in html
    assert NO_SOURCES_TEXT in html


def test_render_answer_keeps_documentation_resource_links() -> None:
    raw = """
<p>Корпоративные ресурсы:</p>
<ul>
<li><a href="https://mail.1c-perspective.ru">Электронная почта</a></li>
<li><a href="https://ovpn.1c-perspective.ru:943">Сервис ВПН-подключения</a></li>
<li>Общие сетевые папки: \\\\pers.local\\Common</li>
</ul>
"""
    html = render_answer(raw, source_names=["Памятка - Корпоративные ресурсы.pdf"])
    assert 'href="https://mail.1c-perspective.ru"' in html
    assert 'href="https://ovpn.1c-perspective.ru:943"' in html
    assert "\\\\pers.local\\Common" in html
    assert "javascript:" not in html


def test_strip_sources_footer_from_history() -> None:
    body = (
        "Ответ без ссылок.\n\n"
        f"{SOURCES_HEADING}:\n"
        "guide.pdf"
    )
    assert strip_sources_footer(body) == "Ответ без ссылок."


@pytest.mark.parametrize(
    "raw",
    [
        "thinking - private content Public",
        "analysis: private",
        "<p>reasoning internal</p><p>answer</p>",
        "content\nPublic answer",
    ],
)
def test_render_answer_rejects_internal_reasoning_prefix(raw: str) -> None:
    with pytest.raises(UnsafeAnswerError):
        render_answer(raw)
