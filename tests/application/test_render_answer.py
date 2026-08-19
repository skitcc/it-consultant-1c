from mail_gateway.application.render_answer import (
    NO_SOURCES_TEXT,
    SOURCES_HEADING,
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
    assert "https://" not in html
    assert "href=" not in html
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


def test_render_answer_strips_unsafe_html_and_links() -> None:
    html = render_answer(
        '<p>ok</p><script>alert(1)</script><a href="http://x">x</a><img src="http://x">',
        source_names=[],
    )
    assert "<script" not in html
    assert "<a " not in html
    assert "href=" not in html
    assert "<img" not in html
    assert "src=" not in html
    assert NO_SOURCES_TEXT in html


def test_strip_sources_footer_from_history() -> None:
    body = (
        "Ответ без ссылок.\n\n"
        f"{SOURCES_HEADING}:\n"
        "guide.pdf"
    )
    assert strip_sources_footer(body) == "Ответ без ссылок."
