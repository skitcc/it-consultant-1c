from types import SimpleNamespace

from mail_gateway.adapters.ews.conversation_history import _body_text
from mail_gateway.application.render_answer import SOURCES_HEADING


def test_history_strips_html_and_sources_footer() -> None:
    html = (
        "<p>Как выпустить подпись.</p>"
        f"<p><strong>{SOURCES_HEADING}:</strong></p>"
        "<ul><li>guide.pdf</li></ul>"
    )
    item = SimpleNamespace(text_body=None, body=html)
    text = _body_text(item)
    assert "Как выпустить подпись." in text
    assert SOURCES_HEADING not in text
    assert "guide.pdf" not in text
