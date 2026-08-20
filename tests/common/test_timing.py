import logging

from common.timing import begin_request, end_request, record, span


def test_request_timer_records_steps_and_summary(caplog) -> None:
    caplog.set_level(logging.INFO)
    timer = begin_request(conversation_id="c1", item_id="i1")
    try:
        with span("history"):
            pass
        record("rerank_1/2", 0.5)
        record("rerank_2/2", 0.25)
        with span("rerank"):
            pass
        record("llm_layer_1", 1.5)
    finally:
        end_request(timer)

    names = [name for name, _elapsed in timer.steps]
    assert names == ["history", "rerank_1/2", "rerank_2/2", "rerank", "llm_layer_1"]
    assert "Timing step=history" in caplog.text
    assert "Timing step=rerank_1/2 elapsed=0.500s" in caplog.text
    assert "Timing step=llm_layer_1 elapsed=1.500s" in caplog.text
    assert "Timing summary conversation_id=c1 item_id=i1" in caplog.text
    assert "total=" in caplog.text
    assert "llm_layer_1=1.500s" in caplog.text


def test_span_is_noop_without_active_request(caplog) -> None:
    caplog.set_level(logging.INFO)
    with span("embed_query"):
        record("should_skip", 1.0)
    assert "Timing step=" not in caplog.text
    assert "Timing summary" not in caplog.text
