"""Batch extraction retries, metrics and the extraction prompt (FR-005, FR-007).

`EntityExtractionService.extract_batch` retries a batch that fails as a whole with
exponential backoff (2s, 4s, 8s), splits it when the retries run out, records the
outcome in the global `BatchMetricsTracker`, and sends the LLM a prompt that asks for
the JSON array the parser reads. The LLM is a fake here and `time.sleep` is recorded,
not slept; the live versions are in tests/integration/test_batch_*.py.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from iris_vector_rag.common.batch_utils import BatchMetricsTracker
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.services.entity_extraction import (
    EntityExtractionError,
    EntityExtractionService,
)

REPLY = '[{"text": "Ada", "type": "PERSON", "confidence": 0.9}]'


def _config(**extraction):
    cfg = Mock(spec=ConfigurationManager)
    cfg.get.return_value = extraction
    cfg._config = {"entity_extraction": extraction}
    return cfg


def _docs(n):
    return [Document(page_content=f"text of d{i}", id=f"d{i}") for i in range(n)]


@pytest.fixture
def sleeps():
    """Record the backoff delays instead of waiting them out."""
    with patch("iris_vector_rag.common.batch_utils.time.sleep") as sleep:
        yield sleep


@pytest.fixture(autouse=True)
def fresh_metrics():
    BatchMetricsTracker.reset_instance()
    yield
    BatchMetricsTracker.reset_instance()


def _delays(sleep):
    return [c.args[0] for c in sleep.call_args_list]


def test_a_batch_that_fails_twice_is_retried_with_2s_then_4s(sleeps):
    calls = []

    def llm(prompt):
        calls.append(prompt)
        if len(calls) <= 2 * 3:  # the first two attempts fail for all three documents
            raise TimeoutError("llm timed out")
        return REPLY

    service = EntityExtractionService(_config(), llm_func=llm)
    result = service.extract_batch(_docs(3))

    assert result.success_status is True
    assert result.retry_count == 2
    assert result.failed == []
    assert set(result.per_document_entities) == {"d0", "d1", "d2"}
    assert _delays(sleeps) == [2.0, 4.0]


def test_a_partial_failure_is_reported_not_retried(sleeps):
    def llm(prompt):
        if "text of d0" in prompt:
            raise ConnectionError("llm down for d0")
        return REPLY

    service = EntityExtractionService(_config(), llm_func=llm)
    result = service.extract_batch(_docs(3))

    # d1 and d2 succeeded, so the batch did not fail as a whole: no retry, no split
    assert [f["document_id"] for f in result.failed] == ["d0"]
    assert set(result.per_document_entities) == {"d1", "d2"}
    assert result.success_status is False
    assert _delays(sleeps) == []


def test_a_whole_batch_failure_splits_down_to_single_documents(sleeps):
    attempts = []
    service = EntityExtractionService(_config(), llm_func=lambda p: REPLY)

    real_impl = service._extract_batch_impl

    def impl(batch):
        attempts.append(len(batch))
        if len(batch) > 1:
            raise EntityExtractionError("batch too big for the llm")
        return real_impl(batch)

    with patch.object(service, "_extract_batch_impl", side_effect=impl):
        result = service.extract_batch(_docs(4))

    assert result.success_status is True
    assert set(result.per_document_entities) == {"d0", "d1", "d2", "d3"}
    # 4 attempts at size 4 (1 + 3 retries), then 2 halves of 2 (4 attempts each), then singles
    assert attempts.count(4) == 4
    assert attempts.count(2) == 8
    assert attempts.count(1) == 4
    assert _delays(sleeps) == [2.0, 4.0, 8.0] * 3


def test_every_document_failing_raises_after_the_retries(sleeps):
    def llm(prompt):
        raise ConnectionError("llm down")

    service = EntityExtractionService(_config(), llm_func=llm)

    with pytest.raises(EntityExtractionError, match="llm down"):
        service.extract_batch(_docs(1))

    assert _delays(sleeps) == [2.0, 4.0, 8.0]
    metrics = service.get_batch_metrics()
    assert metrics.failed_batches_count == 1


def test_no_llm_raises_at_once_without_retrying(sleeps):
    service = EntityExtractionService(_config())

    with pytest.raises(EntityExtractionError, match="llm_func"):
        service.extract_batch(_docs(2))

    assert _delays(sleeps) == []


def test_metrics_record_the_batch(sleeps):
    service = EntityExtractionService(_config(), llm_func=lambda p: REPLY)

    service.extract_batch(_docs(5))
    metrics = service.get_batch_metrics()

    assert metrics.total_batches_processed == 1
    assert metrics.total_documents_processed == 5
    assert metrics.entity_extraction_rate_per_batch == 5
    assert metrics.failed_batches_count == 0
    assert metrics.zero_entity_documents_count == 0


def test_the_prompt_asks_for_json_and_names_the_types():
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return REPLY

    service = EntityExtractionService(
        _config(entity_types=["PERSON", "PRODUCT"]), llm_func=llm
    )
    service.extract_entities(_docs(1)[0])

    (prompt,) = prompts
    assert "JSON array" in prompt
    assert '"text"' in prompt and '"type"' in prompt and '"confidence"' in prompt
    assert "PERSON" in prompt and "PRODUCT" in prompt
    assert "text of d0" in prompt


def test_the_llm_configuration_is_logged_at_startup(caplog):
    cfg = _config()
    cfg.get.side_effect = lambda key, default=None: {
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_type": "openai"},
        "entity_extraction": {},
    }.get(key, default)

    with caplog.at_level(logging.INFO):
        EntityExtractionService(cfg, llm_func=lambda p: REPLY)

    assert "Entity Extraction Service - LLM Configuration" in caplog.text
    assert "gpt-4o-mini" in caplog.text


def test_a_missing_llm_configuration_is_warned_about(caplog):
    cfg = _config()
    cfg.get.side_effect = lambda key, default=None: (
        {} if key == "entity_extraction" else default
    )

    with caplog.at_level(logging.WARNING):
        EntityExtractionService(cfg)

    assert "No LLM configuration found" in caplog.text


# --------------------------------------------------------------------- concurrency


def _slow_llm(delay, fail_on=None, seen=None):
    import threading
    import time

    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def llm(prompt):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        try:
            time.sleep(delay)
            if fail_on and f"text of {fail_on}" in prompt:
                raise TimeoutError("llm timed out")
            doc = prompt.rsplit("text of ", 1)[1].split()[0]
            return (
                f'[{{"text": "entity of {doc}", "type": "PERSON", "confidence": 0.9}}]'
            )
        finally:
            with lock:
                state["now"] -= 1

    llm.state = state
    return llm


def _timed(service, docs):
    import time

    start = time.time()
    result = service.extract_batch(docs)
    return result, time.time() - start


def test_wall_clock_scales_with_max_workers():
    docs = _docs(8)
    sequential = EntityExtractionService(
        _config(batch_processing={"max_workers": 1}), llm_func=_slow_llm(0.1)
    )
    concurrent = EntityExtractionService(
        _config(batch_processing={"max_workers": 8}), llm_func=_slow_llm(0.1)
    )

    _, t_seq = _timed(sequential, docs)
    result, t_con = _timed(concurrent, docs)

    assert t_seq >= 0.8  # eight calls, one after another
    assert t_con < t_seq / 3  # eight calls at once: about one call's time
    assert result.success_status is True


def test_concurrency_is_bounded_by_max_workers():
    llm = _slow_llm(0.05)
    service = EntityExtractionService(
        _config(batch_processing={"max_workers": 3}), llm_func=llm
    )

    service.extract_batch(_docs(12))

    assert llm.state["peak"] == 3


def test_max_workers_defaults_to_8_and_1_is_sequential():
    default = _slow_llm(0.02)
    EntityExtractionService(_config(), llm_func=default).extract_batch(_docs(16))
    assert default.state["peak"] == 8

    one = _slow_llm(0.02)
    EntityExtractionService(
        _config(batch_processing={"max_workers": 1}), llm_func=one
    ).extract_batch(_docs(4))
    assert one.state["peak"] == 1


def test_an_invalid_max_workers_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="max_workers"):
        EntityExtractionService(
            _config(batch_processing={"max_workers": 0}), llm_func=lambda p: REPLY
        )


def test_results_keep_the_order_of_the_documents_under_concurrency():
    import random

    def llm(prompt):
        import time

        time.sleep(random.random() * 0.05)  # finish out of order
        doc = prompt.rsplit("text of ", 1)[1].split()[0]
        return f'[{{"text": "entity of {doc}", "type": "PERSON", "confidence": 0.9}}]'

    service = EntityExtractionService(
        _config(batch_processing={"max_workers": 8}), llm_func=llm
    )
    docs = _docs(20)

    result = service.extract_batch(docs)

    assert list(result.per_document_entities) == [d.id for d in docs]
    for d in docs:
        assert [e.text for e in result.per_document_entities[d.id]] == [
            f"entity of {d.id}"
        ]
        assert result.per_document_entities[d.id][0].source_document_id == d.id


def test_one_failing_document_under_concurrency(sleeps):
    service = EntityExtractionService(
        _config(batch_processing={"max_workers": 4}),
        llm_func=_slow_llm(0.02, fail_on="d3"),
    )
    docs = _docs(8)

    result = service.extract_batch(docs)

    assert [f["document_id"] for f in result.failed] == ["d3"]
    assert "timed out" in result.failed[0]["reason"]
    assert list(result.per_document_entities) == [d.id for d in docs if d.id != "d3"]
    assert result.success_status is False
    # a partial failure is not retried: no backoff delay (the fake's own 0.02s sleeps
    # go through the same patched sleep)
    assert not [d for d in _delays(sleeps) if d >= 2.0]
    metrics = service.get_batch_metrics()
    assert metrics.total_batches_processed == 1
    assert metrics.total_documents_processed == 8
    assert metrics.failed_batches_count == 1


def test_metrics_are_exact_under_concurrent_batches():
    import threading

    service = EntityExtractionService(
        _config(batch_processing={"max_workers": 4}), llm_func=_slow_llm(0.01)
    )
    threads = [
        threading.Thread(target=service.extract_batch, args=(_docs(5),))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    metrics = service.get_batch_metrics()
    assert metrics.total_batches_processed == 6
    assert metrics.total_documents_processed == 30
