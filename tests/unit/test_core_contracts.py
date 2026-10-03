"""Core contracts match what their docstrings and annotations promise (CR-04, 05, 07, 12, 14)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.core.models import (
    BatchExtractionResult,
    Document,
    ProcessingMetrics,
)
from iris_vector_rag.core.validators import PipelineValidator, ViolationSeverity


class RecordingStore:
    """A store with the abstract VectorStore surface: clear_documents, no clear."""

    def __init__(self):
        self.cleared = 0

    def clear_documents(self) -> None:
        self.cleared += 1


class ConformingPipeline(RAGPipeline):
    """A subclass that follows the base's declared signatures exactly."""

    def __init__(self, store):
        super().__init__(connection_manager=object(), vector_store=store)
        self.loaded: Dict[str, Any] = {}
        self.query_calls: List[Dict[str, Any]] = []

    def load_documents(self, documents_path=None, **kwargs) -> Dict[str, Any]:
        self.loaded = {"path": documents_path, **kwargs}
        return {"documents_loaded": len(kwargs.get("documents", []))}

    def query(self, query, top_k=5, generate_answer=True, **kwargs):
        self.query_calls.append(
            {"query": query, "top_k": top_k, "generate_answer": generate_answer}
        )
        return {
            "query": query,
            "answer": "an answer" if generate_answer else "",
            "retrieved_documents": [Document(page_content="hit", id="d1")],
            "contexts": ["hit"],
            "execution_time": 0.0,
            "metadata": {},
        }


class TestRetrieveReturnsWhatItsAnnotationSays:
    def test_retrieve_returns_documents_not_the_query_dict(self):
        pipeline = ConformingPipeline(RecordingStore())
        result = pipeline.retrieve("q", top_k=3)
        assert isinstance(result, list)
        assert [d.id for d in result] == ["d1"]

    def test_retrieve_does_not_generate_an_answer(self):
        pipeline = ConformingPipeline(RecordingStore())
        pipeline.retrieve("q", top_k=3)
        assert pipeline.query_calls == [
            {"query": "q", "top_k": 3, "generate_answer": False}
        ]


class TestIngestMatchesTheAbstractSignature:
    def test_ingest_reaches_load_documents_of_a_conforming_subclass(self):
        pipeline = ConformingPipeline(RecordingStore())
        docs = [Document(page_content="x", id="a")]
        pipeline.ingest(docs)
        assert pipeline.loaded["documents"] == docs

    def test_abstract_load_documents_makes_the_path_optional(self):
        import inspect

        sig = inspect.signature(RAGPipeline.load_documents)
        assert sig.parameters["documents_path"].default is None


class TestValidatorAgreesWithTheBase:
    def test_a_subclass_following_the_base_query_signature_has_no_error(self):
        errors = [
            v
            for v in PipelineValidator().validate_pipeline_class(ConformingPipeline)
            if v.severity is ViolationSeverity.ERROR
        ]
        assert errors == []

    def test_base_query_names_the_parameter_the_validator_requires(self):
        import inspect

        assert "query" in inspect.signature(RAGPipeline.query).parameters
        assert "query_text" not in inspect.signature(RAGPipeline.query).parameters


class TestClear:
    def test_clear_calls_clear_documents(self):
        store = RecordingStore()
        ConformingPipeline(store).clear()
        assert store.cleared == 1

    def test_clear_on_a_store_that_cannot_clear_raises(self):
        pipeline = ConformingPipeline(object())
        with pytest.raises(NotImplementedError, match="clear_documents"):
            pipeline.clear()


class _NoKwargs(ConformingPipeline):
    def query(self, query, top_k=5, generate_answer=True):  # noqa: D401
        return {}


class TestStrictMode:
    def test_default_keeps_warnings_as_warnings(self):
        v = PipelineValidator().validate_pipeline_class(_NoKwargs)
        assert any(x.severity is ViolationSeverity.WARNING for x in v)
        assert not any(x.severity is ViolationSeverity.ERROR for x in v)

    def test_strict_turns_class_warnings_into_errors(self):
        v = PipelineValidator(strict_mode=True).validate_pipeline_class(_NoKwargs)
        assert not any(x.severity is ViolationSeverity.WARNING for x in v)
        assert any(
            x.severity is ViolationSeverity.ERROR and x.category == "method_signature"
            for x in v
        )

    def test_strict_turns_response_warnings_into_errors(self):
        response = {
            "answer": "a",
            "retrieved_documents": [],
            "contexts": [],
            "sources": [],
            "execution_time": 0.1,
            "metadata": {},  # required metadata missing: WARNING by default
        }
        lax = PipelineValidator().validate_response(response)
        strict = PipelineValidator(strict_mode=True).validate_response(response)
        assert lax and all(x.severity is ViolationSeverity.WARNING for x in lax)
        assert len(strict) == len(lax)
        assert all(x.severity is ViolationSeverity.ERROR for x in strict)

    def test_strict_leaves_info_alone(self):
        class Legacy(ConformingPipeline):
            def query(self, query_text, query=None, top_k=5, **kwargs):
                return {}

        v = PipelineValidator(strict_mode=True).validate_pipeline_class(Legacy)
        assert any(x.severity is ViolationSeverity.INFO for x in v)


def _batch(success: bool, seconds: float, retries: int = 0) -> BatchExtractionResult:
    return BatchExtractionResult(
        batch_id="b",
        per_document_entities={"d": []},
        processing_time=seconds,
        success_status=success,
        retry_count=retries,
    )


class TestProcessingMetricsAverages:
    def test_a_failed_batch_does_not_dilute_the_average(self):
        m = ProcessingMetrics()
        m.update_with_batch(_batch(False, 0.0), batch_size=1)
        m.update_with_batch(_batch(True, 10.0), batch_size=1)
        assert m.average_batch_processing_time == pytest.approx(10.0)

    def test_failure_after_success_leaves_the_average_alone(self):
        m = ProcessingMetrics()
        m.update_with_batch(_batch(True, 10.0), batch_size=1)
        m.update_with_batch(_batch(False, 0.0), batch_size=1)
        m.update_with_batch(_batch(True, 20.0), batch_size=1)
        assert m.average_batch_processing_time == pytest.approx(15.0)
        assert m.failed_batches_count == 1
        assert m.total_batches_processed == 3

    def test_entity_rate_averages_over_successful_batches_only(self):
        m = ProcessingMetrics()
        m.update_with_batch(_batch(False, 0.0), batch_size=1)
        ok = _batch(True, 1.0)
        ok.per_document_entities = {"d": [object(), object()]}
        m.update_with_batch(ok, batch_size=1)
        assert m.entity_extraction_rate_per_batch == pytest.approx(2.0)


class TestDocumentHash:
    def test_equal_documents_hash_equal(self):
        a = Document(page_content="x", id="1", metadata={"k": 1})
        b = Document(page_content="x", id="1", metadata={"k": 1})
        assert a == b and hash(a) == hash(b)

    def test_unhashable_metadata_values_do_not_raise(self):
        d = Document(
            page_content="x", id="1", metadata={"tags": ["a", "b"], "n": {"x": 1}}
        )
        assert isinstance(hash(d), int)
        assert {d: 1}[d] == 1

    def test_hash_survives_metadata_mutation(self):
        d = Document(page_content="x", id="1", metadata={"k": 1})
        bucket = {d: "found"}
        before = hash(d)
        d.metadata["similarity_score"] = 0.9  # fusion does this
        assert hash(d) == before
        assert bucket[d] == "found"

    def test_different_ids_hash_apart_and_are_unequal(self):
        a = Document(page_content="x", id="1")
        b = Document(page_content="x", id="2")
        assert a != b
