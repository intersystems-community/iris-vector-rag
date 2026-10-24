"""`where` through QueryOptions, the retrieval engine and the basic pipeline (092 US3, FR-010)."""

from __future__ import annotations

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.core.query_options import QueryOptions, normalize_query_params
from iris_vector_rag.retrieval.engine import RetrievalEngine

WHERE = {"PatientId": "P02", "EmbeddingModel": "m"}


class RecordingStore:
    """A store that records what each search was asked."""

    def __init__(self):
        self.calls = []

    def search_by_text(self, query, top_k=5, metadata_filter=None, where=None):
        self.calls.append(("vector", where))
        return [Document(id="a", page_content="alpha", metadata={})]

    def similarity_search_with_score(self, query, k=4, filter=None, where=None):
        self.calls.append(("vector_scored", where))
        return [(Document(id="a", page_content="alpha", metadata={}), 0.9)]

    def text_search(self, query, top_k=5, metadata_filter=None, where=None):
        self.calls.append(("text", where))
        return [Document(id="a", page_content="alpha", metadata={"text_score": 1.0})]


def engine(store):
    eng = RetrievalEngine(vector_store=store, connection=None)
    return eng


def test_normalize_query_params_carries_where():
    opts = normalize_query_params("q", where=WHERE)
    assert opts.where == WHERE
    assert normalize_query_params("q").where is None


@pytest.mark.parametrize("mode", ["vector", "text", "hybrid", "rrf"])
def test_every_document_leg_is_asked_with_where(mode, monkeypatch):
    monkeypatch.setattr(
        "iris_vector_rag.retrieval.modes.check_prerequisites", lambda *a, **k: None
    )
    store = RecordingStore()
    engine(store).retrieve(QueryOptions(query="q", retrieval=mode, where=WHERE))
    assert store.calls, mode
    assert all(where == WHERE for _, where in store.calls), store.calls


def test_the_scored_vector_leg_is_asked_with_where(monkeypatch):
    monkeypatch.setattr(
        "iris_vector_rag.retrieval.modes.check_prerequisites", lambda *a, **k: None
    )
    store = RecordingStore()
    engine(store).retrieve(
        QueryOptions(
            query="q", retrieval="vector", where=WHERE, similarity_threshold=0.1
        )
    )
    assert store.calls == [("vector_scored", WHERE)]


def test_a_store_written_before_092_is_not_given_where_when_there_is_none(monkeypatch):
    monkeypatch.setattr(
        "iris_vector_rag.retrieval.modes.check_prerequisites", lambda *a, **k: None
    )

    class OldStore:
        def search_by_text(self, query, top_k=5, metadata_filter=None):
            return []

        def text_search(self, query, top_k=5, metadata_filter=None):
            return []

    for mode in ("vector", "text", "rrf"):
        assert (
            engine(OldStore()).retrieve(QueryOptions(query="q", retrieval=mode)) == []
        )


@pytest.mark.parametrize("mode", ["phrase", "concept", "global", "mix"])
def test_modes_that_cannot_scope_by_column_refuse_where(mode, monkeypatch):
    monkeypatch.setattr(
        "iris_vector_rag.retrieval.modes.check_prerequisites", lambda *a, **k: None
    )
    with pytest.raises(ValueError, match="where"):
        engine(RecordingStore()).retrieve(
            QueryOptions(query="q", retrieval=mode, where=WHERE)
        )


def test_the_basic_pipeline_forwards_where_to_the_store():
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    pipeline = BasicRAGPipeline.__new__(BasicRAGPipeline)
    pipeline.vector_store = RecordingStore()

    docs = pipeline._search_text("q", 3, None, WHERE)

    assert pipeline.vector_store.calls == [("vector", WHERE)]
    assert docs and docs[0].id == "a"
