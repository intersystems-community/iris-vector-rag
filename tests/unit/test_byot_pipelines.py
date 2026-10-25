"""`where` through the reranking and multi-query RRF pipelines (feature 092, FR-021).

Every pipeline either applies ``where`` to the searches it runs or refuses it; none
drops it. A required filter's error reaches the caller, never a log line.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import RequiredFilterMissing

WHERE = {"PatientId": "P02", "EmbeddingModel": "m1"}


class RecordingStore:
    """Records the `where` each search was given."""

    use_iris_embedding = False

    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def _answer(self, kind, where):
        self.calls.append((kind, where))
        if self.fail_with:
            raise self.fail_with
        return [Document(id="P02-a", page_content="alpha", metadata={})]

    def search_by_text(self, query, top_k=5, metadata_filter=None, where=None):
        return self._answer("search_by_text", where)

    def similarity_search(self, *args, **kwargs):
        return self._answer("similarity_search", kwargs.get("where"))


def basic_rerank(store):
    from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline

    pipeline = BasicRAGRerankingPipeline.__new__(BasicRAGRerankingPipeline)
    pipeline.vector_store = store
    pipeline.reranker = None
    pipeline.rerank_factor = 2
    pipeline.llm_func = None
    pipeline.use_iris_embedding = False
    pipeline.config_manager = None
    pipeline.connection_manager = None
    return pipeline


def multi_query(store):
    from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline

    pipeline = MultiQueryRRFPipeline.__new__(MultiQueryRRFPipeline)
    pipeline.vector_store = store
    pipeline.retrieved_k = 4
    pipeline.final_k = 2
    pipeline.rrf_k = 60
    pipeline.num_queries = 2
    pipeline.llm_func = None
    pipeline.llm = None
    pipeline.llm_model = None
    pipeline.embedding_func = None
    pipeline.use_llm_expansion = False
    pipeline.generate_query_variations = lambda q: [q, q + " too"]
    return pipeline


def test_the_reranking_pipeline_hands_where_to_the_basic_search():
    store = RecordingStore()
    pipeline = basic_rerank(store)
    try:
        pipeline.query("q", top_k=2, where=WHERE, generate_answer=False)
    except Exception:  # noqa: BLE001 - only the search's arguments are under test
        pass
    assert store.calls and all(where == WHERE for _, where in store.calls)


def test_every_multi_query_search_is_scoped_by_where():
    store = RecordingStore()
    multi_query(store).query("q", top_k=2, where=WHERE, generate_answer=False)
    assert [where for _, where in store.calls] == [WHERE, WHERE]


def test_multi_query_passes_the_metadata_filter_too():
    seen = []

    class Store(RecordingStore):
        def similarity_search(self, *args, **kwargs):
            seen.append(kwargs.get("filter"))
            return []

    multi_query(Store()).query(
        "q", top_k=2, metadata_filter={"source": "bio"}, generate_answer=False
    )
    assert seen == [{"source": "bio"}, {"source": "bio"}]


def test_multi_query_lets_a_missing_required_filter_reach_the_caller():
    store = RecordingStore(fail_with=RequiredFilterMissing("requires PatientId"))
    with pytest.raises(RequiredFilterMissing):
        multi_query(store).query("q", top_k=2, generate_answer=False)
