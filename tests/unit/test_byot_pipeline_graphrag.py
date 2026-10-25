"""`where` through the GraphRAG pipelines (feature 092, FR-021).

GraphRAGPipeline reads documents two ways: a vector fallback (a search of the store) and
the knowledge graph, whose entities name source documents that are then read back from
the store's table by id. Both are scoped by ``where`` -- the first by the search, the
second by fetching the ids under the same predicate -- and a required filter missing
from the query is an error, never a wider read. The global GraphRAG pipeline answers
from its own paper tables and globals, so it refuses ``where``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import (
    FilterError,
    RequiredFilterMissing,
    VectorStoreConnectionError,
)
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline
from iris_vector_rag.storage.attached import FilterSpec
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.table_shape import ColumnRole, TableShape

CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "byot" / "catalog"
SPEC = FilterSpec(
    filter_columns=("PatientRef", "EmbeddingModel"), required_filters=("PatientRef",)
)
WHERE = {"PatientRef": "Patient/2"}

#: id -> the patient who owns the chunk.
CHUNKS = {
    "DocumentReference/1#1": "Patient/1",
    "DocumentReference/2#1": "Patient/2",
    "DocumentReference/2#2": "Patient/2",
    "DocumentReference/3#1": "Patient/3",
}


@pytest.fixture(scope="module")
def shape() -> TableShape:
    snapshot = RecordedCatalog.from_fixture_dir(CATALOG).snapshot("BYOTStf", "DocChunk")
    return TableShape.from_snapshot(
        snapshot,
        declared={ColumnRole.IDENTIFIER: ("DocRef", "Seq"), ColumnRole.TEXT: "Text"},
    )


class ScopedStore:
    """An attached table that honours the predicate it is handed.

    ``fetch_documents_by_ids`` is given `compile_where`'s SQL and parameters, so the
    double reads the patient from the parameters: a store that dropped the predicate
    would return every id it was asked for.
    """

    def __init__(self, shape, spec=SPEC):
        self._shape = shape
        self.filter_spec = spec
        self.searches = []
        self.fetches = []

    def shape(self, cursor=None):
        return self._shape

    def search_by_vector(self, embedding, top_k=5, metadata_filter=None, where=None):
        self.searches.append(where)
        return [
            (Document(id=i, page_content=f"vector {i}"), 0.9)
            for i, patient in CHUNKS.items()
            if where is None or patient == where["PatientRef"]
        ][:top_k]

    def fetch_documents_by_ids(self, ids, where_sql="", where_params=()):
        self.fetches.append((list(ids), where_sql, list(where_params)))
        return [
            Document(id=i, page_content=f"kg {i}", metadata={"patient": CHUNKS[i]})
            for i in ids
            if i in CHUNKS and (not where_sql or CHUNKS[i] == where_params[0])
        ]


class EntityRows:
    """The executor the pipeline reads the knowledge graph through."""

    def __init__(self, doc_ids):
        self.doc_ids = doc_ids
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), params))
        return [{"source_doc_id": i} for i in self.doc_ids]


def pipeline_for(store, doc_ids=(), seeds=("E1",)):
    pipeline = GraphRAGPipeline.__new__(GraphRAGPipeline)
    pipeline.vector_store = store
    pipeline.pipeline_config = {}
    pipeline.llm_func = None
    pipeline.max_depth = 1
    pipeline.embedding_manager = Mock()
    pipeline.embedding_manager.embed_text.return_value = [0.1, 0.2]
    pipeline._executor = EntityRows(doc_ids)
    pipeline._find_seed_entities = lambda query: list(seeds)
    pipeline._expand_neighborhood = lambda start, depth: set(start)
    return pipeline


def retrieved_ids(response):
    return [d.id for d in response["retrieved_documents"]]


# --- the vector fallback ------------------------------------------------------------


def test_the_vector_fallback_searches_under_where(shape):
    store = ScopedStore(shape)
    response = pipeline_for(store, seeds=()).query(
        "anticoagulation", top_k=5, generate_answer=False, where=WHERE
    )
    assert store.searches == [WHERE]
    assert response["metadata"]["retrieval_method"] == "vector_fallback"
    assert retrieved_ids(response) == [
        "DocumentReference/2#1",
        "DocumentReference/2#2",
    ]


def test_a_store_that_takes_no_where_is_called_as_before(shape):
    """FR-017: no `where`, no required filter -> the search has no `where` argument."""
    searched = []

    class PlainStore:
        def search_by_vector(self, embedding, top_k=5):
            searched.append(top_k)
            return [(Document(id="d", page_content="t"), 0.5)]

    response = pipeline_for(PlainStore(), seeds=()).query(
        "q", top_k=3, generate_answer=False
    )
    assert searched == [3]
    assert retrieved_ids(response) == ["d"]


# --- the knowledge graph ------------------------------------------------------------


def test_documents_the_graph_names_are_fetched_under_where(shape):
    store = ScopedStore(shape)
    ids = list(CHUNKS)  # the graph names every patient's chunks
    response = pipeline_for(store, doc_ids=ids).query(
        "anticoagulation", top_k=5, generate_answer=False, where=WHERE
    )
    assert response["metadata"]["retrieval_method"] == "knowledge_graph"
    assert retrieved_ids(response) == [
        "DocumentReference/2#1",
        "DocumentReference/2#2",
    ]
    assert store.fetches and all(
        sql == '"PatientRef" = ?' and params == ["Patient/2"]
        for _, sql, params in store.fetches
    )
    assert store.searches == []  # the graph answered; nothing fell back


def test_an_out_of_scope_hit_does_not_take_a_top_k_slot(shape):
    store = ScopedStore(shape)
    # sorted, Patient/1's chunk is first: cutting at top_k before scoping returns nothing
    response = pipeline_for(store, doc_ids=list(CHUNKS)).query(
        "q", top_k=1, generate_answer=False, where=WHERE
    )
    assert retrieved_ids(response) == ["DocumentReference/2#1"]


def test_entity_documents_are_looked_up_in_the_library_graph_not_the_attached_schema(
    shape,
):
    """`BYOTStf.Entities` does not exist; the knowledge graph is the library's."""
    pipeline = pipeline_for(ScopedStore(shape), doc_ids=list(CHUNKS))
    pipeline.query("q", top_k=5, generate_answer=False, where=WHERE)
    sql, _ = pipeline._executor.statements[0]
    assert "FROM RAG.Entities" in sql and "BYOTStf" not in sql, sql


def test_a_scoped_hit_carries_its_retrieval_method(shape):
    response = pipeline_for(ScopedStore(shape), doc_ids=list(CHUNKS)).query(
        "q", top_k=5, generate_answer=False, where=WHERE
    )
    assert all(
        d.metadata["retrieval_method"] == "knowledge_graph"
        for d in response["retrieved_documents"]
    )


def test_a_failed_scoped_fetch_is_an_error_not_an_empty_result(shape):
    store = ScopedStore(shape)
    store.fetch_documents_by_ids = Mock(side_effect=VectorStoreConnectionError("down"))
    with pytest.raises(VectorStoreConnectionError):
        pipeline_for(store, doc_ids=list(CHUNKS)).query(
            "q", generate_answer=False, where=WHERE
        )


# --- refusals ------------------------------------------------------------------------


def test_a_missing_required_filter_stops_the_query_before_any_work(shape):
    store = ScopedStore(shape)
    pipeline = pipeline_for(store, doc_ids=list(CHUNKS))
    pipeline._find_seed_entities = Mock(side_effect=AssertionError("ran the graph"))
    with pytest.raises(RequiredFilterMissing, match="PatientRef"):
        pipeline.query("q", generate_answer=False)
    assert store.searches == [] and store.fetches == []
    pipeline.embedding_manager.embed_text.assert_not_called()


def test_the_graph_leg_alone_is_not_a_way_around_a_required_filter(shape):
    """A subclass that calls the leg directly, without `where`, is refused too."""
    store = ScopedStore(shape)
    pipeline = pipeline_for(store, doc_ids=list(CHUNKS))
    with pytest.raises(RequiredFilterMissing):
        pipeline._retrieve_via_kg("q", 5)
    with pytest.raises(RequiredFilterMissing):
        pipeline._get_documents_from_entities({"E1"}, 5)
    with pytest.raises(RequiredFilterMissing):
        pipeline._fallback_to_vector_search("q", 5)
    assert store.fetches == [] and store.searches == []


def test_a_key_outside_filter_columns_is_refused_before_any_sql(shape):
    store = ScopedStore(shape)
    pipeline = pipeline_for(store, doc_ids=list(CHUNKS))
    with pytest.raises(FilterError, match="Department"):
        pipeline.query("q", generate_answer=False, where={**WHERE, "Department": "x"})
    assert pipeline._executor.statements == [] and store.searches == []


def test_a_store_that_cannot_scope_refuses_where_rather_than_drop_it(shape):
    class Unscopable:
        def search_by_vector(self, embedding, top_k=5):
            raise AssertionError("searched without the scope")

    pipeline = pipeline_for(Unscopable(), seeds=())
    with pytest.raises(FilterError, match="where"):
        pipeline.query("q", generate_answer=False, where=WHERE)


def test_top_k_zero_still_validates_where(shape):
    with pytest.raises(RequiredFilterMissing):
        pipeline_for(ScopedStore(shape)).query("q", top_k=0, generate_answer=False)


# --- the global GraphRAG pipeline -----------------------------------------------------


def global_pipeline():
    from iris_vector_rag.pipelines.iris_global_graphrag import (
        IRISGlobalGraphRAGPipeline,
    )

    pipeline = IRISGlobalGraphRAGPipeline.__new__(IRISGlobalGraphRAGPipeline)
    pipeline.config = {}
    pipeline.global_graphrag_module = Mock()
    pipeline.iris_engine = pipeline.emb_model = pipeline.irispy = None
    return pipeline


@pytest.mark.parametrize("mode", ["rag", "graphrag"])
def test_the_global_pipeline_refuses_where_and_names_why(mode):
    pipeline = global_pipeline()
    with pytest.raises(ValueError, match="where") as caught:
        pipeline.query("q", mode=mode, where=WHERE)
    assert "not supported" in str(caught.value)
    assert not pipeline.global_graphrag_module.method_calls  # refused before any read


def test_the_global_pipeline_refuses_where_in_a_comparison_too():
    pipeline = global_pipeline()
    with pytest.raises(ValueError, match="where"):
        pipeline.compare_modes("q", where=WHERE)
    assert not pipeline.global_graphrag_module.method_calls


def test_the_global_pipeline_ignores_an_empty_where():
    pipeline = global_pipeline()
    pipeline.global_graphrag_module.ask_question_rag.return_value = "answer"
    response = pipeline.query("q", mode="rag", where={})
    assert response["answer"] == "answer"
