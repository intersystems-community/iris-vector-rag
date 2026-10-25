"""`where` through the CRAG pipeline (feature 092, FR-021).

CRAG reads documents in three places: the first vector search, the chunk search its
"ambiguous" correction runs, and the wider vector search its "disoriented" correction
runs. Each one applies ``where`` or refuses it; none drops it, and a required filter's
error reaches the caller instead of becoming a fallback answer.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import FilterError, RequiredFilterMissing
from iris_vector_rag.pipelines.crag import CRAGPipeline
from tests.unit.test_byot_store_vector import MEDTEAM, QUERY, TeamCursor, team_store
from tests.unit.test_crag_sql_and_embedding import ConnectionManager, legacy_chunk_shape

SCOPED = {
    **MEDTEAM,
    "filter_columns": ["PatientId", "EmbeddingModel", "DocDate"],
    "required_filters": ["PatientId", "EmbeddingModel"],
}
WHERE = {"PatientId": "P02", "EmbeddingModel": "mock:hashed@384"}
VECTOR = "[" + ",".join(map(str, QUERY)) + "]"
CHUNK_VECTOR = "[" + ",".join(f"{x:.10f}" for x in QUERY) + "]"  # CRAG's spelling
SCOPE_SQL = '"PatientId" = ? AND "EmbeddingModel" = ?'
SCOPE_PARAMS = ["P02", "mock:hashed@384"]


class CragCursor(TeamCursor):
    """TeamCursor, plus the three-column rows CRAG's own chunk query reads."""

    def execute(self, sql, params=None):
        super().execute(sql, params)
        if "similarity_score" in sql:
            self._result = [("P02-doc03#1", "chunk text", 0.7)]


class Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class Evaluator:
    def __init__(self, status):
        self.status = status

    def evaluate(self, query, docs):
        return self.status


def crag(store, cursor, status="confident", embedder=None):
    pipeline = CRAGPipeline.__new__(CRAGPipeline)
    pipeline.vector_store = store
    pipeline.embedding_func = embedder or (lambda texts: [QUERY for _ in texts])
    pipeline.connection_manager = ConnectionManager(Connection(cursor))
    pipeline.evaluator = Evaluator(status)
    pipeline.llm_func = None
    return pipeline


def scoped_pipeline(status="confident", **kwargs):
    cursor = CragCursor(
        ranked=[("P02-doc03#1", "apixaban", "Zpráva", "P02-doc03", 0.98)]
    )
    store = team_store(SCOPED, cursor, embedder=lambda t: QUERY)
    return crag(store, cursor, status, **kwargs), cursor


def executed_with_scope(cursor):
    """The statements that carry the predicate, with their parameters."""
    return [(sql, p) for sql, p in cursor.executed if SCOPE_SQL in sql]


# --- the chunk SQL -----------------------------------------------------------------


def test_the_chunk_sql_carries_the_predicate_and_its_params_after_the_vector():
    sql, params = CRAGPipeline._chunk_search_sql(
        legacy_chunk_shape(),
        [0.1, 0.2, 0.3, 0.4],
        3,
        where_sql='"a" = ? AND "b" IN (?, ?)',
        where_params=["x", 1, 2],
    )
    assert (
        'WHERE embedding IS NOT NULL AND "a" = ? AND "b" IN (?, ?) '
        "ORDER BY similarity_score DESC" in sql
    )
    assert sql.count("?") == 4
    assert params[1:] == ["x", 1, 2]
    assert params[0].startswith("[0.1")  # the vector binds to the first marker


def test_the_chunk_sql_without_a_predicate_is_what_it_was():
    sql, params = CRAGPipeline._chunk_search_sql(
        legacy_chunk_shape(), [0.1, 0.2, 0.3, 0.4], 3
    )
    assert sql.endswith("WHERE embedding IS NOT NULL ORDER BY similarity_score DESC")
    assert len(params) == 1


# --- every leg ----------------------------------------------------------------------


def test_the_first_vector_search_is_scoped():
    pipeline, cursor = scoped_pipeline()
    docs = pipeline._initial_retrieval("q", 3, where=WHERE)
    ((sql, params),) = executed_with_scope(cursor)
    assert params == [VECTOR, *SCOPE_PARAMS]
    assert [d.id for d in docs] == ["P02-doc03#1"]


def test_the_chunk_leg_reads_the_attached_table_under_the_predicate():
    pipeline, cursor = scoped_pipeline()
    docs = pipeline._enhance_retrieval("q", [], 3, where=WHERE)
    ((sql, params),) = executed_with_scope(cursor)
    assert "similarity_score" in sql
    assert "FROM BYOTMed.EvidenceChunk" in sql  # the attached table, not RAG.*
    assert 'WHERE "Embedding" IS NOT NULL AND ' + SCOPE_SQL in sql
    assert not any(" (" + SCOPE_SQL in s for s in cursor.sql)  # no leading "("
    assert params == [CHUNK_VECTOR, *SCOPE_PARAMS]
    assert [d.page_content for d in docs] == ["chunk text"]


def test_the_expansion_leg_is_scoped():
    pipeline, cursor = scoped_pipeline()
    pipeline._knowledge_base_expansion("q", 2, where=WHERE)
    ((sql, params),) = executed_with_scope(cursor)
    assert params == [VECTOR, *SCOPE_PARAMS]


@pytest.mark.parametrize(
    "status, legs",
    [("confident", 1), ("ambiguous", 2), ("disoriented", 2)],
)
def test_query_hands_where_to_every_leg_it_runs(status, legs):
    pipeline, cursor = scoped_pipeline(status)
    result = pipeline.query("q", top_k=2, generate_answer=False, where=WHERE)
    assert result["error"] is None
    assert len(executed_with_scope(cursor)) == legs
    assert all(p[1:] == SCOPE_PARAMS for _, p in executed_with_scope(cursor))
    assert result["metadata"]["retrieval_status"] == status


def test_no_where_leaves_a_store_without_required_filters_alone():
    cursor = CragCursor(
        ranked=[("P02-doc03#1", "apixaban", "Zpráva", "P02-doc03", 0.98)]
    )
    store = team_store(MEDTEAM, cursor, embedder=lambda t: QUERY)
    pipeline = crag(store, cursor, "ambiguous")
    result = pipeline.query("q", top_k=2, generate_answer=False)
    assert result["error"] is None
    assert not any("PatientId" in sql for sql in cursor.sql)


# --- refusing, never dropping --------------------------------------------------------


def test_a_missing_required_filter_raises_before_any_sql_or_embedding():
    embedded = []
    pipeline, cursor = scoped_pipeline(
        embedder=lambda t: embedded.append(t) or [QUERY for _ in t]
    )
    with pytest.raises(RequiredFilterMissing, match="EmbeddingModel"):
        pipeline.query("q", generate_answer=False, where={"PatientId": "P02"})
    with pytest.raises(RequiredFilterMissing):
        pipeline.query("q", generate_answer=False)
    assert cursor.executed == [] and embedded == []


def test_a_filter_error_from_the_search_is_not_turned_into_a_fallback():
    pipeline, cursor = scoped_pipeline()

    def refuse(*args, **kwargs):
        raise FilterError("cannot filter on Nope")

    pipeline.vector_store.similarity_search = refuse
    with pytest.raises(FilterError, match="Nope"):
        pipeline.query("q", generate_answer=False, where=WHERE)


def test_any_other_search_failure_still_falls_back():
    pipeline, cursor = scoped_pipeline()

    def fail(*args, **kwargs):
        raise RuntimeError("connection reset")

    pipeline.vector_store.similarity_search = fail
    result = pipeline.query("q", generate_answer=False, where=WHERE)
    assert result["metadata"]["retrieval_status"] == "fallback_vector"


def test_a_store_that_cannot_take_where_refuses_it():
    class PlainStore:
        def similarity_search(self, query_embedding, top_k, filter=None):
            raise AssertionError("the search must not run with where dropped")

    pipeline = crag(PlainStore(), TeamCursor())
    with pytest.raises(FilterError, match="where"):
        pipeline.query("q", generate_answer=False, where={"PatientId": "P02"})


def test_a_chunk_table_without_the_filter_columns_refuses_where():
    """A store on ivr's own tables: RAG.DocumentChunks has no PatientId."""
    cursor = CragCursor(
        ranked=[("P02-doc03#1", "apixaban", "Zpráva", "P02-doc03", 0.98)]
    )
    store = team_store(SCOPED, cursor, embedder=lambda t: QUERY)
    pipeline = crag(store, cursor, "ambiguous")
    pipeline._chunk_shape = lambda connection: legacy_chunk_shape()
    with pytest.raises(FilterError, match="DocumentChunks"):
        pipeline._enhance_retrieval("q", [], 3, where=WHERE)
    with pytest.raises(FilterError, match="DocumentChunks"):
        pipeline.query("q", generate_answer=False, where=WHERE)
    assert not any("similarity_score" in sql for sql in cursor.sql)


def test_the_documents_stay_documents():
    pipeline, _ = scoped_pipeline("ambiguous")
    result = pipeline.query("q", top_k=2, generate_answer=False, where=WHERE)
    assert all(isinstance(d, Document) for d in result["retrieved_documents"])
