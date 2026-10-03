"""GraphRAG and ColBERT pipeline defects PL-03..PL-10 (feature pipelines-fix).

Each test names the defect it pins. They use fake cursors and connections: what the
fakes cannot show (the SQL against a real IRIS) is covered by
tests/integration/test_graph_colbert_pipelines_e2e.py.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
from iris_vector_rag.pipelines.colbert_iris.pipeline import IRISColBERTPipeline
from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import HybridRetrievalMethods
from iris_vector_rag.pipelines.iris_global_graphrag import IRISGlobalGraphRAGPipeline
from tests.unit.test_graphrag_document_join_shape import (
    RecordedStore,
    RecordingCursor,
    pipeline_for,
)


class Entity:
    def __init__(self, text):
        self.text = text


# --- PL-06: no silent swallowing in graph expansion ---------------------------------


def _graphrag(executor):
    pipeline = GraphRAGPipeline.__new__(GraphRAGPipeline)
    pipeline._executor = executor
    pipeline.entity_extraction_service = Mock()
    pipeline.entity_extraction_service.extract_entities.return_value = [Entity("a")]
    return pipeline


def test_seed_lookup_failure_is_raised_not_swallowed():
    executor = Mock()
    executor.execute.side_effect = RuntimeError("table RAG.Entities is gone")

    with pytest.raises(RuntimeError, match="RAG.Entities is gone"):
        _graphrag(executor)._find_seed_entities("anything")


def test_expansion_chunks_an_in_list_above_the_limit():
    calls = []

    def execute(sql, params=None):
        calls.append(list(params))
        return []

    executor = Mock()
    executor.execute.side_effect = execute
    nodes = {f"E{i}" for i in range(450)}

    _graphrag(executor)._expand_neighborhood(nodes, depth=1)

    assert len(calls) == 3, "450 nodes at 200 per statement is three statements"
    assert max(len(p) for p in calls) <= 400
    seen = {node for params in calls for node in params}
    assert seen == nodes, "every node is looked up"


def test_expansion_failure_is_raised_not_a_silent_break():
    executor = Mock()
    executor.execute.side_effect = RuntimeError("driver limit")

    with pytest.raises(RuntimeError, match="driver limit"):
        _graphrag(executor)._expand_neighborhood({"E1"}, depth=2)


def test_expansion_still_walks_the_graph():
    answers = iter([[{"target_entity_id": "E2"}], [{"target_entity_id": "E3"}], []])
    executor = Mock()
    executor.execute.side_effect = lambda sql, params=None: next(answers)

    assert _graphrag(executor)._expand_neighborhood({"E1"}, depth=3) == {
        "E1",
        "E2",
        "E3",
    }


# --- PL-07: get_performance_statistics closes its cursor ----------------------------


def test_performance_statistics_closes_the_cursor_when_a_count_fails():
    cursor = Mock()
    cursor.execute.side_effect = RuntimeError("boom")
    manager = Mock()
    manager.get_connection.return_value.cursor.return_value = cursor
    methods = HybridRetrievalMethods.__new__(HybridRetrievalMethods)
    methods.vector_optimizer = None

    stats = methods.get_performance_statistics(manager)

    assert "entity_count" not in stats
    cursor.close.assert_called_once()


# --- PL-09: delete_node keeps both errors -------------------------------------------


def _hybrid(graph_error=None, vector_error=None):
    pipeline = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    pipeline.iris_engine = Mock()
    pipeline.vector_store = Mock()
    if graph_error:
        pipeline.iris_engine.delete_node.side_effect = graph_error
    if vector_error:
        pipeline.vector_store.delete_documents.side_effect = vector_error
    return pipeline


def test_delete_node_reports_the_vector_failure_after_a_graph_failure():
    pipeline = _hybrid(RuntimeError("graph down"), RuntimeError("vector down"))

    with pytest.raises(RuntimeError) as caught:
        pipeline.delete_node("n1")

    assert "graph down" in str(caught.value) and "vector down" in str(caught.value)
    assert str(caught.value.__cause__) == "graph down"


def test_delete_node_reraises_the_graph_error_when_the_vector_delete_works():
    pipeline = _hybrid(graph_error=RuntimeError("graph down"))

    with pytest.raises(RuntimeError, match="^graph down$"):
        pipeline.delete_node("n1")
    pipeline.vector_store.delete_documents.assert_called_once_with(["n1"])


# --- PL-10: entity content comes from the shape, TOP 1, no made-up text -------------


def test_entity_content_uses_top_one_and_the_shape():
    cursor = RecordingCursor(rows=[])
    cursor.fetchone = lambda: ("some text",)
    pipeline = pipeline_for("legacy_id", cursor)
    pipeline.__class__ = HybridGraphRAGPipeline

    content = pipeline._get_document_content_for_entity("E1")

    sql = cursor.sql
    assert content == "some text"
    assert sql.startswith("SELECT TOP 1 s.content FROM"), sql
    assert "LIMIT" not in sql and "RAG.SourceDocuments" not in sql, sql
    assert "ON e.source_doc_id = s.id" in sql, sql


def test_entity_content_is_none_when_no_document_holds_the_entity():
    cursor = RecordingCursor(rows=[])
    cursor.fetchone = lambda: None
    pipeline = pipeline_for("canonical", cursor)
    pipeline.__class__ = HybridGraphRAGPipeline

    assert pipeline._get_document_content_for_entity("E1") is None


def test_entity_content_raises_on_a_database_error_and_closes_the_cursor():
    cursor = Mock()
    cursor.execute.side_effect = RuntimeError("sql error")
    pipeline = pipeline_for("canonical", RecordingCursor())
    pipeline.__class__ = HybridGraphRAGPipeline
    pipeline.connection_manager.get_connection.return_value.cursor.return_value = cursor

    with pytest.raises(RuntimeError, match="sql error"):
        pipeline._get_document_content_for_entity("E1")
    cursor.close.assert_called_once()


# --- PL-05: load_documents inserts through the DBAPI cursor -------------------------


def _global_pipeline(connection):
    pipeline = IRISGlobalGraphRAGPipeline.__new__(IRISGlobalGraphRAGPipeline)
    pipeline.iris_engine = Mock()
    pipeline.iris_engine.raw_connection.return_value = connection
    return pipeline


def test_global_graphrag_insert_uses_a_dbapi_cursor_and_typed_to_vector():
    connection = Mock()
    pipeline = _global_pipeline(connection)

    pipeline._store_document_sql(Document(id="d1", page_content="hello"), [0.5, 0.25])

    cursor = connection.cursor.return_value
    sql, params = cursor.execute.call_args.args
    assert "TO_VECTOR(?, FLOAT, 2)" in sql
    assert isinstance(params, list) and params[0] == "d1"
    assert params[-1] == "0.5,0.25"
    connection.commit.assert_called_once()
    cursor.close.assert_called_once()
    connection.close.assert_called_once()
    pipeline.iris_engine.connect.assert_not_called()


def test_global_graphrag_insert_rolls_back_and_raises_on_failure():
    connection = Mock()
    connection.cursor.return_value.execute.side_effect = RuntimeError("bad sql")

    with pytest.raises(RuntimeError, match="bad sql"):
        _global_pipeline(connection)._store_document_sql(
            Document(id="d1", page_content="x"), [0.1]
        )
    connection.rollback.assert_called_once()
    connection.close.assert_called_once()


# --- PL-04: ColBERT ingest is atomic per document -----------------------------------


class FakeModel:
    def encode(self, texts, is_query=False):
        return [np.ones((2, 4), dtype=np.float32) for _ in texts]


def test_a_failed_token_insert_rolls_the_document_row_back():
    conn = Mock()
    ingestor = ColBERTIngestor(conn, model=FakeModel(), token_dim=4)
    calls = []
    ingestor._insert_doc = lambda doc: calls.append(("doc", doc["doc_id"]))

    def insert_tokens(doc_id, vecs):
        calls.append(("tokens", doc_id))
        if doc_id == "bad":
            raise RuntimeError("token insert failed")

    ingestor._insert_tokens = insert_tokens

    stats = ingestor.ingest_documents(
        [
            {"doc_id": "good", "text": "a"},
            {"doc_id": "bad", "text": "b"},
            {"doc_id": "good2", "text": "c"},
        ]
    )

    assert stats["docs_failed"] == 1 and stats["docs_ingested"] == 2
    assert stats["total_tokens"] == 4
    assert conn.rollback.call_count == 1
    assert conn.commit.call_count == 2, "each good document commits by itself"
    order = [m[0] for m in conn.method_calls]
    assert order == ["commit", "rollback", "commit"], order


# --- PL-08: drop_tables reports failures and closes its cursor ----------------------


def _schema(errors):
    """`errors` maps a table name to the exception its DROP raises."""
    conn = Mock()
    cursor = conn.cursor.return_value

    def execute(sql):
        for table, error in errors.items():
            if table in sql:
                raise error

    cursor.execute.side_effect = execute
    return ColBERTSchema(conn), conn, cursor


def test_drop_tables_ignores_tables_that_are_already_gone():
    schema, conn, cursor = _schema(
        {"ColBERTCentroids": RuntimeError("Table 'RAG.ColBERTCentroids' not found")}
    )

    schema.drop_tables()

    assert cursor.execute.call_count == 4
    cursor.close.assert_called_once()


def test_drop_tables_raises_on_any_other_failure_after_trying_them_all():
    schema, conn, cursor = _schema({"ColBERTDocuments": RuntimeError("locked")})

    with pytest.raises(RuntimeError, match="ColBERTDocuments.*locked"):
        schema.drop_tables()
    assert cursor.execute.call_count == 4
    cursor.close.assert_called_once()


def test_drop_tables_closes_the_cursor_when_commit_raises():
    schema, conn, cursor = _schema({})
    conn.commit.side_effect = RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        schema.drop_tables()
    cursor.close.assert_called_once()


def test_create_hnsw_index_has_no_unused_token_dim():
    import inspect

    assert (
        "token_dim" not in inspect.signature(ColBERTSchema.create_hnsw_index).parameters
    )


# --- PL-03: HNSW mode does not pay for candidates it discards -----------------------


def _colbert(use_hnsw):
    pipeline = IRISColBERTPipeline.__new__(IRISColBERTPipeline)
    pipeline._model = object()
    pipeline._use_hnsw = use_hnsw
    pipeline._k_per_token = 5
    pipeline._rerank_factor = 3
    pipeline.llm_func = None
    pipeline._encode_query = lambda q: np.ones((1, 4), dtype=np.float32)
    pipeline._fetch_docs_by_ids = lambda ids: [
        Document(id=i, page_content=i) for i in ids
    ]
    pipeline._maxsim = Mock()
    pipeline._maxsim.indb_maxsim.return_value = [("a", 1.0)]
    pipeline._maxsim.bulk_fetch_maxsim.return_value = [("b", 1.0)]
    return pipeline


def test_hnsw_query_skips_the_dense_candidate_retrieval(monkeypatch):
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    parent = Mock(side_effect=AssertionError("dense retrieval must not run"))
    monkeypatch.setattr(BasicRAGPipeline, "query", parent)

    result = _colbert(True).query("q", top_k=2, generate_answer=False)

    assert [d.id for d in result["retrieved_documents"]] == ["a"]
    assert result["metadata"]["initial_candidates"] == 0


def test_bulk_fetch_query_uses_rerank_factor_and_candidate_ids(monkeypatch):
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    captured = {}

    def fake_query(self, query, top_k=5, **kwargs):
        captured["top_k"] = top_k
        return {
            "retrieved_documents": [
                Document(id="x", page_content="text x"),
                Document(id="y", page_content="text y"),
            ]
        }

    monkeypatch.setattr(BasicRAGPipeline, "query", fake_query)
    pipeline = _colbert(False)

    pipeline.query("q", top_k=2, generate_answer=False)

    assert captured["top_k"] == 6, "top_k * rerank_factor"
    ids = pipeline._maxsim.bulk_fetch_maxsim.call_args.args[1]
    assert ids == ["x", "y"], "candidates are matched by document id, not by text"


def test_ingest_turns_autocommit_off_and_restores_it():
    """The driver connection autocommits, so a rollback alone undoes nothing."""

    class Conn:
        autocommit = True

        def __init__(self):
            self.seen = []

        def commit(self):
            self.seen.append(self.autocommit)

        def rollback(self):
            pass

    conn = Conn()
    ingestor = ColBERTIngestor(conn, model=FakeModel(), token_dim=4)
    ingestor._insert_doc = lambda doc: None
    ingestor._insert_tokens = lambda doc_id, vecs: None

    ingestor.ingest_documents([{"doc_id": "a", "text": "x"}])

    assert conn.seen == [False], "documents are written inside a transaction"
    assert conn.autocommit is True, "the caller's mode is restored"
