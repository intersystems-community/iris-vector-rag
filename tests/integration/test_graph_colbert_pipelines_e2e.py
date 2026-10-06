"""End-to-end gate for the GraphRAG and ColBERT stories of pipelines-fix.

Runs on the IRIS the suite provides (`iris_connection` / `iris_target`): an ephemeral
container unless IRIS_CONTAINER_NAME pins one. Everything else is real too: the LLM is
OpenAI's (OPENAI_API_KEY from the environment), the embedder is the real
sentence-transformer and the ColBERT token vectors come from the real ColBERTv2 model.
The one fault injected is a failing token insert, because that failure is the story.
"""

from __future__ import annotations

import numpy as np
import pytest

from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB
from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema
from iris_vector_rag.pipelines.iris_global_graphrag import IRISGlobalGraphRAGPipeline
from tests.support.real_services import real_colbert_encoder, real_llm

pytestmark = pytest.mark.integration

ENTITIES_DDL = """
CREATE TABLE RAG.Entities (
    entity_id VARCHAR(255) NOT NULL, entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100), description VARCHAR(2000), confidence DOUBLE DEFAULT 1.0,
    source_doc_id VARCHAR(255), created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id))
"""
RELATIONSHIPS_DDL = """
CREATE TABLE RAG.EntityRelationships (
    relationship_id VARCHAR(255) NOT NULL, source_entity_id VARCHAR(255) NOT NULL,
    target_entity_id VARCHAR(255) NOT NULL, relationship_type VARCHAR(255),
    weight DOUBLE DEFAULT 1.0, confidence DOUBLE DEFAULT 1.0,
    source_document VARCHAR(255), created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (relationship_id))
"""


def _run(conn, sql, params=None):
    cursor = conn.cursor()
    try:
        cursor.execute(sql, params or [])
        return cursor.fetchall() if cursor.description else None
    finally:
        cursor.close()


def _ensure(conn, ddl):
    try:
        _run(conn, ddl)
        conn.commit()
    except Exception as exc:  # already exists
        conn.rollback()
        if "already exists" not in str(exc).lower() and "-201" not in str(exc):
            raise


# --- GraphRAG story -----------------------------------------------------------------


@pytest.fixture
def graph(iris_connection):
    """A hybrid GraphRAG pipeline over three documents and a star-shaped entity graph."""
    pipeline = create_pipeline(
        "graphrag", validate_requirements=True, llm_func=real_llm()
    )
    _ensure(iris_connection, ENTITIES_DDL)
    _ensure(iris_connection, RELATIONSHIPS_DDL)
    pipeline.load_documents(
        "",
        documents=[
            Document(id=f"pl-doc-{i}", page_content=f"warfarin note {i}", metadata={})
            for i in range(3)
        ],
    )
    _run(iris_connection, "DELETE FROM RAG.EntityRelationships")
    _run(iris_connection, "DELETE FROM RAG.Entities")
    _run(
        iris_connection,
        "INSERT INTO RAG.Entities (entity_id, entity_name, source_doc_id) "
        "VALUES ('pl-hub', 'warfarin', 'pl-doc-0')",
    )
    for i in range(450):
        _run(
            iris_connection,
            "INSERT INTO RAG.Entities (entity_id, entity_name, source_doc_id) VALUES (?, ?, ?)",
            [f"pl-e{i}", f"spoke {i}", "pl-doc-1"],
        )
        _run(
            iris_connection,
            "INSERT INTO RAG.EntityRelationships "
            "(relationship_id, source_entity_id, target_entity_id, relationship_type) "
            "VALUES (?, 'pl-hub', ?, 'related')",
            [f"pl-r{i}", f"pl-e{i}"],
        )
    iris_connection.commit()
    yield pipeline
    _run(iris_connection, "DELETE FROM RAG.EntityRelationships")
    _run(iris_connection, "DELETE FROM RAG.Entities")
    iris_connection.commit()


def test_expansion_reads_a_neighbourhood_larger_than_one_in_list(graph):
    first = graph._expand_neighborhood({"pl-hub"}, depth=1)
    assert len(first) == 451

    # 451 nodes looked up at once: more than one IN list. (IRIS returns ids of this
    # column in upper case, so compare case-insensitively.)
    second = graph._expand_neighborhood(first, depth=1)
    assert {n.upper() for n in second} == {n.upper() for n in first}


def test_seed_entities_are_found_in_the_real_table(graph):
    # The pipeline's own extraction service, over the real LLM, reads the query. A full
    # sentence: the real model returns nothing for the bare phrase "warfarin dosing" on
    # some calls (0 of 5 extractions found an entity), and names the drug every time
    # for a question (5 of 5).
    query = "What is the right dose of warfarin for a patient?"
    assert [e.upper() for e in graph._find_seed_entities(query)] == ["PL-HUB"]


def test_entity_content_comes_from_the_document_or_is_none(graph):
    assert graph._get_document_content_for_entity("pl-hub") == "warfarin note 0"
    assert graph._get_document_content_for_entity("no-such-entity") is None


def test_performance_statistics_count_the_graph(graph):
    stats = graph.get_performance_statistics()
    assert stats["entity_count"] == 451
    assert stats["relationship_count"] == 450


def test_delete_node_removes_the_document(graph, iris_connection):
    assert (
        _run(
            iris_connection,
            "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id = 'pl-doc-2'",
        )[0][0]
        == 1
    )

    graph.delete_node("pl-doc-2")

    assert (
        _run(
            iris_connection,
            "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id = 'pl-doc-2'",
        )[0][0]
        == 0
    )


def test_iris_global_graphrag_load_inserts_a_row(iris_connection, iris_target):
    from tests.support.iris_environment import connect

    _ensure(
        iris_connection,
        "CREATE TABLE paper_content (docid VARCHAR(255), title VARCHAR(1000), "
        "abstract VARCHAR(4000), url VARCHAR(1000), published VARCHAR(100), "
        "authors VARCHAR(1000), combined VARCHAR(5000), paper_vector VECTOR(FLOAT, 4))",
    )

    class Engine:
        """The SQLAlchemy engine's DBAPI door, backed by this test's IRIS."""

        def raw_connection(self):
            return connect(iris_target)

    pipeline = IRISGlobalGraphRAGPipeline.__new__(IRISGlobalGraphRAGPipeline)
    pipeline.iris_engine = Engine()

    pipeline._store_document_sql(
        Document(id="pl-paper", page_content="abstract text", metadata={"title": "T"}),
        [0.1, 0.2, 0.3, 0.4],
    )

    rows = _run(
        iris_connection,
        "SELECT docid, title, VECTOR_COSINE(paper_vector, paper_vector) "
        "FROM paper_content WHERE docid = 'pl-paper'",
    )
    assert rows[0][0] == "pl-paper" and rows[0][1] == "T"
    assert abs(float(rows[0][2]) - 1.0) < 1e-6
    _run(iris_connection, "DELETE FROM paper_content WHERE docid = 'pl-paper'")
    iris_connection.commit()


# --- ColBERT story ------------------------------------------------------------------


@pytest.fixture
def colbert(iris_connection):
    schema = ColBERTSchema(iris_connection)
    schema.drop_tables()
    schema.create_tables()
    yield schema, iris_connection
    schema.drop_tables()


def test_a_failed_token_insert_leaves_no_orphan_document(colbert):
    schema, conn = colbert
    ingestor = ColBERTIngestor(conn, model=real_colbert_encoder())
    # FAULT INJECTION: one document's token insert fails; model, schema and IRIS are real.
    real_insert_tokens = ingestor._insert_tokens

    def insert_tokens(doc_id, vecs):
        if doc_id == "bad":
            raise RuntimeError("injected token failure")
        real_insert_tokens(doc_id, vecs)

    ingestor._insert_tokens = insert_tokens

    stats = ingestor.ingest_documents(
        [
            {"doc_id": "good-1", "text": "alpha"},
            {"doc_id": "bad", "text": "beta"},
            {"doc_id": "good-2", "text": "gamma"},
        ]
    )

    assert stats["docs_failed"] == 1 and stats["docs_ingested"] == 2
    docs = {r[0] for r in _run(conn, "SELECT doc_id FROM RAG.ColBERTDocuments")}
    assert docs == {"good-1", "good-2"}, "no orphan row for the failed document"
    # one token row per token of the two documents that were ingested
    model = real_colbert_encoder()
    expected_tokens = sum(len(v) for v in model.encode(["alpha", "gamma"]))
    assert schema.row_count("RAG.DocumentTokenEmbeddings") == expected_tokens


def test_maxsim_ranks_ingested_documents(colbert):
    schema, conn = colbert
    model = real_colbert_encoder()
    ColBERTIngestor(conn, model=model).ingest_documents(
        [{"doc_id": "d1", "text": "alpha"}, {"doc_id": "d2", "text": "beta"}]
    )
    query = np.array(model.encode(["alpha"])[0], dtype=np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)

    maxsim = MaxSimInDB(conn)
    bulk = maxsim.bulk_fetch_maxsim(query, ["d1", "d2"], top_k=2)
    indb = maxsim.indb_maxsim(query, top_k=2, k_per_token=10)

    assert bulk[0][0] == "d1" and indb[0][0] == "d1"


def test_drop_tables_drops_and_tolerates_tables_already_gone(colbert):
    schema, conn = colbert
    schema.drop_tables()
    assert not schema.table_exists("RAG.ColBERTDocuments")

    schema.drop_tables()  # everything already gone: not an error
