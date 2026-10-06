"""Integration test for US4: hybrid/rrf retrieval modes against live IRIS (Feature 065).

SC-004: hybrid vs rrf produce different ranked sets; weights shift ranking.
FR-011: per-source scores echoed into Document.metadata.
FR-012: prereq error is explicit, never silent fallback.

Everything is real: the session's IRIS, the real embedder, the real vector store and the
real iris-vector-graph `TextSearchEngine`. That engine reads a table `docs(id, text)` of
its own (default table name, no schema) and falls back to `LIKE` when `%FIND` fails, so
each test seeds that table with the rows it wants the text leg to find, keyed by the ids
of the documents the pipeline stored. The text leg therefore returns a known set while
the engine, the SQL and the hydration from the store are the product's own.

PRODUCT DEFECT (reported, not fixed here): the pipeline never creates or fills `docs`,
so on a real deployment the text leg of `hybrid`/`rrf` finds nothing and the result is the
vector leg alone. Run against `RAG.SourceDocuments` the engine cannot work: it addresses
`id` and `text` (the table has `doc_id` and `text_content`), its `%FIND(text, ?)` is not
IRIS SQL (SQLCODE -359, so it always falls back) and `LIKE` on the `LONGVARCHAR` column
matches nothing. `test_text_leg_finds_nothing_without_a_docs_table` pins that.
"""

import uuid
from typing import Dict

import pytest

from iris_vector_rag.core.models import Document

DOCS_TABLE = "docs"


def _execute(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        return cursor.fetchall() if cursor.description else None
    finally:
        cursor.close()


def _clear_source_documents(connection):
    """Empty RAG.SourceDocuments when it exists (a fresh IRIS has no such table yet)."""
    present = _execute(
        connection,
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA='RAG' AND TABLE_NAME='SourceDocuments'",
    )[0][0]
    if present:
        _execute(connection, "DELETE FROM RAG.SourceDocuments")


@pytest.fixture(scope="module")
def basic_pipeline(iris_target):
    """Real BasicRAGPipeline over the session's IRIS, with the real embedder."""
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    pipeline = BasicRAGPipeline()
    pipeline.connection_manager.get_connection()
    return pipeline


@pytest.fixture(scope="module")
def loaded_corpus(basic_pipeline):
    """Load a small mixed corpus into the session's IRIS for retrieval tests."""
    conn = basic_pipeline.connection_manager.get_connection()
    _clear_source_documents(conn)

    tag = uuid.uuid4().hex[:8]
    docs = [
        Document(
            page_content=f"Insulin regulates blood glucose levels. [{tag}]",
            metadata={"source": "bio", "topic": "insulin"},
        ),
        Document(
            page_content=f"Diabetes affects glucose metabolism and insulin response. [{tag}]",
            metadata={"source": "bio", "topic": "diabetes"},
        ),
        Document(
            page_content=f"Neural networks learn from training data. [{tag}]",
            metadata={"source": "cs", "topic": "ml"},
        ),
        Document(
            page_content=f"Transformer architecture uses attention mechanisms. [{tag}]",
            metadata={"source": "cs", "topic": "nlp"},
        ),
        Document(
            page_content=f"Blood sugar regulation involves pancreatic beta cells. [{tag}]",
            metadata={"source": "bio", "topic": "pancreas"},
        ),
    ]
    basic_pipeline.load_documents(documents=docs)
    try:
        yield basic_pipeline, docs
    finally:
        _execute(conn, f"DROP TABLE IF EXISTS {DOCS_TABLE}")
        _clear_source_documents(conn)


def _stored_ids(pipeline) -> Dict[str, str]:
    """page_content (without the tag) -> the doc_id the store gave it."""
    conn = pipeline.connection_manager.get_connection()
    rows = _execute(conn, "SELECT doc_id, text_content FROM RAG.SourceDocuments")
    return {str(text).split(" [")[0]: str(doc_id) for doc_id, text in rows}


def _seed_text_index(pipeline, entries: Dict[str, str]) -> None:
    """Fill the engine's `docs(id, text)` table: stored page_content -> indexed text.

    The engine's LIKE fallback matches the query as a phrase inside `text`, so the
    indexed text decides which documents the text leg returns for a query.
    """
    conn = pipeline.connection_manager.get_connection()
    ids = _stored_ids(pipeline)
    _execute(conn, f"DROP TABLE IF EXISTS {DOCS_TABLE}")
    _execute(conn, f"CREATE TABLE {DOCS_TABLE} (id VARCHAR(255), text VARCHAR(4000))")
    for content, indexed_text in entries.items():
        _execute(
            conn,
            f"INSERT INTO {DOCS_TABLE} (id, text) VALUES (?, ?)",
            [ids[content], indexed_text],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_vector_mode_returns_documents(loaded_corpus):
    """Baseline: vector mode retrieves documents from real IRIS."""
    pipeline, _ = loaded_corpus
    result = pipeline.query(
        "glucose regulation", top_k=3, retrieval="vector", generate_answer=False
    )
    docs = result["retrieved_documents"]
    assert docs, "vector mode must return at least one document"
    assert all(hasattr(d, "page_content") for d in docs)


def test_rrf_metadata_contains_per_source_scores(loaded_corpus):
    """FR-011: rrf mode echoes vector_score, text_score, fusion_score into each doc.metadata."""
    pipeline, _ = loaded_corpus

    # The text leg finds one document, which overlaps the vector results
    _seed_text_index(
        pipeline,
        {"Insulin regulates blood glucose levels.": "glucose insulin"},
    )

    result = pipeline.query(
        "glucose insulin", top_k=3, retrieval="rrf", generate_answer=False
    )

    docs = result["retrieved_documents"]
    assert docs, "rrf mode must return documents"
    # At least the first doc should have fusion metadata
    first = docs[0]
    assert (
        "fusion_score" in first.metadata
    ), "rrf must set fusion_score in doc.metadata (FR-011)"
    # vector_score or text_score must appear on docs that came from both legs
    assert (
        "vector_score" in first.metadata or "text_score" in first.metadata
    ), "rrf must record per-source contribution scores in metadata (FR-011)"
    # the text leg really ran: the seeded hit carries the text engine's score
    assert any(
        "text_score" in d.metadata for d in docs
    ), "the real text engine's hit must appear with a text_score"


def test_hybrid_metadata_contains_per_source_scores(loaded_corpus):
    """FR-011: hybrid mode echoes per-source scores."""
    pipeline, _ = loaded_corpus

    _seed_text_index(
        pipeline,
        {
            "Diabetes affects glucose metabolism and insulin response.": "glucose diabetes"
        },
    )

    result = pipeline.query(
        "glucose diabetes",
        top_k=3,
        retrieval="hybrid",
        weights={"vector": 0.7, "text": 0.3},
        generate_answer=False,
    )

    docs = result["retrieved_documents"]
    assert docs, "hybrid mode must return documents"
    first = docs[0]
    assert "fusion_score" in first.metadata


def test_hybrid_vs_rrf_produce_different_rankings(loaded_corpus):
    """SC-004: hybrid and rrf rankings differ when text leg has divergent results.

    The text index is seeded so the text leg returns a doc that is NOT top-ranked by
    vector, making RRF (rank-only) vs weighted-score fusion produce different top docs.
    """
    pipeline, _ = loaded_corpus

    # The text leg finds only the CS/NLP doc, which vector ranks lower for "glucose"
    _seed_text_index(
        pipeline,
        {"Transformer architecture uses attention mechanisms.": "glucose neural"},
    )

    rrf_result = pipeline.query(
        "glucose neural", top_k=3, retrieval="rrf", generate_answer=False
    )

    hybrid_result = pipeline.query(
        "glucose neural",
        top_k=3,
        retrieval="hybrid",
        weights={"vector": 0.9, "text": 0.1},
        generate_answer=False,
    )

    rrf_docs = rrf_result["retrieved_documents"]
    hybrid_docs = hybrid_result["retrieved_documents"]

    assert rrf_docs, "rrf must return documents"
    assert hybrid_docs, "hybrid must return documents"

    # Ordering may differ: extract top doc page_content sequences
    rrf_order = [d.page_content for d in rrf_docs]
    hybrid_order = [d.page_content for d in hybrid_docs]

    # At least one position should differ (if all docs same, that's a degenerate corpus)
    # Accept identical ordering only if top_k=1 (single doc, no permutation possible)
    if len(rrf_docs) > 1 and len(hybrid_docs) > 1:
        assert rrf_order != hybrid_order, (
            "hybrid and rrf should produce different rankings when text leg diverges from vector "
            "(SC-004). If this fails, check that weights are being applied in weighted fusion."
        )


def test_weights_shift_hybrid_ranking(loaded_corpus):
    """SC-004: increasing text weight promotes text-only docs in hybrid mode."""
    pipeline, _ = loaded_corpus

    # The text leg finds a doc that vector would rank last (CS topic for a bio query)
    cs_doc = Document(page_content="Neural networks learn from training data.")
    _seed_text_index(pipeline, {cs_doc.page_content: "glucose"})

    # High vector weight → bio docs dominate
    high_vec = pipeline.query(
        "glucose",
        top_k=3,
        retrieval="hybrid",
        weights={"vector": 0.95, "text": 0.05},
        generate_answer=False,
    )

    # High text weight → cs doc should climb
    high_text = pipeline.query(
        "glucose",
        top_k=3,
        retrieval="hybrid",
        weights={"vector": 0.1, "text": 0.9},
        generate_answer=False,
    )

    high_vec_tops = [
        d.page_content.split(" [")[0] for d in high_vec["retrieved_documents"]
    ]
    high_text_tops = [
        d.page_content.split(" [")[0] for d in high_text["retrieved_documents"]
    ]

    assert high_vec_tops or high_text_tops, "at least one mode must return docs"

    # CS doc is the lone text hit: under a high text weight it must be retrieved and must
    # rank at least as high as under a high vector weight
    cs_content = cs_doc.page_content
    assert (
        cs_content in high_text_tops
    ), f"the lone text hit must be retrieved under a 0.9 text weight: {high_text_tops}"
    text_rank = high_text_tops.index(cs_content)
    vec_rank = (
        high_vec_tops.index(cs_content)
        if cs_content in high_vec_tops
        else len(high_vec_tops)
    )
    assert text_rank <= vec_rank, (
        f"CS doc should rank higher under high text weight (rank {text_rank}) "
        f"than high vector weight (rank {vec_rank}). weights parameter not shifting ranking."
    )


def test_unknown_mode_returns_empty_with_warning(loaded_corpus, caplog):
    """FR-012: unknown mode surfaces a clear error message (not silent).

    The pipeline catches and logs the ValueError from RetrievalEngine so that
    callers receive an empty-result response rather than an unhandled exception.
    The error must be named — the mode name must appear in the log output.
    """
    import logging

    pipeline, _ = loaded_corpus
    with caplog.at_level(logging.WARNING, logger="iris_vector_rag.pipelines.basic"):
        result = pipeline.query(
            "test", top_k=1, retrieval="bogus_mode", generate_answer=False
        )

    # Empty results (not a crash)
    assert (
        result["retrieved_documents"] == []
    ), "unknown mode must return empty docs, not crash"

    # Error is named — the mode name appears in the warning
    assert any(
        "bogus_mode" in record.message for record in caplog.records
    ), f"mode name must appear in logged warning. Got: {[r.message for r in caplog.records]}"


def test_text_leg_finds_nothing_without_a_docs_table(loaded_corpus):
    """PRODUCT DEFECT: the text leg has no index of its own, so rrf is the vector leg alone.

    The engine reads `docs(id, text)`, which nothing in the product creates; against
    `RAG.SourceDocuments` it finds no row whatever the query (see the module docstring).
    When the text index exists, the same query does return text hits (the tests above).
    """
    pipeline, _ = loaded_corpus
    conn = pipeline.connection_manager.get_connection()
    _execute(conn, f"DROP TABLE IF EXISTS {DOCS_TABLE}")

    result = pipeline.query("glucose", top_k=5, retrieval="rrf", generate_answer=False)

    docs = result["retrieved_documents"]
    assert docs, "the vector leg still answers"
    assert not any(
        "text_score" in d.metadata for d in docs
    ), "without a docs table no document can carry a text score"
