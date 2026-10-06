"""Integration test for US4: hybrid/rrf retrieval modes against live IRIS (Feature 065).

SC-004: hybrid vs rrf produce different ranked sets; weights shift ranking.
FR-011: per-source scores echoed into Document.metadata.
FR-012: prereq error is explicit, never silent fallback.

Everything is real: the session's IRIS, the real embedder, the real vector store and the
real text leg. The text leg is the document table's own full-text index
(`SchemaManager.ensure_text_index`, created when the pipeline first stores a document),
queried with `%FIND`. Nothing is seeded or faked: which documents the text leg returns
is decided by the words in the stored documents, and the tests choose queries whose
keyword hits are known. The missing-index error and the per-table index checks are in
`test_text_search_iris.py`.

History: until this feature the text leg read an iris-vector-graph `docs(id, text)` table
that nothing created, so on a real deployment `hybrid` and `rrf` were the vector leg alone.
"""

import uuid

import pytest

from iris_vector_rag.core.models import Document

#: The one stored document that contains the word "zephyrquill". Vector search cannot
#: know that word; only the text leg can bring this document in for it.
KEYWORD_ONLY_CONTENT = "Neural networks learn from training data, see zephyrquill."


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
            page_content=f"{KEYWORD_ONLY_CONTENT} [{tag}]",
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
        _clear_source_documents(conn)


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

    # "glucose" and "insulin" are words of stored documents, so the text leg finds them
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
    # the text leg really ran: a hit found by keyword carries the index's score
    assert any(
        "text_score" in d.metadata for d in docs
    ), "a document found by the text leg must appear with a text_score"


def test_hybrid_metadata_contains_per_source_scores(loaded_corpus):
    """FR-011: hybrid mode echoes per-source scores."""
    pipeline, _ = loaded_corpus

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
    assert any(
        "text_score" in d.metadata for d in docs
    ), "hybrid must carry the text leg's score on a document the index found"


def test_hybrid_vs_rrf_produce_different_rankings(loaded_corpus):
    """SC-004: hybrid and rrf rankings differ when text leg has divergent results.

    The keyword "zephyrquill" is in one CS document only, which vector ranks low for a
    glucose query but the text leg ranks first (the rarest word), so RRF (rank-only) and
    weighted-score fusion produce different top docs.
    """
    pipeline, _ = loaded_corpus
    query = "glucose zephyrquill"

    rrf_result = pipeline.query(query, top_k=3, retrieval="rrf", generate_answer=False)

    hybrid_result = pipeline.query(
        query,
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

    # The text leg finds a doc that vector would rank last (CS topic for a bio query):
    # the one stored document that contains "zephyrquill"
    cs_doc = Document(page_content=KEYWORD_ONLY_CONTENT)
    query = "glucose zephyrquill"

    # High vector weight → bio docs dominate
    high_vec = pipeline.query(
        query,
        top_k=3,
        retrieval="hybrid",
        weights={"vector": 0.95, "text": 0.05},
        generate_answer=False,
    )

    # High text weight → cs doc should climb
    high_text = pipeline.query(
        query,
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


def test_a_document_findable_only_by_keyword_is_returned_by_text_rrf_and_hybrid(
    loaded_corpus,
):
    """The regression this feature fixes: the text leg contributes on a real deployment.

    "zephyrquill" is a word of exactly one stored document and means nothing to the
    embedder. Before the text index existed, `rrf`/`hybrid` were the vector leg alone,
    so that document came back only by luck. Now each mode returns it through the text
    leg, and it carries the index's score.
    """
    pipeline, _ = loaded_corpus
    tagged = lambda d: d.page_content.split(" [")[0]  # noqa: E731

    text = pipeline.query(
        "zephyrquill", top_k=3, retrieval="text", generate_answer=False
    )
    assert [tagged(d) for d in text["retrieved_documents"]] == [KEYWORD_ONLY_CONTENT]
    assert text["retrieved_documents"][0].metadata["text_score"] > 0

    for mode, extra in (
        ("rrf", {}),
        ("hybrid", {"weights": {"vector": 0.7, "text": 0.3}}),
    ):
        result = pipeline.query(
            "zephyrquill", top_k=3, retrieval=mode, generate_answer=False, **extra
        )
        docs = {tagged(d): d for d in result["retrieved_documents"]}
        assert KEYWORD_ONLY_CONTENT in docs, f"{mode} dropped the keyword-only document"
        hit = docs[KEYWORD_ONLY_CONTENT]
        assert "text_score" in hit.metadata, f"{mode}: the text leg did not contribute"
        assert hit.metadata["fusion_score"] > 0


def test_text_mode_honours_metadata_filters(loaded_corpus):
    pipeline, _ = loaded_corpus
    for mode in ("text", "rrf", "hybrid"):
        kwargs = {"weights": {"vector": 0.5, "text": 0.5}} if mode == "hybrid" else {}
        result = pipeline.query(
            "insulin glucose zephyrquill",
            top_k=5,
            retrieval=mode,
            metadata_filter={"source": "bio"},
            generate_answer=False,
            **kwargs,
        )
        docs = result["retrieved_documents"]
        assert docs, mode
        assert {d.metadata["source"] for d in docs} == {"bio"}, mode
        assert not any(KEYWORD_ONLY_CONTENT in d.page_content for d in docs), mode


def test_a_missing_text_index_fails_loudly_through_the_pipeline(loaded_corpus):
    """No silent vector-only answer: without the index, text-leg modes raise."""
    from iris_vector_rag.exceptions import TextIndexMissingError
    from iris_vector_rag.storage.text_search import TEXT_INDEX_NAME

    pipeline, _ = loaded_corpus
    conn = pipeline.connection_manager.get_connection()
    _execute(conn, f"DROP INDEX {TEXT_INDEX_NAME} ON RAG.SourceDocuments")
    try:
        for mode in ("text", "rrf", "hybrid"):
            with pytest.raises(TextIndexMissingError, match="ensure_text_index"):
                pipeline.query(
                    "glucose", top_k=3, retrieval=mode, generate_answer=False
                )
        # the vector leg alone is unaffected
        assert pipeline.query(
            "glucose", top_k=3, retrieval="vector", generate_answer=False
        )["retrieved_documents"]
    finally:
        # the explicit method an existing deployment uses to add the index
        pipeline.vector_store.schema_manager.ensure_text_index("SourceDocuments")
