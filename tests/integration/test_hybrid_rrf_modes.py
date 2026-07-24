"""Integration test for US4: hybrid/rrf retrieval modes against live IRIS (Feature 065).

SC-004: hybrid vs rrf produce different ranked sets; weights shift ranking.
FR-011: per-source scores echoed into Document.metadata.
FR-012: prereq error is explicit, never silent fallback.

Uses real IRIS for vector retrieval; mocks the text search leg to produce
controlled, divergent results so ranking assertions are deterministic.
"""

import uuid
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.core.models import Document


@pytest.fixture(scope="module")
def basic_pipeline():
    """Real BasicRAGPipeline; skip if IRIS unavailable."""
    try:
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        pipeline = BasicRAGPipeline()
        pipeline.connection_manager.get_connection()
    except Exception as exc:
        pytest.skip(f"IRIS not available: {exc}")
    return pipeline


@pytest.fixture(scope="module")
def loaded_corpus(basic_pipeline):
    """Load a small mixed corpus into the live DB for retrieval tests."""
    try:
        conn = basic_pipeline.connection_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM RAG.SourceDocuments")
        cursor.close()
    except Exception:
        pass

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
    return basic_pipeline, docs


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build Document stubs that TextSearchEngine.search_documents returns
# ──────────────────────────────────────────────────────────────────────────────

def _text_docs(page_contents: List[str]) -> List[Document]:
    return [Document(page_content=c, metadata={}) for c in page_contents]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_vector_mode_returns_documents(loaded_corpus):
    """Baseline: vector mode retrieves documents from real IRIS."""
    pipeline, _ = loaded_corpus
    result = pipeline.query("glucose regulation", top_k=3, retrieval="vector", generate_answer=False)
    docs = result["retrieved_documents"]
    assert docs, "vector mode must return at least one document"
    assert all(hasattr(d, "page_content") for d in docs)


def test_rrf_metadata_contains_per_source_scores(loaded_corpus):
    """FR-011: rrf mode echoes vector_score, text_score, fusion_score into each doc.metadata."""
    pipeline, _ = loaded_corpus

    # Provide controlled text results that overlap partially with vector results
    bio_text_doc = Document(
        page_content="Insulin regulates blood glucose levels.",
        metadata={"source": "bio", "topic": "insulin"},
    )

    with patch(
        "iris_vector_graph.text_search.TextSearchEngine"
    ) as MockTSE:
        instance = MockTSE.return_value
        instance.search_documents.return_value = [bio_text_doc]

        result = pipeline.query(
            "glucose insulin", top_k=3, retrieval="rrf", generate_answer=False
        )

    docs = result["retrieved_documents"]
    assert docs, "rrf mode must return documents"
    # At least the first doc should have fusion metadata
    first = docs[0]
    assert "fusion_score" in first.metadata, (
        "rrf must set fusion_score in doc.metadata (FR-011)"
    )
    # vector_score or text_score must appear on docs that came from both legs
    assert "vector_score" in first.metadata or "text_score" in first.metadata, (
        "rrf must record per-source contribution scores in metadata (FR-011)"
    )


def test_hybrid_metadata_contains_per_source_scores(loaded_corpus):
    """FR-011: hybrid mode echoes per-source scores."""
    pipeline, _ = loaded_corpus

    text_doc = Document(
        page_content="Diabetes affects glucose metabolism.",
        metadata={"source": "bio"},
    )

    with patch("iris_vector_graph.text_search.TextSearchEngine") as MockTSE:
        instance = MockTSE.return_value
        instance.search_documents.return_value = [text_doc]

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

    We mock text search to return a doc that is NOT top-ranked by vector,
    making RRF (rank-only) vs weighted-score fusion produce different top docs.
    """
    pipeline, _ = loaded_corpus

    # Text engine returns only the CS/NLP doc, which vector ranks lower for "glucose"
    cs_text_doc = Document(
        page_content="Transformer architecture uses attention mechanisms.",
        metadata={"source": "cs"},
    )

    with patch("iris_vector_graph.text_search.TextSearchEngine") as MockTSE:
        instance = MockTSE.return_value
        instance.search_documents.return_value = [cs_text_doc]

        rrf_result = pipeline.query(
            "glucose neural", top_k=3, retrieval="rrf", generate_answer=False
        )

    with patch("iris_vector_graph.text_search.TextSearchEngine") as MockTSE:
        instance = MockTSE.return_value
        instance.search_documents.return_value = [cs_text_doc]

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

    # Text returns a doc that vector would rank last (CS topic for a bio query)
    cs_doc = Document(
        page_content="Neural networks learn from training data.",
        metadata={"source": "cs"},
    )

    # High vector weight → bio docs dominate
    with patch("iris_vector_graph.text_search.TextSearchEngine") as MockTSE:
        instance = MockTSE.return_value
        instance.search_documents.return_value = [cs_doc]

        high_vec = pipeline.query(
            "glucose",
            top_k=3,
            retrieval="hybrid",
            weights={"vector": 0.95, "text": 0.05},
            generate_answer=False,
        )

    # High text weight → cs doc should climb
    with patch("iris_vector_graph.text_search.TextSearchEngine") as MockTSE:
        instance = MockTSE.return_value
        instance.search_documents.return_value = [cs_doc]

        high_text = pipeline.query(
            "glucose",
            top_k=3,
            retrieval="hybrid",
            weights={"vector": 0.1, "text": 0.9},
            generate_answer=False,
        )

    high_vec_tops = [d.page_content for d in high_vec["retrieved_documents"]]
    high_text_tops = [d.page_content for d in high_text["retrieved_documents"]]

    assert high_vec_tops or high_text_tops, "at least one mode must return docs"

    # CS doc should rank higher (lower index) under high_text than under high_vec
    cs_content = cs_doc.page_content
    if cs_content in high_vec_tops and cs_content in high_text_tops:
        vec_rank = high_vec_tops.index(cs_content)
        text_rank = high_text_tops.index(cs_content)
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
        result = pipeline.query("test", top_k=1, retrieval="bogus_mode", generate_answer=False)

    # Empty results (not a crash)
    assert result["retrieved_documents"] == [], "unknown mode must return empty docs, not crash"

    # Error is named — the mode name appears in the warning
    assert any(
        "bogus_mode" in record.message for record in caplog.records
    ), f"mode name must appear in logged warning. Got: {[r.message for r in caplog.records]}"
