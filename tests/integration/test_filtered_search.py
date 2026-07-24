"""Integration test for US1: filtered search against a real IRIS database (Feature 065).

Verifies SC-001 (100% of results match the filter) end-to-end through
BasicRAGPipeline.query() and IRISVectorStore.

Uses <10 programmatically-loaded documents, so per Constitution Principle II a
`.DAT` fixture is not required for this case (the repo currently ships only the
5-doc `mcp-basic-rag-5docs` DAT fixture, which lacks multiple `source` values).

Skips cleanly when no IRIS instance is reachable.
"""

import uuid

import pytest

from iris_vector_rag.core.models import Document


@pytest.fixture(scope="module")
def basic_pipeline():
    """Construct a real BasicRAGPipeline; skip if IRIS is not reachable."""
    try:
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        pipeline = BasicRAGPipeline()
        # Touch the connection to fail fast if the DB is unavailable.
        pipeline.connection_manager.get_connection()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"IRIS not available for integration test: {exc}")
    return pipeline


@pytest.fixture(scope="module")
def loaded_corpus(basic_pipeline):
    # Clear any stale data from previous runs before loading fresh docs.
    try:
        conn = basic_pipeline.connection_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM RAG.SourceDocuments")
        cursor.close()
    except Exception:
        pass

    tag = uuid.uuid4().hex[:8]
    docs = [
        Document(page_content=f"Insulin regulates blood glucose. [{tag}]", metadata={"source": "pubmed"}),
        Document(page_content=f"Diabetes mellitus affects glucose metabolism. [{tag}]", metadata={"source": "pubmed"}),
        Document(page_content=f"A transformer is a neural network architecture. [{tag}]", metadata={"source": "arxiv"}),
        Document(page_content=f"Attention mechanisms improve sequence models. [{tag}]", metadata={"source": "arxiv"}),
    ]
    basic_pipeline.load_documents(documents=docs)
    return basic_pipeline, tag


def test_metadata_filter_returns_only_matching_source(loaded_corpus):
    """SC-001: with a source filter, 100% of results are from that source."""
    pipeline, _ = loaded_corpus
    result = pipeline.query(
        "glucose regulation",
        top_k=5,
        metadata_filter={"source": "pubmed"},
        generate_answer=False,
    )
    docs = result["retrieved_documents"]
    assert docs, "expected at least one matching document"
    assert all(d.metadata.get("source") == "pubmed" for d in docs), (
        "filter leaked non-matching sources: "
        f"{[d.metadata.get('source') for d in docs]}"
    )


def test_no_filter_returns_mixed_sources(loaded_corpus):
    """Sanity: without a filter, results are not restricted to one source."""
    pipeline, _ = loaded_corpus
    result = pipeline.query("neural network glucose", top_k=5, generate_answer=False)
    assert result["retrieved_documents"], "expected documents without a filter"
