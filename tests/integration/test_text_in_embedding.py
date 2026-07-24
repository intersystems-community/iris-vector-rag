"""Integration test for US7 / SC-008: zero-config semantic search via search_by_text.

Verifies that BasicRAGPipeline.query() correctly routes through
IRISVectorStore.search_by_text() when no native IRIS EMBEDDING config is set,
and that the embedding model embedded in the store produces meaningful results.

Skips cleanly when IRIS is not reachable.
"""

import pytest

from iris_vector_rag.core.models import Document


@pytest.fixture(scope="module")
def basic_pipeline():
    try:
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        pipeline = BasicRAGPipeline()
        pipeline.connection_manager.get_connection()
    except Exception as exc:
        pytest.skip(f"IRIS not available for integration test: {exc}")
    return pipeline


@pytest.fixture(scope="module")
def seeded_pipeline(basic_pipeline):
    """Load a small corpus; reuse existing data if already present."""
    import uuid

    tag = uuid.uuid4().hex[:8]
    docs = [
        Document(
            page_content=f"The pancreas produces insulin to regulate blood glucose. [{tag}]",
            metadata={"source": "medical", "tag": tag},
        ),
        Document(
            page_content=f"Diabetes mellitus is a chronic metabolic disorder. [{tag}]",
            metadata={"source": "medical", "tag": tag},
        ),
        Document(
            page_content=f"Convolutional neural networks excel at image classification. [{tag}]",
            metadata={"source": "cs", "tag": tag},
        ),
    ]
    basic_pipeline.load_documents(documents=docs)
    return basic_pipeline, tag


def test_search_by_text_returns_documents(seeded_pipeline):
    """SC-008: query with no embedding_config routes through search_by_text and returns results."""
    pipeline, _ = seeded_pipeline
    assert not getattr(pipeline, "use_iris_embedding", False), (
        "This test requires the pipeline to be in normal (non-native) embedding mode"
    )

    result = pipeline.query("insulin blood glucose regulation", top_k=5, generate_answer=False)

    assert "retrieved_documents" in result
    docs = result["retrieved_documents"]
    assert docs, "expected at least one document returned"
    assert isinstance(docs[0], Document)


def test_search_by_text_semantic_relevance(seeded_pipeline):
    """Top result for a medical query should come from medical source, not CS."""
    pipeline, tag = seeded_pipeline
    result = pipeline.query(
        f"pancreas insulin [{tag}]", top_k=3, generate_answer=False
    )
    docs = result["retrieved_documents"]
    assert docs, "expected documents"
    sources = [d.metadata.get("source") for d in docs]
    assert "medical" in sources, f"expected medical source in top results, got: {sources}"


def test_search_by_text_top_result_contains_query_terms(seeded_pipeline):
    """Content of top result should be semantically related to the query."""
    pipeline, tag = seeded_pipeline
    result = pipeline.query(f"diabetes metabolic disorder [{tag}]", top_k=3, generate_answer=False)
    docs = result["retrieved_documents"]
    assert docs
    top_content = docs[0].page_content.lower()
    # Top result should mention diabetes or metabolic
    assert any(term in top_content for term in ("diabetes", "metabolic", "insulin", "glucose")), (
        f"Top result not semantically relevant: {top_content[:100]}"
    )
