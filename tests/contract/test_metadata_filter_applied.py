"""Contract tests for US1: filtered search actually filters (Feature 065).

Covers FR-001 (metadata_filter forwarded), FR-002 (similarity_threshold applied),
FR-003 (invalid filter key raises a clear error, not silently ignored).

These are hermetic contract tests: the vector store is a fake that records how
BasicRAGPipeline.query() calls it, so no live IRIS database is required. SQL-level
injection safety is exercised at the integration layer (test_filtered_search.py);
here we assert the pipeline forwards filter values verbatim to the store.
"""

import pytest
from unittest.mock import MagicMock

from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import VectorStoreConfigurationError


class _FakeVectorStore:
    """Records similarity_search calls and simulates store-side filtering."""

    def __init__(self, docs, bad_keys=("nonexistent_key",)):
        self.docs = docs
        self.bad_keys = set(bad_keys)
        self.calls = []

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        self.calls.append({"query": query, "k": k, "filter": filter})
        if filter:
            for key in filter:
                if key in self.bad_keys:
                    raise VectorStoreConfigurationError(
                        f"Invalid filter key: '{key}' is not an allowed metadata field"
                    )
        results = self.docs
        if filter:
            results = [
                d
                for d in results
                if all(str(d.metadata.get(fk)) == str(fv) for fk, fv in filter.items())
            ]
        return list(results)[:k]


def _doc(content, source, score):
    return Document(
        page_content=content,
        metadata={"source": source, "score": score, "similarity": score},
    )


@pytest.fixture
def pipeline(monkeypatch):
    # Keep construction hermetic: no real embedding model, no DB.
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.basic.EmbeddingManager", lambda *a, **k: MagicMock()
    )
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    store = _FakeVectorStore(
        docs=[
            _doc("alpha doc from A", "A", 0.92),
            _doc("beta doc from B", "B", 0.55),
            _doc("gamma doc from A", "A", 0.31),
        ]
    )
    p = BasicRAGPipeline(
        connection_manager=MagicMock(),
        config_manager=MagicMock(),
        llm_func=None,
        vector_store=store,
    )
    return p, store


def test_metadata_filter_is_forwarded_to_store(pipeline):
    """FR-001: a supplied metadata_filter reaches the vector store."""
    p, store = pipeline
    result = p.query("q", top_k=5, metadata_filter={"source": "A"}, generate_answer=False)

    assert store.calls[-1]["filter"] == {"source": "A"}
    assert result["retrieved_documents"], "expected matching documents"
    assert all(d.metadata["source"] == "A" for d in result["retrieved_documents"])


def test_no_filter_forwards_none(pipeline):
    """FR-013 backward-compat: omitting the filter forwards None (prior behavior)."""
    p, store = pipeline
    p.query("q", top_k=5, generate_answer=False)
    assert store.calls[-1]["filter"] is None


def test_similarity_threshold_excludes_low_scores(pipeline):
    """FR-002: similarity_threshold drops documents below the cutoff."""
    p, _ = pipeline
    result = p.query("q", top_k=5, similarity_threshold=0.6, generate_answer=False)

    scores = [d.metadata["score"] for d in result["retrieved_documents"]]
    assert scores, "expected at least one document above threshold"
    assert all(s >= 0.6 for s in scores)
    # Only the 0.92 doc clears 0.6 in the fixture set.
    assert len(result["retrieved_documents"]) == 1


def test_invalid_filter_key_raises_clear_error(pipeline):
    """FR-003: an invalid filter key raises, rather than being silently ignored."""
    p, _ = pipeline
    with pytest.raises(VectorStoreConfigurationError):
        p.query("q", top_k=5, metadata_filter={"nonexistent_key": "x"}, generate_answer=False)


def test_filter_value_forwarded_verbatim(pipeline):
    """The pipeline must not mangle filter values (store owns escaping/parameterization)."""
    p, store = pipeline
    injection = "a' OR '1'='1"
    p.query("q", top_k=5, metadata_filter={"source": injection}, generate_answer=False)
    assert store.calls[-1]["filter"] == {"source": injection}
