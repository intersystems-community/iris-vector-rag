"""BasicRAGRerankingPipeline: a failed rerank is not reported as reranked (PL-01),
and an injected llm_func needs no OPENAI_API_KEY (PL-02)."""

from __future__ import annotations

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline


class FakeConfig:
    def get(self, key, default=None):
        return default


def _docs(n=4):
    return [
        Document(id=f"d{i}", page_content=f"text {i}", metadata={}) for i in range(n)
    ]


def _pipeline(monkeypatch, reranker, llm_func=None):
    """A rerank pipeline whose retrieval is replaced by a fixed candidate list."""
    pipeline = BasicRAGRerankingPipeline.__new__(BasicRAGRerankingPipeline)
    from iris_vector_rag.pipelines.basic_rerank import Reranker

    pipeline.rerank_factor = 2
    pipeline.reranker = Reranker(reranker)
    pipeline.reranker_func = reranker
    pipeline.llm_func = llm_func
    pipeline.config_manager = FakeConfig()

    def fake_parent_query(self, query, top_k=5, **kwargs):
        return {"retrieved_documents": _docs(), "execution_time": 0.0}

    monkeypatch.setattr(
        "iris_vector_rag.pipelines.basic.BasicRAGPipeline.query", fake_parent_query
    )
    return pipeline


def _failing_reranker(query, docs):
    raise RuntimeError("cross-encoder exploded")


def _reverse_reranker(query, docs):
    return [(d, float(i)) for i, d in enumerate(docs)]


def test_rerank_failure_raises_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = _pipeline(monkeypatch, _failing_reranker)
    with pytest.raises(RuntimeError, match="cross-encoder exploded"):
        pipeline.query("q", top_k=2, generate_answer=False)


def test_rerank_failure_opt_in_fallback_is_flagged(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = _pipeline(monkeypatch, _failing_reranker)
    result = pipeline.query(
        "q", top_k=2, generate_answer=False, allow_rerank_fallback=True
    )
    meta = result["metadata"]
    assert meta["reranked"] is False
    assert meta["retrieval_method"] != "rerank"
    assert "cross-encoder exploded" in meta["rerank_error"]
    assert [d.id for d in result["retrieved_documents"]] == ["d0", "d1"]


def test_successful_rerank_is_flagged_and_has_no_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = _pipeline(monkeypatch, _reverse_reranker)
    result = pipeline.query("q", top_k=2, generate_answer=False)
    assert result["metadata"]["reranked"] is True
    assert result["metadata"]["retrieval_method"] == "rerank"
    assert "rerank_error" not in result["metadata"]
    assert [d.id for d in result["retrieved_documents"]] == ["d3", "d2"]


def test_injected_llm_needs_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = _pipeline(
        monkeypatch, _reverse_reranker, llm_func=lambda p: "local answer"
    )
    monkeypatch.setattr(
        pipeline, "_generate_answer", lambda q, d, c=None: "local answer", raising=False
    )
    result = pipeline.query("q", top_k=2)
    assert result["answer"] == "local answer"


def test_no_llm_and_no_key_still_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = _pipeline(monkeypatch, _reverse_reranker, llm_func=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        pipeline.query("q", top_k=2)
