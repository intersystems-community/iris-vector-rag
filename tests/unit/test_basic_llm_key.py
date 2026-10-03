"""BasicRAGPipeline needs OPENAI_API_KEY only when it has no llm_func (PL-02)."""

from __future__ import annotations

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.basic import BasicRAGPipeline


class FakeStore:
    def search_by_text(self, query, top_k=5, metadata_filter=None):
        return [Document(id="a", page_content="alpha", metadata={})]


def _pipeline(llm_func):
    pipeline = BasicRAGPipeline.__new__(BasicRAGPipeline)
    pipeline.llm_func = llm_func
    pipeline.vector_store = FakeStore()
    pipeline.use_iris_embedding = False
    pipeline.embedding_config = None
    pipeline.default_top_k = 5
    pipeline._ensure_initialized = lambda: None
    pipeline._generate_answer = lambda q, d, c=None: pipeline.llm_func("p")
    return pipeline


def test_injected_llm_runs_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = _pipeline(lambda p: "local answer").query("what", top_k=1)
    assert result["answer"] == "local answer"


def test_no_llm_no_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _pipeline(None).query("what", top_k=1)


def test_no_llm_with_key_returns_documents_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = _pipeline(None).query("what", top_k=1)
    assert "No LLM function" in result["answer"]
