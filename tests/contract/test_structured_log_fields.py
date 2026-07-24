"""T047: structured log fields emitted by all pipelines (Principle VII).

Verifies that each pipeline emits INFO-level log records containing
retrieval_mode, rerank_strategy, and rerank_degraded after query().
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.core.models import Document

STUB_DOCS = [Document(id="1", page_content="test doc", metadata={})]


def _capture_logs(logger_name: str):
    """Return a list that collects formatted log records from logger_name."""
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = _Handler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger(logger_name)
    log.addHandler(handler)
    old_level = log.level
    log.setLevel(logging.DEBUG)
    return records, handler, log, old_level


def _cleanup(log, handler, old_level):
    log.removeHandler(handler)
    log.setLevel(old_level)


class TestBasicPipelineStructuredLog:
    def test_completion_log_has_retrieval_mode(self):
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        with patch.object(BasicRAGPipeline, "__init__", lambda self, *a, **kw: None):
            p = BasicRAGPipeline.__new__(BasicRAGPipeline)
        p.connection_manager = MagicMock()
        p.config_manager = MagicMock()
        p.config_manager.get = MagicMock(side_effect=lambda k, d=None: d)
        p.vector_store = MagicMock()
        p.vector_store.search_by_text.return_value = list(STUB_DOCS)
        p.vector_store.similarity_search.return_value = list(STUB_DOCS)
        p.logger = logging.getLogger("iris_vector_rag.pipelines.basic")
        p.llm_func = None
        p.embedding_manager = MagicMock()
        p.embedding_config = None
        p.use_iris_embedding = False
        p.pipeline_config = {}
        p.chunk_size = 1000
        p.chunk_overlap = 200
        p.default_top_k = 5

        records, handler, log, old_level = _capture_logs("iris_vector_rag.pipelines.basic")
        try:
            p.query(query="test", top_k=3, generate_answer=False, rerank=False)
        finally:
            _cleanup(log, handler, old_level)

        completion = [r for r in records if "RAG query completed" in r]
        assert completion, f"No completion log found. Records: {records}"
        msg = completion[-1]
        assert "retrieval_mode=" in msg, f"retrieval_mode missing: {msg}"
        assert "rerank_strategy=" in msg, f"rerank_strategy missing: {msg}"
        assert "rerank_degraded=" in msg, f"rerank_degraded missing: {msg}"

    def test_debug_log_has_retrieval_mode(self):
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        with patch.object(BasicRAGPipeline, "__init__", lambda self, *a, **kw: None):
            p = BasicRAGPipeline.__new__(BasicRAGPipeline)
        p.connection_manager = MagicMock()
        p.config_manager = MagicMock()
        p.config_manager.get = MagicMock(side_effect=lambda k, d=None: d)
        p.vector_store = MagicMock()
        p.vector_store.search_by_text.return_value = list(STUB_DOCS)
        p.vector_store.similarity_search.return_value = list(STUB_DOCS)
        p.logger = logging.getLogger("iris_vector_rag.pipelines.basic")
        p.llm_func = None
        p.embedding_manager = MagicMock()
        p.embedding_config = None
        p.use_iris_embedding = False
        p.pipeline_config = {}
        p.chunk_size = 1000
        p.chunk_overlap = 200
        p.default_top_k = 5

        records, handler, log, old_level = _capture_logs("iris_vector_rag.pipelines.basic")
        try:
            p.query(query="test", top_k=3, generate_answer=False, retrieval="rrf")
        finally:
            _cleanup(log, handler, old_level)

        debug = [r for r in records if "BasicRAG retrieval" in r]
        assert debug, f"No debug retrieval log. Records: {records}"
        msg = debug[-1]
        assert "retrieval_mode=" in msg, f"retrieval_mode missing in debug: {msg}"
        assert "rerank=" in msg, f"rerank missing in debug: {msg}"
        assert "weights=" in msg, f"weights missing in debug: {msg}"
