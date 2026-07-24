"""Benchmark: composable layer overhead must be <5ms when no options are set (Principle VI)."""
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_pipeline():
    """Return a mock BasicRAGPipeline that records timing without IRIS."""
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    cm = MagicMock()
    cfg = MagicMock()
    vs = MagicMock()
    vs.similarity_search.return_value = []
    with patch.object(BasicRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = BasicRAGPipeline.__new__(BasicRAGPipeline)
        p.connection_manager = cm
        p.config_manager = cfg
        p.vector_store = vs
        p.logger = MagicMock()
        return p


@pytest.mark.benchmark
def test_composable_overhead_under_5ms():
    """Normalize path (no retrieval/rerank/weights) must add <5ms."""
    from iris_vector_rag.core.query_options import normalize_query_params

    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        normalize_query_params(query="test query", top_k=5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 5.0, f"normalize_query_params avg {avg_ms:.3f}ms exceeds 5ms budget"
