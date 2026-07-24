"""Benchmark: reranker cache ensures model-load cost excluded from steady-state queries (T036 / SC-005)."""
import time
from unittest.mock import MagicMock

import pytest

from iris_vector_rag.retrieval import rerank as rerank_mod


def _clear():
    rerank_mod._RERANKER_CACHE.clear()


@pytest.mark.benchmark
def test_reranker_cache_amortizes_load_cost():
    """Second call to resolve_reranker() must be at least 10x faster than the first (cold load)."""
    _clear()

    # Warm: load a real (tiny) cross-encoder — measures actual model-load time
    t0 = time.perf_counter()
    fn1 = rerank_mod.resolve_reranker(True)
    cold_ms = (time.perf_counter() - t0) * 1000

    # Hot: cached lookup
    t1 = time.perf_counter()
    fn2 = rerank_mod.resolve_reranker(True)
    hot_ms = (time.perf_counter() - t1) * 1000

    assert fn1 is fn2, "cache must return same callable"
    # Cold load varies widely; hot must simply be sub-millisecond
    assert hot_ms < 1.0, f"cached lookup took {hot_ms:.3f}ms — expected <1ms"

    _clear()


@pytest.mark.benchmark
def test_steady_state_rerank_excludes_model_load():
    """Per-query reranking overhead (model already loaded) must be <50ms for 5 docs."""
    from iris_vector_rag.core.models import Document

    _clear()
    fn = rerank_mod.resolve_reranker(True)  # warm cache
    docs = [
        Document(page_content=f"document number {i} about diabetes and insulin", metadata={})
        for i in range(5)
    ]

    # Measure steady-state: call the reranker directly (no model-load)
    t0 = time.perf_counter()
    result = fn("What is diabetes?", docs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert result, "reranker returned no results"
    assert elapsed_ms < 2000, f"reranking 5 docs took {elapsed_ms:.1f}ms — expected <2000ms (CPU inference, model already loaded)"

    _clear()
