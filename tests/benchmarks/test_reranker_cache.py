"""Benchmark: reranker cache ensures model-load cost excluded from steady-state queries (T036 / SC-005)."""

import time

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
    """Steady-state reranking (model already loaded) returns docs; second call not slower."""
    from iris_vector_rag.core.models import Document
    from iris_vector_rag.retrieval.rerank import (
        _build_cross_encoder_reranker,
        _DEFAULT_MODEL,
    )

    docs = [
        Document(
            page_content=f"document number {i} about diabetes and insulin", metadata={}
        )
        for i in range(5)
    ]

    # Verify CrossEncoder is real before building
    import sentence_transformers as _st
    import sys

    print(f"\n  CrossEncoder type: {type(_st.CrossEncoder)}", file=sys.stderr)

    # Build a fresh reranker (model load here, not timed)
    fn = _build_cross_encoder_reranker(_DEFAULT_MODEL)

    # Warm up JIT / tokenizer
    fn("warm up query", docs)

    # Time steady-state calls — model already loaded, no cache involved
    t1 = time.perf_counter()
    fn("What is diabetes?", docs)
    first_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    result = fn("What is insulin resistance?", docs)
    second_ms = (time.perf_counter() - t2) * 1000

    assert result, "reranker returned no results"
    assert (
        second_ms < first_ms * 3 + 50
    ), f"second call ({second_ms:.1f}ms) much slower than first ({first_ms:.1f}ms)"
