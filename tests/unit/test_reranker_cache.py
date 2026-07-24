"""US6 unit tests: reranker cache — single load across N queries, separate configs cached
separately, thread-safe first-load (T035 — TDD, must fail before T037).

FR-015: Reranker models are loaded once per (strategy, model_name) tuple per process.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache():
    from iris_vector_rag.retrieval import rerank
    rerank._RERANKER_CACHE.clear()


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Clear the reranker cache before and after every test in this module."""
    _clear_cache()
    yield
    _clear_cache()


def _make_mock_build(scores=None):
    """Return a (counting_build, call_count) pair — patches _build_cross_encoder_reranker."""
    scores = scores or [0.9, 0.5, 0.3]
    call_count = {"n": 0, "models": []}

    def counting_build(model_name):
        call_count["n"] += 1
        call_count["models"].append(model_name)
        fn = MagicMock()
        fn.return_value = scores
        return fn

    return counting_build, call_count


_PATCH = "iris_vector_rag.retrieval.rerank._build_cross_encoder_reranker"


# ---------------------------------------------------------------------------
# FR-015 / SC-005: model loaded ONCE per (strategy, model) config
# ---------------------------------------------------------------------------

class TestSingleLoadAcrossQueries:
    def test_resolve_reranker_true_loads_model_once(self):
        """resolve_reranker(True) called N times → _build called once."""
        _clear_cache()
        build, cnt = _make_mock_build()
        with patch(_PATCH, side_effect=build):
            from iris_vector_rag.retrieval.rerank import resolve_reranker
            r1 = resolve_reranker(True)
            r2 = resolve_reranker(True)
            r3 = resolve_reranker(True)

        assert r1 is r2 is r3, "Same callable must be returned from cache"
        assert cnt["n"] == 1, (
            f"_build_cross_encoder_reranker must be called exactly once; got {cnt['n']}"
        )

    def test_resolve_reranker_returns_same_callable(self):
        """Cache returns same callable object on repeated calls."""
        _clear_cache()
        build, _ = _make_mock_build()
        with patch(_PATCH, side_effect=build):
            from iris_vector_rag.retrieval.rerank import resolve_reranker
            r1 = resolve_reranker(True)
            r2 = resolve_reranker(True)

        assert r1 is r2

    def test_resolve_reranker_string_strategy_cached(self):
        """resolve_reranker('cross-encoder') also caches."""
        _clear_cache()
        build, cnt = _make_mock_build()
        with patch(_PATCH, side_effect=build):
            from iris_vector_rag.retrieval.rerank import resolve_reranker
            r1 = resolve_reranker("cross-encoder")
            r2 = resolve_reranker("cross-encoder")

        assert r1 is r2
        assert cnt["n"] == 1


# ---------------------------------------------------------------------------
# Separate configs cached separately
# ---------------------------------------------------------------------------

class TestSeparateConfigsCachedSeparately:
    def test_different_model_names_get_different_cache_entries(self):
        """Two different model names → two _build calls."""
        _clear_cache()
        build, cnt = _make_mock_build()
        with patch(_PATCH, side_effect=build):
            from iris_vector_rag.retrieval.rerank import resolve_reranker
            r1 = resolve_reranker(True, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            r2 = resolve_reranker(True, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

        assert r1 is not r2, "Different model names must produce different reranker instances"
        assert cnt["n"] == 2, (
            f"Expected 2 _build calls; got {cnt['n']}"
        )

    def test_same_model_name_reuses_cached_instance(self):
        """Same model name → single _build call."""
        _clear_cache()
        build, cnt = _make_mock_build()
        with patch(_PATCH, side_effect=build):
            from iris_vector_rag.retrieval.rerank import resolve_reranker
            r1 = resolve_reranker(True, model_name="my-model")
            r2 = resolve_reranker(True, model_name="my-model")

        assert r1 is r2
        assert cnt["n"] == 1

    def test_true_and_default_model_name_share_cache(self):
        """resolve_reranker(True) uses default model; same as passing default explicitly."""
        _clear_cache()
        from iris_vector_rag.retrieval.rerank import _DEFAULT_MODEL
        build, cnt = _make_mock_build()
        with patch(_PATCH, side_effect=build):
            from iris_vector_rag.retrieval.rerank import resolve_reranker
            r1 = resolve_reranker(True)
            r2 = resolve_reranker(True, model_name=_DEFAULT_MODEL)

        assert r1 is r2
        assert cnt["n"] == 1


# ---------------------------------------------------------------------------
# Thread-safety: first-load under concurrent callers
# ---------------------------------------------------------------------------

class TestThreadSafeFirstLoad:
    def test_concurrent_calls_load_model_once(self):
        """N threads calling resolve_reranker(True) simultaneously → 1 instantiation."""
        _clear_cache()
        call_count = {"n": 0}
        original_build = None

        def counting_build(model_name):
            call_count["n"] += 1
            fn = MagicMock()
            fn.return_value = []
            return fn

        with patch("iris_vector_rag.retrieval.rerank._build_cross_encoder_reranker", side_effect=counting_build):
            from iris_vector_rag.retrieval.rerank import resolve_reranker

            results = []
            errors = []

            def worker():
                try:
                    results.append(resolve_reranker(True))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Threads raised exceptions: {errors}"
        assert len(results) == 10, "All threads must get a result"
        # All threads must get the same callable
        assert len(set(id(r) for r in results)) == 1, (
            "All concurrent calls must return the same cached callable"
        )
        assert call_count["n"] == 1, (
            f"CrossEncoder must be instantiated exactly once; got {call_count['n']}"
        )


# ---------------------------------------------------------------------------
# Callables are never cached (FR-015 spec)
# ---------------------------------------------------------------------------

class TestCallablesUncached:
    def test_callable_returned_directly_not_cached(self):
        """Passing a callable to resolve_reranker returns it directly without caching."""
        _clear_cache()
        my_fn = lambda q, docs: docs
        from iris_vector_rag.retrieval.rerank import resolve_reranker, _RERANKER_CACHE
        result = resolve_reranker(my_fn)
        assert result is my_fn
        # No cache entry created for callable
        assert len(_RERANKER_CACHE) == 0, "Callables must not be cached"
