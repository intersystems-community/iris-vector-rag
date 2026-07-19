"""
Integration tests for iris_llm external mode.

All tests in this file are skipped unless:
  1. The iris_llm wheel is installed (checked via importlib.util.find_spec)
  2. SKIP_IRIS_TESTS != "true"
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_IRIS_LLM_AVAILABLE = importlib.util.find_spec("iris_llm") is not None
_SKIP_IRIS = os.environ.get("SKIP_IRIS_TESTS", "false").lower() == "true"

skip_if_no_iris_llm = pytest.mark.skipif(
    not _IRIS_LLM_AVAILABLE,
    reason="iris_llm wheel not installed",
)
skip_if_iris_tests_disabled = pytest.mark.skipif(
    _SKIP_IRIS,
    reason="SKIP_IRIS_TESTS=true",
)


@skip_if_no_iris_llm
def test_get_llm_func_iris_llm_live():
    """get_llm_func(provider='iris_llm') returns a callable that produces a non-empty string."""
    from iris_vector_rag.common.utils import get_llm_func

    fn = get_llm_func(provider="iris_llm", model_name="gpt-4o-mini")
    assert callable(fn)
    # Smoke: invoke — only runs if OPENAI_API_KEY is set
    if os.getenv("OPENAI_API_KEY"):
        result = fn("Say hello in one word.")
        assert isinstance(result, str)
        assert len(result) > 0


@skip_if_no_iris_llm
def test_iris_llm_dspy_adapter_live():
    """IrisLLMDSPyAdapter can be constructed and called with the live wheel."""
    from iris_llm import Provider
    from iris_llm.langchain import ChatIris

    from iris_vector_rag.dspy_modules.iris_llm_lm import IrisLLMDSPyAdapter

    api_key = os.getenv("OPENAI_API_KEY", "")
    provider = Provider.new_openai(api_key=api_key)
    chat = ChatIris(provider=provider, model="gpt-4o-mini")
    adapter = IrisLLMDSPyAdapter(chat_iris=chat)

    assert adapter.provider == "openai"
    assert adapter.model == "iris_llm"

    # Only invoke if key is present
    if api_key:
        result = adapter("Say hello in one word.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)


@skip_if_no_iris_llm
@skip_if_iris_tests_disabled
def test_graphrag_toolset_live_iris():
    """GraphRAGToolSet can be constructed with a live IRIS connection."""
    from iris_devtester import IRISContainer

    from iris_vector_rag.tools import GraphRAGToolSet

    port = IRISContainer.attach("los-iris").get_exposed_port(1972)

    class LiveExecutor:
        def __init__(self):
            import iris as iris_dbapi  # type: ignore[import]

            self._conn = iris_dbapi.createConnection(
                "localhost", port, "USER", "_SYSTEM", "SYS"
            )

        def execute(self, sql, params=None):
            cursor = self._conn.cursor()
            try:
                cursor.execute(sql, params or [])
                if cursor.description is None:
                    return []
                cols = [d[0] for d in cursor.description]
                return [dict(zip(cols, row)) for row in cursor.fetchall()]
            finally:
                cursor.close()

    executor = LiveExecutor()
    toolset = GraphRAGToolSet(executor=executor)

    result = toolset.search_entities("fever")
    import json

    data = json.loads(result)
    assert "entities_found" in data
    assert "entities" in data
