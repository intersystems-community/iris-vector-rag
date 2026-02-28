"""
Unit tests for iris_llm substrate: get_llm_func(provider='iris_llm') and IrisLLMDSPyAdapter.

All tests mock iris_llm at the sys.modules level — no wheel required.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_iris_llm():
    """Return a fake iris_llm module tree sufficient for all tests."""
    mock_iris_llm = MagicMock()
    mock_provider_instance = MagicMock()
    mock_iris_llm.Provider.new_openai.return_value = mock_provider_instance

    mock_chat_response = MagicMock()
    mock_chat_response.content = "hello from iris_llm"
    mock_chat_iris_instance = MagicMock()
    mock_chat_iris_instance.invoke.return_value = mock_chat_response

    mock_iris_llm_langchain = MagicMock()
    mock_iris_llm_langchain.ChatIris.return_value = mock_chat_iris_instance

    return mock_iris_llm, mock_iris_llm_langchain, mock_chat_iris_instance


# ---------------------------------------------------------------------------
# get_llm_func(provider="iris_llm")
# ---------------------------------------------------------------------------

def test_get_llm_func_iris_llm_provider():
    """get_llm_func(provider='iris_llm') returns a callable when wheel is present."""
    mock_iris_llm, mock_iris_llm_langchain, _ = _make_mock_iris_llm()
    mock_lc = MagicMock()

    # Reset the module-level LLM cache
    import iris_vector_rag.common.utils as utils_mod
    utils_mod._llm_instance = None
    utils_mod._current_llm_key = None

    with patch.dict(sys.modules, {
        "iris_llm": mock_iris_llm,
        "iris_llm.langchain": mock_iris_llm_langchain,
        "langchain_core.messages": mock_lc,
    }), patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        from iris_vector_rag.common.utils import get_llm_func
        fn = get_llm_func(provider="iris_llm", model_name="gpt-4o-mini")

    assert callable(fn)


def test_get_llm_func_iris_llm_missing_raises_import_error():
    """get_llm_func(provider='iris_llm') without wheel raises ImportError, not ValueError."""
    import iris_vector_rag.common.utils as utils_mod
    utils_mod._llm_instance = None
    utils_mod._current_llm_key = None

    with patch.dict(sys.modules, {"iris_llm": None, "iris_llm.langchain": None}):
        from iris_vector_rag.common.utils import get_llm_func
        with pytest.raises(ImportError, match="iris_llm"):
            get_llm_func(provider="iris_llm")


def test_get_llm_func_iris_llm_missing_key_raises_value_error():
    """get_llm_func(provider='iris_llm') with no API key and no base_url raises ValueError."""
    mock_iris_llm, mock_iris_llm_langchain, _ = _make_mock_iris_llm()

    import iris_vector_rag.common.utils as utils_mod
    utils_mod._llm_instance = None
    utils_mod._current_llm_key = None

    import os
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with patch.dict(sys.modules, {
            "iris_llm": mock_iris_llm,
            "iris_llm.langchain": mock_iris_llm_langchain,
        }):
            from iris_vector_rag.common.utils import get_llm_func
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_llm_func(provider="iris_llm")
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


# ---------------------------------------------------------------------------
# get_llm_func_for_embedded
# ---------------------------------------------------------------------------

def test_get_llm_func_for_embedded_falls_back_to_stub():
    """When iris_llm is absent, get_llm_func_for_embedded returns a stub callable."""
    with patch.dict(sys.modules, {"iris_llm": None, "iris_llm.langchain": None}):
        from iris_vector_rag.common.utils import get_llm_func_for_embedded
        fn = get_llm_func_for_embedded()

    assert callable(fn)
    result = fn("hello")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# IrisLLMDSPyAdapter
# ---------------------------------------------------------------------------

def test_iris_llm_dspy_adapter_attributes():
    """IrisLLMDSPyAdapter sets provider='openai', kwargs dict, and model attribute."""
    mock_chat = MagicMock()

    with patch.dict(sys.modules, {"dspy": MagicMock()}):
        from iris_vector_rag.dspy_modules.iris_llm_lm import IrisLLMDSPyAdapter
        adapter = IrisLLMDSPyAdapter(chat_iris=mock_chat, model="gpt-4o-mini")

    assert adapter.provider == "openai"
    assert "model" in adapter.kwargs
    assert adapter.model == "gpt-4o-mini"


def test_iris_llm_dspy_adapter_call():
    """IrisLLMDSPyAdapter.__call__ returns list[str] with response content."""
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "The answer is 42"
    mock_chat.invoke.return_value = mock_response

    mock_lc = MagicMock()

    with patch.dict(sys.modules, {
        "dspy": MagicMock(),
        "langchain_core.messages": mock_lc,
    }):
        from iris_vector_rag.dspy_modules.iris_llm_lm import IrisLLMDSPyAdapter
        adapter = IrisLLMDSPyAdapter(chat_iris=mock_chat)
        result = adapter("What is the answer?")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "The answer is 42"


def test_iris_llm_dspy_adapter_basic_request_delegates():
    """basic_request delegates to __call__ and returns list[str]."""
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "delegated"
    mock_chat.invoke.return_value = mock_response

    with patch.dict(sys.modules, {
        "dspy": MagicMock(),
        "langchain_core.messages": MagicMock(),
    }):
        from iris_vector_rag.dspy_modules.iris_llm_lm import IrisLLMDSPyAdapter
        adapter = IrisLLMDSPyAdapter(chat_iris=mock_chat)
        result = adapter.basic_request("hello")

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# iris_globals
# ---------------------------------------------------------------------------

def test_iris_globals_no_iris_module():
    """gset and gget are no-ops when iris is not installed — never raise."""
    with patch.dict(sys.modules, {"iris": None}):
        import importlib
        import iris_vector_rag.common.iris_globals as ig
        importlib.reload(ig)

        ig.gset("IVR", "Test", value="hello")      # must not raise
        result = ig.gget("IVR", "Test")
        assert result is None


# ---------------------------------------------------------------------------
# Core import isolation
# ---------------------------------------------------------------------------

def test_import_iris_vector_rag_without_iris_llm():
    """Core package imports cleanly without iris_llm present."""
    with patch.dict(sys.modules, {"iris_llm": None, "iris_llm.langchain": None}):
        import importlib
        import iris_vector_rag
        importlib.reload(iris_vector_rag)
        assert hasattr(iris_vector_rag, "SqlExecutor")
