"""get_llm_func caches by every argument that shapes the client (OS-23) and
create_pipeline hands it only LLM arguments (OS-10).

The OpenAI client is replaced by a fake at the import boundary; no network.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import iris_vector_rag.common.utils as utils_mod


class FakeChatOpenAI:
    """Records its constructor arguments; rejects unknown ones like the real client."""

    created: list = []
    ALLOWED = {
        "model_name",
        "openai_api_key",
        "temperature",
        "max_tokens",
        "timeout",
        "max_retries",
    }

    def __init__(self, **kwargs):
        unknown = set(kwargs) - self.ALLOWED
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
        self.kwargs = kwargs
        FakeChatOpenAI.created.append(self)

    def invoke(self, prompt):
        return types.SimpleNamespace(
            content=f"{self.kwargs.get('temperature')}:{prompt}"
        )


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    FakeChatOpenAI.created = []
    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        utils_mod, "_setup_caching_if_needed", lambda enable=None: False
    )
    utils_mod._llm_instances.clear()
    yield
    utils_mod._llm_instances.clear()


def test_different_kwargs_get_different_clients():
    cold = utils_mod.get_llm_func("openai", "m", temperature=0.0)
    hot = utils_mod.get_llm_func("openai", "m", temperature=0.9)
    assert len(FakeChatOpenAI.created) == 2
    assert cold("x") == "0.0:x"
    assert hot("x") == "0.9:x"


def test_earlier_function_keeps_its_own_client():
    """A later call with other arguments must not change what an earlier one calls."""
    first = utils_mod.get_llm_func("openai", "m", temperature=0.1)
    utils_mod.get_llm_func("openai", "m", temperature=0.7)
    assert first("p") == "0.1:p"


def test_same_arguments_reuse_the_client():
    utils_mod.get_llm_func("openai", "m", temperature=0.3)
    utils_mod.get_llm_func("openai", "m", temperature=0.3)
    assert len(FakeChatOpenAI.created) == 1


def test_kwarg_order_does_not_split_the_cache():
    utils_mod.get_llm_func("openai", "m", temperature=0.3, max_tokens=5)
    utils_mod.get_llm_func("openai", "m", max_tokens=5, temperature=0.3)
    assert len(FakeChatOpenAI.created) == 1


def test_create_pipeline_passes_only_llm_arguments():
    """create_pipeline("basic", embedding_func=f) must not reach the OpenAI client."""
    import iris_vector_rag as pkg

    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return lambda prompt: "ok"

    config = MagicMock()
    config.get.side_effect = lambda key, default=None: default
    with (
        patch.object(pkg, "get_llm_func", side_effect=spy),
        patch.object(pkg, "ConfigurationManager", return_value=config),
        patch.object(pkg, "ConnectionManager"),
        patch.object(pkg, "_create_pipeline_legacy") as legacy,
    ):
        pkg.create_pipeline(
            "basic",
            embedding_func=lambda t: [0.0],
            validate_requirements=False,
            some_pipeline_option=1,
        )
    assert set(captured) <= {"provider", "model_name"}, captured
    # pipeline kwargs still reach the pipeline
    assert "embedding_func" in legacy.call_args.kwargs
    assert legacy.call_args.kwargs["some_pipeline_option"] == 1


def test_create_pipeline_with_engine_passes_only_llm_arguments():
    import iris_vector_rag as pkg

    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return lambda prompt: "ok"

    config = MagicMock()
    config.get.side_effect = lambda key, default=None: default
    engine = MagicMock(config_manager=config)
    with (
        patch.object(pkg, "get_llm_func", side_effect=spy),
        patch.object(pkg, "_create_pipeline_legacy"),
    ):
        pkg.create_pipeline(
            "basic",
            engine=engine,
            embedding_func=lambda t: [0.0],
            validate_requirements=False,
        )
    assert set(captured) <= {"provider", "model_name"}, captured
