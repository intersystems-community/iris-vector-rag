"""The LLM factory bounds every call: a timeout and a retry count reach the
provider client, and a timeout surfaces as a typed error naming the provider,
the model and the limit. The client is a fake at the import boundary; no network.
"""

from __future__ import annotations

import sys
import types

import httpx
import openai
import pytest

import iris_vector_rag.common.utils as utils_mod
from iris_vector_rag.exceptions import GenerationError, LLMTimeoutError


class FakeChatOpenAI:
    """Records its constructor arguments; `invoke` raises whatever `raises` holds."""

    created: list = []
    raises: BaseException | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeChatOpenAI.created.append(self)

    def invoke(self, prompt):
        if FakeChatOpenAI.raises is not None:
            raise FakeChatOpenAI.raises
        return types.SimpleNamespace(content=f"ok:{prompt}")


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    FakeChatOpenAI.created = []
    FakeChatOpenAI.raises = None
    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("RAG_LLM__TIMEOUT", raising=False)
    monkeypatch.delenv("RAG_LLM__MAX_RETRIES", raising=False)
    monkeypatch.setattr(
        utils_mod, "_setup_caching_if_needed", lambda enable=None: False
    )
    utils_mod._llm_instances.clear()
    yield
    utils_mod._llm_instances.clear()


def _client() -> FakeChatOpenAI:
    assert len(FakeChatOpenAI.created) == 1
    return FakeChatOpenAI.created[0]


def test_defaults_are_sixty_seconds_and_two_retries():
    utils_mod.get_llm_func("openai", "m")
    assert _client().kwargs["timeout"] == 60.0
    assert _client().kwargs["max_retries"] == 2


def test_environment_overrides_the_defaults(monkeypatch):
    monkeypatch.setenv("RAG_LLM__TIMEOUT", "7.5")
    monkeypatch.setenv("RAG_LLM__MAX_RETRIES", "0")
    utils_mod.get_llm_func("openai", "m")
    assert _client().kwargs["timeout"] == 7.5
    assert _client().kwargs["max_retries"] == 0


def test_explicit_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv("RAG_LLM__TIMEOUT", "7.5")
    monkeypatch.setenv("RAG_LLM__MAX_RETRIES", "5")
    utils_mod.get_llm_func("openai", "m", timeout=3, max_retries=1)
    assert _client().kwargs["timeout"] == 3.0
    assert _client().kwargs["max_retries"] == 1


def test_other_limits_get_their_own_client():
    utils_mod.get_llm_func("openai", "m", timeout=5)
    utils_mod.get_llm_func("openai", "m", timeout=9)
    assert [c.kwargs["timeout"] for c in FakeChatOpenAI.created] == [5.0, 9.0]


@pytest.mark.parametrize("bad", [0, -1, "soon", True, float("nan")])
def test_a_bad_timeout_is_refused(bad):
    with pytest.raises(ValueError, match="timeout"):
        utils_mod.get_llm_func("openai", "m", timeout=bad)


@pytest.mark.parametrize("bad", [-1, 1.5, "two", True])
def test_bad_retries_are_refused(bad):
    with pytest.raises(ValueError, match="max_retries"):
        utils_mod.get_llm_func("openai", "m", max_retries=bad)


def test_limits_come_from_the_llm_config_section():
    class Config:
        def __init__(self, values):
            self.values = values

        def get(self, key, default=None):
            return self.values.get(key, default)

    got = utils_mod.llm_limits_from_config(
        Config({"llm.timeout": 12, "llm.max_retries": 3})
    )
    assert got == {"timeout": 12, "max_retries": 3}
    assert utils_mod.llm_limits_from_config(Config({})) == {}


@pytest.mark.parametrize(
    "failure",
    [
        openai.APITimeoutError(request=httpx.Request("POST", "https://x.invalid")),
        httpx.ReadTimeout("read timed out"),
        TimeoutError("timed out"),
    ],
    ids=["openai", "httpx", "builtin"],
)
def test_a_timeout_surfaces_as_a_typed_error(failure):
    FakeChatOpenAI.raises = failure
    llm = utils_mod.get_llm_func("openai", "gpt-x", timeout=4)
    with pytest.raises(LLMTimeoutError) as caught:
        llm("hello")
    err = caught.value
    assert err.provider == "openai"
    assert err.model_name == "gpt-x"
    assert err.timeout == 4.0
    assert "openai" in str(err) and "4" in str(err)
    assert err.__cause__ is failure
    assert isinstance(err, GenerationError)


def test_a_timeout_wrapped_by_the_sdk_is_still_found():
    inner = httpx.ConnectTimeout("connect timed out")
    outer = RuntimeError("request failed")
    outer.__cause__ = inner
    FakeChatOpenAI.raises = outer
    with pytest.raises(LLMTimeoutError):
        utils_mod.get_llm_func("openai", "m")("hello")


def test_other_failures_propagate_unchanged():
    boom = openai.APIConnectionError(request=httpx.Request("POST", "https://x.invalid"))
    FakeChatOpenAI.raises = boom
    with pytest.raises(openai.APIConnectionError) as caught:
        utils_mod.get_llm_func("openai", "m")("hello")
    assert caught.value is boom


def test_a_good_call_returns_the_text():
    assert utils_mod.get_llm_func("openai", "m")("hello") == "ok:hello"
