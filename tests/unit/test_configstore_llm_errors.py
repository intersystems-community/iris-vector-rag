"""Every way a ConfigStore entry fails is one LLMConfigError, at build (spec 093, US2).

The server's own messages, as the probe captured them on 2026.3.0AI.126 and 2026.1, are
replayed by the fake driver; each error must name the entry and say what to do.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.common.configstore_llm import configstore_llm
from iris_vector_rag.exceptions import LLMConfigError
from tests.unit.test_configstore_llm import lab  # noqa: F401  (fixture)
from tests.unit.test_configstore_llm import ENTRY, FakeConnectionManager, FakeNative

NO_ENTRY = (
    "ERROR #5809: Object to Load not found, class '%ConfigStore.Configuration', "
    "ID 'AI.LLM.nosuch'"
)
NO_WALLET = (
    "ERROR #26220: Error using secret ivrprobe.openai. > ERROR #822: Access Denied"
)
NO_NATIVE = "<PROTECT> *Resource %Native_ClassExecution required"
NO_CONFIGSTORE = "<CLASS DOES NOT EXIST> *%ConfigStore.Configuration"


def build(name: str = "ivrtest") -> LLMConfigError:
    with pytest.raises(LLMConfigError) as caught:
        configstore_llm(name, FakeConnectionManager())
    err = caught.value
    assert err.entry == name
    assert f"AI.LLM.{name}" in str(err)
    assert err.fix
    assert "sk-probe-123" not in str(err)
    return err


@pytest.mark.parametrize("name", ["ops.review", "1ops", "", "ops review", "ops/x"])
def test_invalid_name_is_refused_before_any_call(lab, name):
    state = lab()

    err = build(name)

    assert "^[A-Za-z][A-Za-z0-9_-]*$" in str(err)
    assert state["iris"].connections == []


@pytest.mark.parametrize("name", ["opsreview", "Ops-Review_2", "a"])
def test_valid_names_are_queried(lab, name):
    state = lab()

    configstore_llm(name, FakeConnectionManager())

    assert state["native"].calls[0][2][0] == f"AI.LLM.{name}"


def test_missing_entry(lab):
    lab(FakeNative(get_status=NO_ENTRY))

    err = build("nosuch")

    assert "does not exist" in str(err)


def test_other_get_failure_keeps_the_server_text(lab):
    lab(FakeNative(get_status="ERROR #5001: something else"))

    err = build()

    assert "ERROR #5001: something else" in str(err)


def test_no_wallet_privilege_names_the_reference_and_the_privilege(lab):
    state = lab(FakeNative(wallet_status=NO_WALLET))

    err = build()

    assert "secret://ivrprobe.openai#api_key" in str(err)
    assert "%Admin_Wallet:USE" in str(err)
    assert "READ" in str(err)
    # the reference came from the unresolved copy, CopyDetails(det, 0)
    assert ("invoke", "CopyDetails", (0,)) in state["native"].calls


def test_no_native_class_execution_privilege(lab):
    lab(FakeNative(raises=RuntimeError(NO_NATIVE)))

    err = build()

    assert "%Native_ClassExecution:USE" in str(err)


@pytest.mark.parametrize("key", ["model", "model_provider"])
def test_missing_required_key(lab, key):
    stored = {k: v for k, v in ENTRY.items() if k != key}
    lab(FakeNative(stored, {**stored, "api_key": "sk-probe-123"}))

    err = build()

    assert f"`{key}`" in str(err)


def test_no_configstore_on_the_server_gives_the_version(lab):
    lab(FakeNative(raises=RuntimeError(NO_CONFIGSTORE), version="2026.1.0.240.0"))

    err = build()

    assert "2026.3" in str(err)
    assert "2026.1.0.240.0" in str(err)


@pytest.mark.parametrize("enabled", [0, "0", False])
def test_disabled_entry(lab, enabled):
    state = lab(FakeNative(enabled=enabled))

    err = build()

    assert "disabled" in str(err)
    # refused before the Wallet is touched
    assert ("invoke", "CopyDetails", ()) not in state["native"].calls


def test_missing_provider_package_names_it(lab, monkeypatch):
    from iris_vector_rag.common import configstore_llm as module

    lab()

    def init_chat_model(**kwargs):
        raise ImportError(
            "Unable to import langchain_anthropic. Please install with "
            "`pip install -U langchain-anthropic`"
        )

    monkeypatch.setattr(module, "_init_chat_model", lambda: init_chat_model)

    err = build()

    assert "langchain-anthropic" in str(err)


def test_unresolved_reference_is_never_handed_to_the_model(lab):
    state = lab(
        FakeNative(ENTRY, ENTRY)
    )  # read "succeeds" but the key is still a reference

    err = build()

    assert "secret://ivrprobe.openai#api_key" in str(err)
    assert "%Admin_Wallet:USE" in str(err)
    assert state["chats"] == []


def test_unresolved_reference_nested_in_a_setting(lab):
    stored = {**ENTRY, "default_headers": {"x-key": "secret://ivrprobe.gw#token"}}
    lab(FakeNative(stored, {**stored, "api_key": "sk-probe-123"}))

    err = build()

    assert "secret://ivrprobe.gw#token" in str(err)


def test_no_connection(lab):
    lab()
    with pytest.raises(LLMConfigError) as caught:
        configstore_llm("ivrtest", FakeConnectionManager(connection=None))

    assert "IRIS connection" in str(caught.value)
