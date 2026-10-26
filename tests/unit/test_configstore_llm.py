"""Reading the answer model from an AI Hub ConfigStore entry (spec 093, US1).

The fakes stand in for the released driver's Native API, as the probe found it on
2026.3.0AI.126: ``createIRIS(conn).classMethodValue("%ConfigStore.Configuration", "Get",
"AI.LLM.<name>", ref)`` and ``cfg.invoke("CopyDetails", det[, 0])``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from iris_vector_rag.common import configstore_llm as module
from iris_vector_rag.common.configstore_llm import configstore_llm, read_llm_entry

ENTRY = {
    "model": "gpt-4o-mini",
    "model_provider": "openai",
    "temperature": 0.1,
    "api_key": "secret://ivrprobe.openai#api_key",
}
RESOLVED = {**ENTRY, "api_key": "sk-probe-123"}
OK = 1


class FakeReference:
    def __init__(self, value: Any = None):
        self.value = value

    def getValue(self) -> Any:
        return self.value

    def setValue(self, value: Any) -> None:
        self.value = value


class FakeDynamicObject:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def invoke(self, method: str, *args: Any) -> Any:
        assert method == "%ToJSON"
        return json.dumps(self.data)


class FakeConfiguration:
    """An `%ConfigStore.Configuration` instance: `Enabled` and `CopyDetails`."""

    def __init__(self, native: "FakeNative"):
        self.native = native

    def get(self, prop: str) -> Any:
        assert prop == "Enabled"
        return self.native.enabled

    def invoke(self, method: str, *args: Any) -> Any:
        self.native.calls.append(("invoke", method, args[1:]))
        assert method == "CopyDetails"
        resolve = not (len(args) > 1 and args[1] == 0)
        if resolve and self.native.wallet_status != OK:
            return self.native.wallet_status
        args[0].setValue(
            FakeDynamicObject(self.native.resolved if resolve else self.native.stored)
        )
        return OK


class FakeNative:
    """The `IRIS` object `createIRIS` returns, scripted per test."""

    def __init__(
        self,
        stored: Optional[Dict[str, Any]] = None,
        resolved: Optional[Dict[str, Any]] = None,
        *,
        get_status: Any = OK,
        wallet_status: Any = OK,
        enabled: Any = 1,
        raises: Optional[Exception] = None,
        version: str = "2026.3.0AI.126.0",
    ):
        self.stored = ENTRY if stored is None else stored
        self.resolved = (
            (RESOLVED if stored is None else stored) if resolved is None else resolved
        )
        self.get_status = get_status
        self.wallet_status = wallet_status
        self.enabled = enabled
        self.raises = raises
        self.version = version
        self.calls: List[tuple] = []

    def classMethodValue(self, cls: str, method: str, *args: Any) -> Any:
        self.calls.append((cls, method, args))
        if cls == "%SYSTEM.Status" and method == "GetErrorText":
            return args[0]
        if cls == "%SYSTEM.Version" and method == "GetNumber":
            return self.version
        if self.raises is not None:
            raise self.raises
        assert (cls, method) == ("%ConfigStore.Configuration", "Get")
        if self.get_status != OK:
            return self.get_status
        args[1].setValue(FakeConfiguration(self))
        return OK

    def methods(self) -> List[tuple]:
        return [(c[0], c[1]) for c in self.calls]


class FakeIris:
    """The `iris` module, as far as the reader uses it."""

    def __init__(self, native: FakeNative):
        self.native = native
        self.connections: List[Any] = []

    def createIRIS(self, connection: Any) -> FakeNative:
        self.connections.append(connection)
        return self.native

    @staticmethod
    def IRISReference(value: Any) -> FakeReference:
        return FakeReference(value)


class FakeConnectionManager:
    def __init__(self, connection: Any = "the-pipeline-connection"):
        self.connection = connection
        self.asked = 0

    def get_connection(self, backend_name: str = "iris") -> Any:
        self.asked += 1
        return self.connection


class FakeChat:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.prompts: List[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return type("Message", (), {"content": f"answer to {prompt}"})()


@pytest.fixture
def lab(monkeypatch):
    """Install a fake driver and a fake `init_chat_model`; return their recorders."""

    def install(native: Optional[FakeNative] = None) -> Dict[str, Any]:
        state: Dict[str, Any] = {"native": native or FakeNative(), "chats": []}
        state["iris"] = FakeIris(state["native"])
        monkeypatch.setattr(module, "_iris", lambda: state["iris"])

        def init_chat_model(**kwargs: Any) -> FakeChat:
            chat = FakeChat(**kwargs)
            state["chats"].append(chat)
            return chat

        monkeypatch.setattr(module, "_init_chat_model", lambda: init_chat_model)
        return state

    return install


def test_reads_the_named_entry_over_the_given_connection(lab):
    state = lab()

    entry = read_llm_entry("ivrtest", "conn")

    assert entry == RESOLVED
    assert state["iris"].connections == ["conn"]
    get = state["native"].calls[0]
    assert get[:2] == ("%ConfigStore.Configuration", "Get")
    assert get[2][0] == "AI.LLM.ivrtest"


def test_only_get_and_resolving_copy_details_are_called(lab):
    """FR-009: no GetDetails, which refuses every AI.LLM entry on 2026.3."""
    state = lab()

    read_llm_entry("ivrtest", "conn")

    assert state["native"].methods() == [
        ("%ConfigStore.Configuration", "Get"),
        ("invoke", "CopyDetails"),
    ]
    assert state["native"].calls[1][2] == ()  # resolveSecrets left at its default, 1


def test_entry_reaches_init_chat_model_unchanged_including_unknown_keys(lab):
    stored = {**ENTRY, "base_url": "https://llm.example.internal/v1", "max_tokens": 256}
    resolved = {**stored, "api_key": "sk-probe-123"}
    state = lab(FakeNative(stored, resolved))

    configstore_llm("ivrtest", FakeConnectionManager())

    assert state["chats"][0].kwargs == resolved


def test_result_is_text_in_text_out(lab):
    state = lab()

    answer = configstore_llm("ivrtest", FakeConnectionManager())

    assert answer("why?") == "answer to why?"
    assert state["chats"][0].prompts == ["why?"]


def test_key_comes_from_the_wallet_not_the_environment(lab, monkeypatch):
    """FR-003: with an entry named, an environment key is never used."""
    state = lab()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")

    configstore_llm("ivrtest", FakeConnectionManager())

    assert state["chats"][0].kwargs["api_key"] == "sk-probe-123"


def test_reader_never_touches_the_environment():
    with open(module.__file__) as handle:
        source = handle.read()

    assert "environ" not in source and "getenv" not in source


def test_read_happens_once_at_build_not_per_question(lab):
    state = lab()
    connections = FakeConnectionManager()

    answer = configstore_llm("ivrtest", connections)
    answer("one")
    answer("two")

    assert len(state["chats"]) == 1
    assert connections.asked == 1
    assert state["native"].methods().count(("%ConfigStore.Configuration", "Get")) == 1


def test_uses_the_pipeline_connection(lab):
    state = lab()

    configstore_llm("ivrtest", FakeConnectionManager("pipeline-conn"))

    assert state["iris"].connections == ["pipeline-conn"]
