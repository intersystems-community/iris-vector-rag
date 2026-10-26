"""The answer model from a real AI Hub ConfigStore entry (spec 093), on ivr-aihub-lab.

The entries, the Wallet secret and the users come from tests/fixtures/aihub/ivrsetup.mac:
``ivrtest`` (key ``secret://ivrprobe.openai#api_key`` = ``sk-probe-123``), ``ivrnokey``
(no ``model``) and ``ivroff`` (disabled). Building a model makes no network call, so the
fake key is never sent anywhere.
"""

from __future__ import annotations

import logging

import iris
import pytest

from iris_vector_rag import create_pipeline
from iris_vector_rag.common.configstore_llm import configstore_llm, read_llm_entry
from iris_vector_rag.exceptions import LLMConfigError
from tests.integration.conftest_aihub import aihub_iris  # noqa: F401  (fixture)

pytestmark = [pytest.mark.aihub, pytest.mark.integration]

KEY = "sk-probe-123"


class Connections:
    """The one method of a connection manager that the reader uses."""

    def __init__(self, connection):
        self.connection = connection

    def get_connection(self, backend_name: str = "iris"):
        return self.connection


def connect(lab, role: str = "admin"):
    user, password = lab["users"][role]
    return iris.dbapi.connect(
        hostname=lab["host"],
        port=lab["port"],
        namespace=lab["namespace"],
        username=user,
        password=password,
        sharedmemory=False,
    )


def assert_built_from_ivrtest(answer):
    chat = answer.chat_model
    assert type(chat).__name__ == "ChatOpenAI"
    assert chat.model_name == "gpt-4o-mini"
    assert chat.temperature == pytest.approx(0.1)
    assert chat.openai_api_key.get_secret_value() == KEY


# --- US1: use the configured model -------------------------------------------------


def test_create_pipeline_builds_the_entry_model(aihub_iris, caplog, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")
    caplog.set_level(logging.DEBUG)
    connection = connect(aihub_iris)
    try:
        pipeline = create_pipeline(
            "basic",
            external_connection=connection,
            embedding_func=lambda texts: [[0.0] * 384 for _ in texts],
            llm_config="ivrtest",
        )
        assert_built_from_ivrtest(pipeline.llm_func)
    finally:
        connection.close()
    assert KEY not in caplog.text


def test_configstore_llm_as_llm_func(aihub_iris, caplog):
    caplog.set_level(logging.DEBUG)
    connection = connect(aihub_iris)
    try:
        answer = configstore_llm("ivrtest", Connections(connection))
    finally:
        connection.close()
    assert_built_from_ivrtest(answer)
    assert KEY not in caplog.text


def test_read_returns_the_entry_with_the_wallet_reference_resolved(aihub_iris):
    connection = connect(aihub_iris)
    try:
        entry = read_llm_entry("ivrtest", connection)
    finally:
        connection.close()
    assert entry == {
        "model": "gpt-4o-mini",
        "model_provider": "openai",
        "temperature": 0.1,
        "api_key": KEY,
    }


# --- US2: a clear error instead of a bad model ---------------------------------------


def build_error(lab, name: str, role: str = "admin") -> LLMConfigError:
    connection = connect(lab, role)
    try:
        with pytest.raises(LLMConfigError) as caught:
            configstore_llm(name, Connections(connection))
    finally:
        connection.close()
    assert KEY not in str(caught.value)
    return caught.value


def test_missing_entry(aihub_iris):
    err = build_error(aihub_iris, "nosuch")

    assert err.entry == "nosuch"
    assert "AI.LLM.nosuch does not exist" in str(err)


def test_no_wallet_privilege(aihub_iris):
    err = build_error(aihub_iris, "ivrtest", role="no_wallet")

    assert "secret://ivrprobe.openai#api_key" in str(err)
    assert "%Admin_Wallet:USE" in str(err)


def test_no_native_class_execution_privilege(aihub_iris):
    err = build_error(aihub_iris, "ivrtest", role="no_native")

    assert "%Native_ClassExecution:USE" in str(err)


def test_missing_model_key(aihub_iris):
    err = build_error(aihub_iris, "ivrnokey")

    assert "`model`" in str(err)


def test_disabled_entry(aihub_iris):
    err = build_error(aihub_iris, "ivroff")

    assert "disabled" in str(err)
