"""SessionIdentity reads its settings through the one resolver (G5), and a
connection that fails on nothing-but-defaults says so (refusal hint)."""

from __future__ import annotations

import subprocess

import pytest

from iris_vector_rag.config import connection_settings as cs
from iris_vector_rag.session.errors import (
    ConnectFailed,
    HandshakeTimeout,
    IRISUnreachable,
)
from iris_vector_rag.session.identity import SessionIdentity

IRIS_VARS = (
    "IRIS_HOST",
    "IRIS_PORT",
    "IRIS_NAMESPACE",
    "IRIS_USERNAME",
    "IRIS_USER",
    "IRIS_PASSWORD",
    "IRIS_TIMEOUT",
    "IRIS_CONTAINER_NAME",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for name in IRIS_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    cs._logged.clear()


@pytest.fixture
def no_subprocess(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"subprocess.run called: {args}")

    monkeypatch.setattr(subprocess, "run", refuse)


def test_an_unset_port_is_1972_and_docker_is_never_asked(no_subprocess):
    identity = SessionIdentity.from_env()
    assert identity.key == ("localhost", 1972, "USER", "_SYSTEM")


def test_iris_username_beats_iris_user_in_the_identity(monkeypatch):
    monkeypatch.setenv("IRIS_USER", "legacy")
    monkeypatch.setenv("IRIS_USERNAME", "current")
    assert SessionIdentity.from_env().username == "current"


def test_the_pinned_container_supplies_the_identitys_port(monkeypatch):
    monkeypatch.setenv("IRIS_CONTAINER_NAME", "mine")
    monkeypatch.setattr(cs, "_docker_port", lambda name: 44444)
    assert SessionIdentity.from_env().port == 44444


def test_identity_is_defaulted_only_when_nothing_was_configured(monkeypatch):
    assert SessionIdentity.from_env().from_defaults
    monkeypatch.setenv("IRIS_PORT", "51972")
    assert not SessionIdentity.from_env().from_defaults
    assert not SessionIdentity(
        host="localhost", port=1972, namespace="USER", username="_SYSTEM"
    ).from_defaults


def test_from_mapping_with_a_value_is_not_defaulted():
    assert not SessionIdentity.from_mapping({"host": "h"}).from_defaults


def test_from_config_passes_the_config_section_as_the_file_tier(monkeypatch):
    class Config:
        def get_connection_settings(self):
            return cs.resolve_connection_settings(config_file={"host": "cfg"})

    identity = SessionIdentity.from_config(Config())
    assert identity.host == "cfg"
    assert not identity.from_defaults


def test_from_config_with_only_defaults_is_defaulted():
    class Config:
        def get_connection_settings(self):
            return cs.resolve_connection_settings()

    assert SessionIdentity.from_config(Config()).from_defaults


# -- the refusal hint ---------------------------------------------------------


def _defaulted():
    return SessionIdentity.from_env()


def _configured(monkeypatch):
    monkeypatch.setenv("IRIS_PORT", "51972")
    return SessionIdentity.from_env()


def test_an_unreachable_default_target_names_the_target_and_the_fix():
    message = str(IRISUnreachable(_defaulted(), "port 1972 not reachable"))
    assert "localhost:1972" in message
    assert "_SYSTEM" in message
    assert "IRIS_PORT" in message and "IRIS_CONTAINER_NAME" in message


def test_an_unreachable_configured_target_gets_the_plain_error(monkeypatch):
    message = str(IRISUnreachable(_configured(monkeypatch), "port 51972 not reachable"))
    assert "IRIS_PORT" not in message
    assert "IRIS_CONTAINER_NAME" not in message


def test_a_timeout_on_defaults_gets_the_hint():
    assert "IRIS_PORT" in str(HandshakeTimeout(_defaulted(), "handshake timed out"))


def test_a_refusal_from_the_driver_on_defaults_gets_the_hint():
    assert "IRIS_PORT" in str(ConnectFailed(_defaulted(), "Connection refused"))


def test_bad_credentials_on_defaults_are_not_called_a_refusal():
    assert "IRIS_PORT" not in str(ConnectFailed(_defaulted(), "Access Denied"))
