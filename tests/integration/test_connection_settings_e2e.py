"""End-to-end gates for the connection-settings resolver, against a real IRIS.

Every public connection entry point reaches IRIS with the IRIS_* variables alone (the
environment `iris_target` sets up) and reads `SELECT 1`; a pinned container is found by
exact name; the settings a process logs are the ones it connects with.

Feature: connection-settings (tickets G5, G11)
"""

from __future__ import annotations

import logging
import subprocess

import pytest

from iris_vector_rag.config import connection_settings as cs
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.session import session_for
from iris_vector_rag.session.registry import reset_sessions_for_tests


@pytest.fixture(autouse=True)
def fresh_sessions():
    reset_sessions_for_tests()
    cs._logged.clear()
    yield
    reset_sessions_for_tests()


@pytest.fixture
def env_only(iris_target, monkeypatch, tmp_path):
    """Nothing but the IRIS_* variables and no .env in the working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("IRIS_USER", raising=False)
    monkeypatch.delenv("IRIS_CONTAINER_NAME", raising=False)
    for name, value in iris_target.environment().items():
        monkeypatch.setenv(name, value)
    return iris_target


def _select_one(connection) -> int:
    cursor = connection.cursor()
    cursor.execute("SELECT 1")
    row = cursor.fetchone()
    value = row[0]
    cursor.close()
    return value


def test_the_resolver_reads_the_target_from_the_environment(env_only):
    settings = cs.resolve_connection_settings()
    assert (settings.host, settings.port) == (env_only.host, env_only.port)
    assert settings.sources["port"] == "environment"


def test_a_session_from_environment_only_connects(env_only):
    assert session_for().one("SELECT 1") == (1,)


def test_a_session_from_a_config_manager_connects(env_only):
    assert session_for(ConfigurationManager()).one("SELECT 1") == (1,)


def test_the_deprecated_get_iris_connection_connects(env_only):
    from iris_vector_rag.common.iris_connection import get_iris_connection

    with pytest.warns(DeprecationWarning):
        connection = get_iris_connection()
    assert _select_one(connection) == 1


def test_the_deprecated_dbapi_wrapper_connects_with_a_config_mapping(env_only):
    from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection

    with pytest.warns(DeprecationWarning):
        connection = get_iris_dbapi_connection(
            {"hostname": env_only.host, "port": str(env_only.port)}
        )
    assert _select_one(connection) == 1


def test_the_deprecated_connection_manager_connects(env_only):
    from iris_vector_rag.common.iris_connection_manager import IRISConnectionManager

    with pytest.warns(DeprecationWarning):
        manager = IRISConnectionManager()
    assert _select_one(manager.get_connection()) == 1


def test_the_resolved_settings_are_logged_once_and_are_the_ones_used(env_only, caplog):
    with caplog.at_level(logging.INFO, logger=cs.logger.name):
        session_for().one("SELECT 1")
        session_for().one("SELECT 1")
    lines = [
        r.getMessage()
        for r in caplog.records
        if f"{env_only.host}:{env_only.port}" in r.getMessage()
    ]
    assert len(lines) == 1
    assert env_only.username in lines[0]
    assert env_only.password not in lines[0].replace(env_only.username, "")


def test_nothing_configured_and_nothing_listening_gets_the_refusal_hint(
    monkeypatch, tmp_path
):
    """The defaults point at localhost:1972; if nothing listens there, say so."""
    import socket

    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", 1972)) == 0:
            pytest.skip("something is listening on 1972 here, so there is no refusal")

    for name in cs._ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("IRIS_USER", raising=False)
    monkeypatch.delenv("IRIS_CONTAINER_NAME", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConnectionError) as caught:
        session_for().one("SELECT 1")
    message = str(caught.value)
    assert "localhost:1972" in message
    assert "IRIS_PORT" in message and "IRIS_CONTAINER_NAME" in message


def test_a_pinned_container_is_found_by_its_exact_name(
    iris_target, monkeypatch, tmp_path
):
    name = iris_target.container
    monkeypatch.chdir(tmp_path)
    for var in cs._ENV_NAMES.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("IRIS_USER", raising=False)
    monkeypatch.setenv("IRIS_CONTAINER_NAME", name)
    # credentials are not discoverable from Docker; the target's are set explicitly
    monkeypatch.setenv("IRIS_USERNAME", iris_target.username)
    monkeypatch.setenv("IRIS_PASSWORD", iris_target.password)
    monkeypatch.setenv("IRIS_NAMESPACE", iris_target.namespace)

    settings = cs.resolve_connection_settings()
    assert settings.port == iris_target.port
    assert settings.sources["port"] == "container"
    assert session_for().one("SELECT 1") == (1,)

    # the lookup is by exact name: a longer name that merely contains it is no match
    assert cs._docker_port(name + "-not-a-container") is None
