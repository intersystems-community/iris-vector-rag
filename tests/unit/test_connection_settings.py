"""The production connection-settings resolver (tickets G5 and G11).

Precedence: explicit settings, environment, `.env`, pinned container, defaults.
Variables are iris-devtester's: IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME,
IRIS_PASSWORD, IRIS_TIMEOUT, IRIS_CONTAINER_NAME. IRIS_USER is a deprecated alias.
"""

from __future__ import annotations

import logging
import os
import subprocess
import warnings
from types import SimpleNamespace

import pytest

from iris_vector_rag.config import connection_settings as cs
from iris_vector_rag.config.connection_settings import (
    ConnectionSettingsError,
    resolve_connection_settings,
)

NO_DOTENV = "/nonexistent/.env"


def no_docker(name):  # pragma: no cover - must never be called
    raise AssertionError(f"docker was asked about {name!r}")


def resolve(explicit=None, environ=None, dotenv=NO_DOTENV, docker_port=no_docker):
    return resolve_connection_settings(
        explicit, environ=environ or {}, dotenv_path=dotenv, docker_port=docker_port
    )


@pytest.fixture(autouse=True)
def _fresh_log_memory():
    cs._logged.clear()
    yield
    cs._logged.clear()


def test_defaults_are_devtesters():
    s = resolve()
    assert (s.host, s.port, s.namespace, s.username, s.password) == (
        "localhost",
        1972,
        "USER",
        "_SYSTEM",
        "SYS",
    )
    assert s.all_defaults
    assert set(s.sources.values()) == {"default"}


def test_environment_beats_defaults():
    s = resolve(
        environ={
            "IRIS_HOST": "h",
            "IRIS_PORT": "51972",
            "IRIS_NAMESPACE": "N",
            "IRIS_USERNAME": "u",
            "IRIS_PASSWORD": "p",
            "IRIS_TIMEOUT": "7",
        }
    )
    assert (s.host, s.port, s.namespace, s.username, s.password, s.timeout) == (
        "h",
        51972,
        "N",
        "u",
        "p",
        7,
    )
    assert s.sources["port"] == "environment"
    assert not s.all_defaults


def test_explicit_beats_environment_and_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("IRIS_HOST=fromfile\nIRIS_PORT=2222\n")
    s = resolve(
        {"host": "explicit", "port": 3333},
        environ={"IRIS_HOST": "fromenv", "IRIS_PORT": "1111"},
        dotenv=env_file,
    )
    assert (s.host, s.port) == ("explicit", 3333)
    assert s.sources["host"] == "explicit"


def test_environment_beats_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("IRIS_HOST=fromfile\nIRIS_NAMESPACE=FILE\n")
    s = resolve(environ={"IRIS_HOST": "fromenv"}, dotenv=env_file)
    assert s.host == "fromenv"
    assert s.namespace == "FILE"
    assert s.sources["namespace"] == "dotenv"


def test_env_file_is_read_without_touching_os_environ(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("IRIS_PASSWORD=only-in-file\n")
    before = dict(os.environ)
    s = resolve_connection_settings(
        None, environ={}, dotenv_path=env_file, docker_port=no_docker
    )
    assert s.password == "only-in-file"
    assert dict(os.environ) == before


def test_explicit_none_means_unset_and_an_empty_environment_value_means_unset():
    s = resolve(
        {"host": None, "port": None, "username": None}, environ={"IRIS_HOST": ""}
    )
    assert s.all_defaults


def test_an_explicit_empty_string_is_a_value_and_is_not_replaced():
    assert resolve({"host": ""}, environ={"IRIS_HOST": "fromenv"}).host == ""


def test_devtester_config_object_is_accepted_as_is():
    config = SimpleNamespace(
        host="devhost",
        port=12345,
        namespace="DEV",
        username="dev",
        password="pw",
        timeout=None,
        container_name=None,
    )
    s = resolve(config)
    assert (s.host, s.port, s.namespace, s.username, s.password) == (
        "devhost",
        12345,
        "DEV",
        "dev",
        "pw",
    )
    assert s.timeout == 30


def test_hostname_is_the_drivers_spelling_of_host():
    assert resolve({"hostname": "drv"}).host == "drv"


# -- IRIS_USER alias -----------------------------------------------------------


def test_iris_user_is_a_deprecated_alias():
    with pytest.warns(DeprecationWarning, match="IRIS_USERNAME"):
        s = resolve(environ={"IRIS_USER": "legacy"})
    assert s.username == "legacy"


def test_iris_username_beats_the_alias_and_the_alias_is_then_unused():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = resolve(environ={"IRIS_USER": "legacy", "IRIS_USERNAME": "current"})
    assert s.username == "current"


def test_no_warning_without_the_alias():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve(environ={"IRIS_USERNAME": "current"})


# -- bad values ---------------------------------------------------------------


@pytest.mark.parametrize("bad", ["not-a-port", "0", "70000", "-1"])
def test_a_bad_port_is_refused_by_variable_name(bad):
    with pytest.raises(ConnectionSettingsError) as caught:
        resolve(environ={"IRIS_PORT": bad})
    assert "IRIS_PORT" in str(caught.value)
    assert bad in str(caught.value)
    assert isinstance(caught.value, ValueError)


def test_a_bad_timeout_is_refused():
    with pytest.raises(ConnectionSettingsError, match="IRIS_TIMEOUT"):
        resolve(environ={"IRIS_TIMEOUT": "soon"})


# -- pinned container, no scanning ---------------------------------------------


def test_no_docker_is_consulted_without_a_pin():
    resolve(environ={})  # no_docker raises if called


def test_no_docker_is_consulted_when_the_port_is_set():
    s = resolve(environ={"IRIS_CONTAINER_NAME": "mine", "IRIS_PORT": "4242"})
    assert s.port == 4242
    assert s.container_name == "mine"


def test_a_pinned_container_supplies_the_port():
    asked = []

    def docker_port(name):
        asked.append(name)
        return 55555

    s = resolve(environ={"IRIS_CONTAINER_NAME": "mine"}, docker_port=docker_port)
    assert s.port == 55555
    assert s.sources["port"] == "container"
    assert asked == ["mine"]
    assert not s.all_defaults


def test_a_pin_in_the_env_file_counts(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("IRIS_CONTAINER_NAME=filed\n")
    s = resolve(dotenv=env_file, docker_port=lambda name: 4321)
    assert s.port == 4321 and s.container_name == "filed"


def test_a_pinned_container_that_publishes_nothing_raises():
    with pytest.raises(ConnectionSettingsError, match="mine"):
        resolve(environ={"IRIS_CONTAINER_NAME": "mine"}, docker_port=lambda name: None)


def test_docker_port_looks_up_the_exact_name_and_nothing_else(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="0.0.0.0:51972\n", stderr="")

    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    assert cs._docker_port("iris-vector-rag-iris") == 51972
    assert calls == [["docker", "port", "iris-vector-rag-iris", "1972/tcp"]]


def test_no_library_module_scans_docker_or_fixes_passwords():
    import pathlib

    root = pathlib.Path(cs.__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if "testing" in path.parts:
            continue
        text = path.read_text()
        for needle in (
            'docker", "ps"',
            "docker', 'ps'",
            "UnExpireUserPasswords",
            "_hard_fix_iris_passwords",
            "auto_detect_iris_port",
            "recover_password",
        ):
            if needle in text:
                offenders.append((str(path.relative_to(root)), needle))
    assert offenders == []


# -- the one log line ------------------------------------------------------------


def test_the_resolution_logs_host_port_and_user_once(caplog):
    with caplog.at_level(logging.INFO, logger=cs.logger.name):
        resolve(environ={"IRIS_PASSWORD": "topsecret"})
        resolve(environ={"IRIS_PASSWORD": "topsecret"})
    lines = [
        r.getMessage() for r in caplog.records if "localhost:1972" in r.getMessage()
    ]
    assert len(lines) == 1
    assert "_SYSTEM" in lines[0]
    assert "topsecret" not in lines[0]
    assert "default" in lines[0]


def test_the_password_is_not_in_the_repr():
    assert "hunter2" not in repr(resolve(environ={"IRIS_PASSWORD": "hunter2"}))


# -- the config-file tier ------------------------------------------------------


def test_environment_beats_the_config_file():
    s = resolve_connection_settings(
        None,
        config_file={"port": 1111, "host": "filehost"},
        environ={"IRIS_PORT": "2000"},
        dotenv_path=NO_DOTENV,
        docker_port=no_docker,
    )
    assert (s.port, s.host) == (2000, "filehost")
    assert s.sources["host"] == "config_file"
    assert not s.all_defaults


def test_the_pinned_container_beats_the_config_file():
    s = resolve_connection_settings(
        None,
        config_file={"port": 1972},
        environ={"IRIS_CONTAINER_NAME": "mine"},
        dotenv_path=NO_DOTENV,
        docker_port=lambda name: 60000,
    )
    assert s.port == 60000 and s.sources["port"] == "container"


def test_a_bad_config_file_port_is_named():
    with pytest.raises(ConnectionSettingsError, match="database.iris.port"):
        resolve_connection_settings(
            None,
            config_file={"port": "x"},
            environ={},
            dotenv_path=NO_DOTENV,
            docker_port=no_docker,
        )
