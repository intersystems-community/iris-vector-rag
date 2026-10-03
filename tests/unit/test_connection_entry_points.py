"""Every connection entry point reads through the resolver (G5 decision 4).

CM-05 (one username precedence), CM-06 (LOG_LEVEL beats the file), CM-09 (no DB URL
printed), CM-10 (config honoured), CM-14 (no dead re-check), CM-16 (no docker per call),
OS-17 (import does not load .env), and the duplicate `embeddings` key in the default
configuration.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import yaml

from iris_vector_rag.config import connection_settings as cs
from iris_vector_rag.config.manager import ConfigurationManager

PACKAGE = Path(cs.__file__).resolve().parents[1]

IRIS_VARS = (
    "IRIS_HOST",
    "IRIS_PORT",
    "IRIS_NAMESPACE",
    "IRIS_USERNAME",
    "IRIS_USER",
    "IRIS_PASSWORD",
    "IRIS_TIMEOUT",
    "IRIS_CONTAINER_NAME",
    "IRIS_DRIVER_PATH",
    "LOG_LEVEL",
    "LOG_PATH",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for name in IRIS_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    cs._logged.clear()


@pytest.fixture
def no_subprocess(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"subprocess.run called: {args}")

    monkeypatch.setattr(subprocess, "run", refuse)


def _config_file(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(body))
    return str(path)


# -- the default configuration ---------------------------------------------------


class _NoDuplicateKeys(yaml.SafeLoader):
    pass


def _no_dupes(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_NoDuplicateKeys.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: _no_dupes(loader, node),
)


def test_default_config_has_no_duplicate_keys():
    text = (PACKAGE / "config" / "default_config.yaml").read_text()
    yaml.load(text, Loader=_NoDuplicateKeys)


def test_default_config_embeddings_keeps_every_setting():
    loaded = yaml.safe_load((PACKAGE / "config" / "default_config.yaml").read_text())
    embeddings = loaded["embeddings"]
    assert embeddings["mode"] == "sentence_transformers"
    assert embeddings["text_in"] is False
    assert embeddings["default_provider"] == "sentence_transformers"
    assert embeddings["sentence_transformers"] == {
        "model_name": "all-MiniLM-L6-v2",
        "device": "cpu",
    }


def test_default_config_does_not_restate_the_connection_defaults():
    """The defaults live in the resolver; a copy in the file would outrank a pin."""
    loaded = yaml.safe_load((PACKAGE / "config" / "default_config.yaml").read_text())
    section = loaded["database"]["iris"]
    for key in ("host", "port", "namespace", "username", "password"):
        assert key not in section


# -- ConfigurationManager --------------------------------------------------------


def test_database_config_defaults_come_from_the_resolver(no_subprocess):
    db = ConfigurationManager().get_database_config()
    assert (
        db["host"],
        db["port"],
        db["namespace"],
        db["username"],
        db["password"],
    ) == (
        "localhost",
        1972,
        "USER",
        "_SYSTEM",
        "SYS",
    )


def test_database_config_honours_the_pinned_container(monkeypatch):
    monkeypatch.setenv("IRIS_CONTAINER_NAME", "mine")
    monkeypatch.setattr(cs, "_docker_port", lambda name: 40404)
    assert ConfigurationManager().get_database_config()["port"] == 40404


def test_one_username_everywhere(monkeypatch, tmp_path):
    """CM-05: identity, database config and cloud config agree, alias or not."""
    from iris_vector_rag.session.identity import SessionIdentity

    monkeypatch.setenv("IRIS_USER", "legacy")
    monkeypatch.setenv("IRIS_USERNAME", "current")
    manager = ConfigurationManager()
    assert manager.get_database_config()["username"] == "current"
    assert manager.get_cloud_config().connection.username == "current"
    assert SessionIdentity.from_config(manager).username == "current"


def test_the_alias_alone_is_honoured_everywhere_with_a_warning(monkeypatch):
    from iris_vector_rag.session.identity import SessionIdentity

    monkeypatch.setenv("IRIS_USER", "legacy")
    manager = ConfigurationManager()
    with pytest.warns(DeprecationWarning):
        assert manager.get_database_config()["username"] == "legacy"
    with pytest.warns(DeprecationWarning):
        assert manager.get_cloud_config().connection.username == "legacy"
    with pytest.warns(DeprecationWarning):
        assert SessionIdentity.from_config(manager).username == "legacy"


def test_a_bad_port_is_refused_the_same_way_by_every_getter(monkeypatch):
    monkeypatch.setenv("IRIS_PORT", "not-a-port")
    manager = ConfigurationManager()
    with pytest.raises(ValueError, match="IRIS_PORT"):
        manager.get_database_config()
    with pytest.raises(ValueError, match="IRIS_PORT"):
        manager.get_cloud_config()


def test_cloud_config_reports_where_each_value_came_from(monkeypatch, tmp_path):
    from iris_vector_rag.config.entities import ConfigSource

    monkeypatch.setenv("IRIS_HOST", "fromenv")
    path = _config_file(tmp_path, {"database": {"iris": {"namespace": "FILED"}}})
    connection = ConfigurationManager(path).get_cloud_config().connection
    assert connection.host == "fromenv"
    assert connection.source["host"] == ConfigSource.ENVIRONMENT
    assert connection.namespace == "FILED"
    assert connection.source["namespace"] == ConfigSource.CONFIG_FILE
    assert connection.source["port"] == ConfigSource.DEFAULT


def test_log_level_from_the_environment_beats_the_file(monkeypatch, tmp_path):
    """CM-06: environment wins in every getter, logging included."""
    path = _config_file(tmp_path, {"logging": {"level": "DEBUG", "path": "file.log"}})
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    logging_config = ConfigurationManager(path).get_logging_config()
    assert logging_config["level"] == "ERROR"
    assert logging_config["path"] == "file.log"


def test_a_file_only_setup_needs_no_host_in_the_file(tmp_path):
    """CM-14: the host is never missing; the resolver always has one."""
    path = _config_file(tmp_path, {"storage": {"iris": {"table_name": "T"}}})
    ConfigurationManager(path)  # does not raise


# -- the deprecated entry points -----------------------------------------------


def _capture_identity(monkeypatch):
    """Replace the session factory; return what identity each entry point asked for."""
    asked = []

    class FakeSession:
        def connection_for_adapter(self):
            return "connection"

    def fake_session_for(config_manager=None, *, identity=None, **kwargs):
        from iris_vector_rag.session.identity import SessionIdentity

        asked.append(SessionIdentity.resolve(config_manager, identity))
        return FakeSession()

    import iris_vector_rag.session as session_module

    monkeypatch.setattr(session_module, "session_for", fake_session_for)
    return asked


def test_get_iris_connection_honours_explicit_arguments(monkeypatch, no_subprocess):
    from iris_vector_rag.common import iris_connection

    asked = _capture_identity(monkeypatch)
    monkeypatch.setenv("IRIS_HOST", "fromenv")
    with pytest.warns(DeprecationWarning):
        iris_connection.get_iris_connection(host="explicit", port=4000)
    assert (asked[0].host, asked[0].port) == ("explicit", 4000)


def test_get_iris_connection_reads_the_rest_through_the_resolver(
    monkeypatch, no_subprocess
):
    from iris_vector_rag.common import iris_connection

    asked = _capture_identity(monkeypatch)
    monkeypatch.setenv("IRIS_USERNAME", "current")
    monkeypatch.setenv("IRIS_NAMESPACE", "RAG")
    with pytest.warns(DeprecationWarning):
        iris_connection.get_iris_connection()
    assert (asked[0].username, asked[0].namespace, asked[0].port) == (
        "current",
        "RAG",
        1972,
    )


def test_get_iris_connection_names_the_parameter_that_is_invalid(monkeypatch):
    from iris_vector_rag.common import iris_connection
    from iris_vector_rag.common.exceptions import ValidationError

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValidationError) as caught:
            iris_connection.get_iris_connection(port=99999)
    assert caught.value.parameter_name == "port"


def test_the_dbapi_wrapper_honours_its_config(monkeypatch, no_subprocess):
    from iris_vector_rag.common import iris_dbapi_connector

    asked = _capture_identity(monkeypatch)
    with pytest.warns(DeprecationWarning):
        iris_dbapi_connector.get_iris_dbapi_connection(
            {"hostname": "cfg-host", "port": 4001, "namespace": "CFG"}
        )
    assert (asked[0].host, asked[0].port, asked[0].namespace) == (
        "cfg-host",
        4001,
        "CFG",
    )


def test_the_connection_manager_honours_its_config(monkeypatch, no_subprocess):
    from iris_vector_rag.common import iris_connection_manager

    asked = _capture_identity(monkeypatch)
    with pytest.warns(DeprecationWarning):
        manager = iris_connection_manager.IRISConnectionManager()
    manager.get_connection({"host": "mgr-host", "port": "4002", "username": "mgr"})
    assert (asked[0].host, asked[0].port, asked[0].username) == (
        "mgr-host",
        4002,
        "mgr",
    )


def test_the_env_scanner_and_its_duplicate_are_gone():
    from iris_vector_rag.common import iris_connection

    assert not hasattr(iris_connection, "auto_detect_iris_port")
    assert not hasattr(iris_connection, "_get_connection_params_from_env")


# -- CM-09 and OS-17 ------------------------------------------------------------


def test_get_iris_connector_does_not_print_the_url(monkeypatch, capsys):
    from iris_vector_rag.common import utils

    class Engine:
        def connect(self):
            return "conn"

    import sqlalchemy

    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url: Engine())
    utils.get_iris_connector("iris://_SYSTEM:hunter2@host:1972/USER")
    captured = capsys.readouterr()
    assert "hunter2" not in captured.out + captured.err


def test_importing_the_package_does_not_load_a_dotenv_file(tmp_path):
    """OS-17: `.env` is read by the resolver, not loaded into os.environ at import."""
    script = textwrap.dedent("""
        import dotenv

        def refuse(*args, **kwargs):
            raise SystemExit("import called load_dotenv")

        dotenv.load_dotenv = refuse
        import iris_vector_rag
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(PACKAGE.parent),
            "HOME": str(tmp_path),
        },
    )
    assert result.returncode == 0, result.stderr[-800:]


def test_graph_core_discovery_reads_the_resolver_not_its_own_defaults(monkeypatch):
    """It had its own defaults, 1974 and SuperUser, which nothing else shared."""
    from iris_vector_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

    config = GraphCoreDiscovery(None).get_connection_config()
    assert (config["port"], config["username"]) == (1972, "_SYSTEM")
    monkeypatch.setenv("IRIS_PORT", "4444")
    assert GraphCoreDiscovery(None).get_connection_config()["port"] == 4444


@pytest.mark.parametrize(
    "module",
    [
        "iris_vector_rag.pipelines.colbert_iris.plaid",
        "iris_vector_rag.pipelines.colbert_iris.vecindex_phase2",
    ],
)
def test_colbert_native_connections_use_the_resolver(
    module, monkeypatch, no_subprocess
):
    import importlib
    from types import SimpleNamespace

    asked = _capture_identity(monkeypatch)
    monkeypatch.setenv("IRIS_USERNAME", "current")
    importlib.import_module(module)._ensure_native_conn(SimpleNamespace(port=4545))
    assert (asked[0].port, asked[0].username, asked[0].host) == (
        4545,
        "current",
        "localhost",
    )
