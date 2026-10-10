"""An ivg engine logs in as the account IVR connected with, never as a default superuser.

iris-vector-graph 4.2.0 stopped opening its class-method connection as a hard-coded
``_SYSTEM`` / ``SYS``. ``IRISGraphEngine._iris_obj`` now uses the credentials the engine
was built with. ``IRISGraphEngine.from_connect`` keeps them (``_connection_params``); an
engine built from a bare connection keeps none, so ``_iris_obj`` falls back to that
connection itself, which here is the session's connection and is shared with cursor SQL.

IVR builds its engines from the session's connection, so it records the credentials the
session resolved. The BM25, vector and PLAID calls then run on their own connection, as
the account IVR was configured with, and the role of that account decides what they may do.

Feature: iris-vector-graph 4.2.0 (docs/IVG_4X_UPGRADE.md, "4.2.0").
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_vector_rag.integrations import ivg as integration

APP = {
    "host": "db.example",
    "port": 4321,
    "namespace": "APPNS",
    "username": "ivr_app",
    "password": "app-secret",
}
APP_PARAMS = {
    "hostname": "db.example",
    "port": 4321,
    "namespace": "APPNS",
    "username": "ivr_app",
    "password": "app-secret",
}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in (
        "IRIS_HOST",
        "IRIS_PORT",
        "IRIS_NAMESPACE",
        "IRIS_USERNAME",
        "IRIS_USER",
        "IRIS_PASSWORD",
        "IRIS_CONTAINER_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_the_params_have_the_shape_from_connect_keeps():
    assert integration.ivg_connection_params(**APP) == APP_PARAMS


def test_unset_parameters_come_from_the_settings_resolver(monkeypatch):
    """The same resolver the session used, so the account is the one the session has."""
    monkeypatch.setenv("IRIS_USERNAME", "from_env")
    monkeypatch.setenv("IRIS_PASSWORD", "env-secret")

    params = integration.ivg_connection_params(
        host="h", port=1, namespace="N", username=None, password=None
    )

    assert params["username"] == "from_env"
    assert params["password"] == "env-secret"
    assert (params["hostname"], params["port"], params["namespace"]) == ("h", 1, "N")


def test_remember_credentials_sets_them_on_the_engine_and_returns_it():
    engine = SimpleNamespace()

    assert integration.remember_credentials(engine, **APP) is engine
    assert engine._connection_params == APP_PARAMS


def test_the_params_are_not_shared_between_engines():
    first, second = SimpleNamespace(), SimpleNamespace()
    integration.remember_credentials(first, **APP)
    integration.remember_credentials(second, **APP)

    first._connection_params["username"] = "changed"

    assert second._connection_params["username"] == "ivr_app"


# ---------------------------------------------------------------------------
# The contract with ivg itself: what _iris_obj does with them
# ---------------------------------------------------------------------------


def _real_engine(connection):
    ivg = pytest.importorskip("iris_vector_graph")
    # The constructor reads the index registry and the stored vector dtype; neither
    # matters to what is checked here, and neither needs a server.
    with patch.object(
        ivg.IRISGraphEngine, "_build_index_registry", return_value={}
    ), patch.object(
        ivg.IRISGraphEngine, "_detect_stored_vector_dtype", return_value="DOUBLE"
    ):
        return ivg.IRISGraphEngine(connection, embedding_dimension=3)


def _connection():
    connection = MagicMock()
    connection.hostname = "db.example"
    connection.port = 4321
    connection.namespace = "APPNS"
    return connection


def _record_logins(monkeypatch):
    """Replace the two driver entry points `_iris_obj` calls; return what they saw."""
    import iris
    import iris.dbapi

    logins: list = []
    objects: list = []

    def connect(**kwargs):
        logins.append(kwargs)
        return SimpleNamespace(isClosed=lambda: False, tag="dedicated")

    def create(connection):
        objects.append(connection)
        return SimpleNamespace(connection=connection)

    monkeypatch.setattr(iris.dbapi, "connect", connect)
    monkeypatch.setattr(iris, "createIRIS", create)
    return logins, objects


def test_ivg_logs_in_as_the_remembered_account(monkeypatch):
    connection = _connection()
    engine = integration.remember_credentials(_real_engine(connection), **APP)
    logins, objects = _record_logins(monkeypatch)

    engine._iris_obj()

    assert len(logins) == 1
    assert logins[0]["username"] == "ivr_app"
    assert logins[0]["password"] == "app-secret"
    assert logins[0]["username"] != "_SYSTEM"
    assert objects[0].tag == "dedicated"  # not the session's own connection


def test_without_the_credentials_ivg_uses_the_session_connection(monkeypatch):
    """The 4.2.0 fallback this module exists to avoid: shared with cursor SQL."""
    connection = _connection()
    engine = _real_engine(connection)
    logins, objects = _record_logins(monkeypatch)

    engine._iris_obj()

    assert logins == []
    assert objects == [connection]


# ---------------------------------------------------------------------------
# The three places IVR builds an engine
# ---------------------------------------------------------------------------


class _Engine(SimpleNamespace):
    """What `IRISGraphEngine(connection, ...)` returns, as far as these tests look."""

    def __init__(self, connection, **kwargs):
        super().__init__(conn=connection, kwargs=kwargs)

    def _iris_obj(self):
        return Mock()


def test_plaid_engines_remember_the_resolved_account(monkeypatch):
    from iris_vector_rag.pipelines.colbert_iris import plaid

    monkeypatch.setenv("IRIS_USERNAME", "plaid_user")
    monkeypatch.setenv("IRIS_PASSWORD", "plaid-secret")
    session_connection = Mock(name="session connection")

    with patch(
        "iris_vector_rag.common.iris_connection._session_connection",
        return_value=session_connection,
    ), patch("iris_vector_graph.IRISGraphEngine", _Engine):
        engine = plaid._make_engine(
            SimpleNamespace(hostname="h", port=9, namespace="N")
        )

    assert engine.conn is session_connection
    assert engine._connection_params["username"] == "plaid_user"
    assert engine._connection_params["password"] == "plaid-secret"
    assert engine._connection_params["hostname"] == "h"
    assert engine._connection_params["port"] == 9
    assert engine._connection_params["namespace"] == "N"


def test_vecindex_engines_remember_the_resolved_account(monkeypatch):
    from iris_vector_rag.pipelines.colbert_iris import vecindex_phase2

    monkeypatch.setenv("IRIS_USERNAME", "vec_user")
    monkeypatch.setenv("IRIS_PASSWORD", "vec-secret")

    with patch(
        "iris_vector_rag.common.iris_connection._session_connection",
        return_value=Mock(name="session connection"),
    ), patch("iris_vector_graph.IRISGraphEngine", _Engine):
        searcher = vecindex_phase2.VecIndexSearcher(
            SimpleNamespace(hostname="h", port=9, namespace="N")
        )

    assert searcher._engine._connection_params["username"] == "vec_user"
    assert searcher._engine._connection_params["password"] == "vec-secret"


def test_create_graph_engine_remembers_the_account_it_is_given():
    connection = Mock()
    with patch.object(integration, "import_ivg_module") as module, patch.object(
        integration, "assert_ivg_compatible"
    ):
        module.return_value.IRISGraphEngine = _Engine
        engine = integration.create_graph_engine(
            connection, embedding_dimension=8, credentials=APP
        )

    assert engine._connection_params == APP_PARAMS
    assert engine.kwargs == {"embedding_dimension": 8}


# ---------------------------------------------------------------------------
# The schema manager's engine (initialize_schema deploys classes over this login)
# ---------------------------------------------------------------------------


def _manager(identity):
    from iris_vector_rag.storage.schema_manager import SchemaManager

    manager = SchemaManager.__new__(SchemaManager)
    manager.base_embedding_dimension = 384
    manager.connection_manager = MagicMock()
    manager.connection_manager.session.identity = identity
    manager.connection_manager.get_connection.return_value = MagicMock(
        name="connection"
    )
    return manager


def test_the_schema_manager_passes_the_session_account_to_the_engine():
    from iris_vector_rag.session import SessionIdentity

    identity = SessionIdentity(
        host="db.example",
        port=4321,
        namespace="APPNS",
        username="ivr_app",
        password="app-secret",
    )
    manager = _manager(identity)
    engine = MagicMock()
    engine.initialize_schema.return_value = {}

    with patch(
        "iris_vector_rag.storage.schema_manager.create_graph_engine",
        return_value=engine,
    ) as factory:
        manager._initialize_ivg_schema()

    factory.assert_called_once_with(
        manager.connection_manager.get_connection.return_value,
        embedding_dimension=384,
        credentials=APP,
    )


def test_a_connection_manager_with_no_identity_passes_no_credentials():
    manager = _manager(identity=None)
    engine = MagicMock()
    engine.initialize_schema.return_value = {}

    with patch(
        "iris_vector_rag.storage.schema_manager.create_graph_engine",
        return_value=engine,
    ) as factory:
        manager._initialize_ivg_schema()

    factory.assert_called_once_with(
        manager.connection_manager.get_connection.return_value, embedding_dimension=384
    )
