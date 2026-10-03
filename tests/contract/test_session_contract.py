"""Contract: the session surface other code is allowed to depend on.

The point of Feature 083 is that there is one way in. This file is what a caller --
including `iris-vector-graph` and the HTTP API -- can rely on without reading the
implementation. It touches no database.

Feature: 083-iris-session (FR-001, FR-002, FR-005, FR-011, FR-013)
"""

import inspect

import pytest

from iris_vector_rag.session import (
    Driver,
    IRISSession,
    Rows,
    SessionIdentity,
    session_for,
)
from iris_vector_rag.session.errors import SessionError
from tests.support.fake_driver import (
    FakeConnection,
    FakeDriver,
    fake_identity,
    fake_session,
)


class TestSessionSurface:
    """The names and shapes callers may use."""

    def test_the_package_exports_the_two_names_a_caller_needs(self):
        import iris_vector_rag.session as package

        assert "session_for" in package.__all__
        assert "IRISSession" in package.__all__

    def test_session_for_takes_a_config_manager_and_optional_overrides(self):
        parameters = inspect.signature(session_for).parameters

        assert list(parameters) == [
            "config_manager",
            "identity",
            "driver",
            "health_interval",
        ]
        assert all(
            parameters[name].default is None for name in parameters
        ), "every argument is optional: the environment is a valid source"

    @pytest.mark.parametrize(
        "name",
        [
            "identity",
            "driver",
            "driver_name",
            "is_connected",
            "in_backoff",
            "health_interval",
            "connect_timeout",
            "backoff_seconds",
            "execute",
            "execute_many",
            "one",
            "connection_for_adapter",
            "close",
            "from_connection",
        ],
    )
    def test_the_session_carries_its_documented_member(self, name):
        assert hasattr(fake_session(), name), f"IRISSession must expose {name}"

    def test_a_session_is_a_context_manager(self):
        assert hasattr(IRISSession, "__enter__")
        assert hasattr(IRISSession, "__exit__")


class TestIdentityContract:
    """Identity is four fields, and the password is not one that can leak."""

    def test_the_key_is_host_port_namespace_and_user(self):
        identity = fake_identity()

        assert identity.key == (
            identity.host,
            identity.port,
            identity.namespace,
            identity.username,
        )

    def test_the_password_never_reaches_a_repr(self):
        identity = fake_identity(password="hunter2")

        assert "hunter2" not in repr(identity)
        assert "hunter2" not in identity.label


class TestAcquisitionContract:
    """Acquiring a session is free; the first statement pays for the connection."""

    def test_acquiring_a_session_does_not_connect(self):
        driver = FakeDriver()

        session = IRISSession(identity=fake_identity(), driver=driver)

        assert driver.connects == 0
        assert session.is_connected is False

    def test_asking_who_it_is_does_not_connect(self):
        session = fake_session()

        assert session.identity.namespace == "USER"
        assert session.driver_name == "fake"
        assert session.driver.connects == 0

    def test_a_borrowed_connection_is_used_and_not_closed(self):
        connection = FakeConnection(FakeDriver())

        session = IRISSession.from_connection(connection)
        session.close()

        assert connection.closed is False


class TestResultContract:
    """Rows are values, not cursor state."""

    def test_rows_are_iterable_sized_and_indexable(self):
        rows = fake_session(columns=("a",), rows=[(1,), (2,)]).execute(
            "SELECT a FROM t"
        )

        assert isinstance(rows, Rows)
        assert len(rows) == 2
        assert rows[0] == (1,)
        assert list(rows) == [(1,), (2,)]

    def test_one_row_or_nothing_is_an_answer_and_two_is_an_error(self):
        assert fake_session(rows=[]).one("SELECT x FROM t") is None
        assert fake_session(rows=[(1,)]).one("SELECT x FROM t") == (1,)

        with pytest.raises(ValueError):
            fake_session(rows=[(1,), (2,)]).one("SELECT x FROM t")

    def test_values_are_readable_after_everything_is_closed(self):
        session = fake_session(columns=("a",), rows=[(1,)])
        rows = session.execute("SELECT a FROM t")
        session.close()

        assert rows.rows == ((1,),)
        assert rows.dicts() == [{"a": 1}]


class TestErrorContract:
    """Every failure a caller can see is one type, and it names the instance."""

    def test_every_session_error_is_a_connection_error(self):
        assert issubclass(SessionError, ConnectionError)

    def test_a_failure_names_the_instance_it_could_not_reach(self):
        session = fake_session(connect_failures=1)

        with pytest.raises(SessionError) as caught:
            session.execute("SELECT 1")

        assert "fake-host:51972/USER" in str(caught.value)
        assert caught.value.identity == session.identity


class TestDriverContract:
    """Anything with the seven operations is a driver, including a test's own."""

    def test_a_substituted_driver_satisfies_the_protocol(self):
        assert isinstance(FakeDriver(), Driver)

    @pytest.mark.parametrize(
        "operation",
        [
            "available",
            "connect",
            "ping",
            "close",
            "password_expired",
        ],
    )
    def test_the_protocol_names_its_operation(self, operation):
        assert hasattr(Driver, operation)

    def test_a_driver_has_a_name_to_report(self):
        assert FakeDriver().name == "fake"


class TestAdapterContract:
    """The entry points that hand out a driver connection go through a session."""

    def test_the_connection_manager_hands_out_the_sessions_connection(
        self, monkeypatch
    ):
        from iris_vector_rag.core.connection import ConnectionManager

        session = fake_session()
        monkeypatch.setattr(
            "iris_vector_rag.session.session_for", lambda *a, **k: session
        )

        manager = ConnectionManager()

        assert manager.session is session
        assert manager.get_connection() is session.connection_for_adapter()

    def test_the_engine_shares_that_one_session(self, monkeypatch):
        from iris_vector_rag.core.connection import ConnectionManager
        from iris_vector_rag.core.engine import IRISVectorEngine

        session = fake_session()
        monkeypatch.setattr(
            "iris_vector_rag.session.session_for", lambda *a, **k: session
        )

        engine = IRISVectorEngine(ConnectionManager())

        assert engine.session is session
        assert session.driver.connects == 0, "sharing a session costs no connection"
        assert engine.connection is session.connection_for_adapter()
        assert session.driver.connects == 1, "and then exactly one"

    def test_the_identity_is_resolved_from_the_configuration_manager(self):
        from iris_vector_rag.config.manager import ConfigurationManager

        config = ConfigurationManager()
        identity = SessionIdentity.from_config(config)
        database = config.get_database_config()

        assert identity.host == database["host"]
        assert identity.port == int(database["port"])
        assert identity.namespace == database["namespace"]
        assert identity.username == database["username"]
