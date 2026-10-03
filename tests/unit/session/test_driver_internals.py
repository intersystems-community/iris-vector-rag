"""What each driver does when the environment misbehaves, without an IRIS in sight.

The drivers are the only place in the library allowed to call a DBAPI connect, so
their failure paths are the ones a caller actually meets: a port with nothing behind
it, a handshake that never returns, a container that ships with an expired password.
Every one of them is reachable here with a stand-in ``iris`` module.

Feature: 083-iris-session (FR-006, FR-007, FR-018)
"""

import logging
import socket
import subprocess
import sys
import types

import pytest

from iris_vector_rag.session import drivers as drivers_module
from iris_vector_rag.session.drivers import (
    EmbeddedDriver,
    NetworkDriver,
    default_connect_timeout,
)
from iris_vector_rag.session.errors import (
    HandshakeTimeout,
    IRISUnreachable,
    NoUsableDriver,
    SessionError,
)
from tests.support.fake_driver import fake_identity


def stub_iris(monkeypatch, module):
    """Put a stand-in `iris` module in front of both drivers."""
    monkeypatch.setattr(drivers_module, "_import_iris", lambda: module)
    return module


def iris_with_dbapi(connect):
    module = types.SimpleNamespace()
    module.dbapi = types.SimpleNamespace(connect=connect)
    return module


class TestAHandleThatMisbehaves:
    """A driver reports; it does not raise on top of whatever already went wrong."""

    def test_a_ping_that_cannot_run_is_a_failed_ping_not_an_exception(self):
        class Broken:
            def cursor(self):
                raise ConnectionResetError("connection reset by peer")

        assert NetworkDriver().ping(Broken()) is False
        assert EmbeddedDriver().ping(Broken()) is False

    def test_a_cursor_is_closed_even_when_the_statement_fails(self):
        closed = []

        class Cursor:
            def execute(self, sql):
                raise RuntimeError("no")

            def close(self):
                closed.append(True)

        class Handle:
            def cursor(self):
                return Cursor()

        assert NetworkDriver().ping(Handle()) is False
        assert closed == [True], "a failed probe must not leak its cursor"

    def test_closing_a_handle_that_refuses_is_not_an_error(self):
        """Nothing is waiting on a close, so a raising one has nothing to tell."""

        class Stubborn:
            def close(self):
                raise RuntimeError("already gone")

        NetworkDriver().close(Stubborn())
        EmbeddedDriver().close(Stubborn())

    def test_a_handle_that_raises_when_asked_is_treated_as_open(self):
        class Confused:
            def isClosed(self):
                raise RuntimeError("no idea")

        assert NetworkDriver().is_closed(Confused()) is False


class TestTheEmbeddedDriverSaysWhyItCannotBeUsed:
    """`available()` is a question about this process, answered without connecting."""

    def test_no_iris_module_is_the_reason_given(self, monkeypatch):
        stub_iris(monkeypatch, None)

        assert EmbeddedDriver().available() is False
        assert "no iris.runtime/iris.dbapi" in EmbeddedDriver().unavailable_because()

    def test_a_module_without_a_runtime_is_the_same_answer(self, monkeypatch):
        stub_iris(monkeypatch, types.SimpleNamespace(dbapi=object()))

        assert "no iris.runtime/iris.dbapi" in EmbeddedDriver().unavailable_because()

    def test_a_runtime_that_will_not_say_is_not_usable(self, monkeypatch):
        def explode():
            raise RuntimeError("runtime not initialised")

        stub_iris(
            monkeypatch,
            types.SimpleNamespace(
                runtime=types.SimpleNamespace(get=explode), dbapi=object()
            ),
        )

        assert EmbeddedDriver().available() is False

    @pytest.mark.parametrize("state", ["embedded-kernel", "embedded-local"])
    def test_running_inside_the_instance_is_usable(self, monkeypatch, state):
        stub_iris(monkeypatch, _iris_in_state(state))

        assert EmbeddedDriver().unavailable_because() is None
        assert EmbeddedDriver().available() is True

    def test_an_install_dir_makes_an_unavailable_runtime_usable(self, monkeypatch):
        stub_iris(monkeypatch, _iris_in_state("unavailable"))
        monkeypatch.setenv("IRISINSTALLDIR", "/usr/irissys")

        assert EmbeddedDriver().available() is True

    def test_without_an_install_dir_the_state_itself_is_the_reason(self, monkeypatch):
        stub_iris(monkeypatch, _iris_in_state("unavailable"))
        monkeypatch.delenv("IRISINSTALLDIR", raising=False)
        monkeypatch.delenv("ISC_PACKAGE_INSTALLDIR", raising=False)

        assert "'unavailable'" in EmbeddedDriver().unavailable_because()

    def test_any_other_state_is_named_in_the_reason(self, monkeypatch):
        stub_iris(monkeypatch, _iris_in_state("remote"))

        assert "'remote'" in EmbeddedDriver().unavailable_because()


def _iris_in_state(state, connect=None, configure=None):
    configured = configure or (lambda **kwargs: None)
    return types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            get=lambda: types.SimpleNamespace(state=state),
            configure=configured,
        ),
        dbapi=types.SimpleNamespace(connect=connect or (lambda **kwargs: object())),
    )


class TestTheEmbeddedDriverConnects:
    def test_it_asks_only_for_the_namespace(self, monkeypatch):
        asked = {}

        def connect(**kwargs):
            asked.update(kwargs)
            return "connection"

        stub_iris(monkeypatch, _iris_in_state("embedded-kernel", connect=connect))

        assert EmbeddedDriver().connect(fake_identity()) == "connection"
        # No host, port or password: a process inside the instance has none to give.
        assert asked == {"namespace": "USER"}

    def test_an_unconfigured_runtime_is_configured_from_the_install_dir(
        self, monkeypatch, caplog
    ):
        configured = {}
        stub_iris(
            monkeypatch,
            _iris_in_state("unavailable", configure=lambda **kw: configured.update(kw)),
        )
        monkeypatch.setenv("IRISINSTALLDIR", "/usr/irissys")

        with caplog.at_level(logging.DEBUG, logger=drivers_module.__name__):
            EmbeddedDriver().connect(fake_identity())

        assert configured == {"mode": "embedded", "install_dir": "/usr/irissys"}

    def test_a_failure_arrives_as_a_session_error(self, monkeypatch):
        def refuse(**kwargs):
            raise RuntimeError("namespace USER does not exist")

        stub_iris(monkeypatch, _iris_in_state("embedded-kernel", connect=refuse))

        with pytest.raises(SessionError) as caught:
            EmbeddedDriver().connect(fake_identity())

        assert "fake-host:51972/USER" in str(caught.value)

    def test_no_iris_module_at_connect_time_names_the_driver_problem(self, monkeypatch):
        stub_iris(monkeypatch, None)

        with pytest.raises(NoUsableDriver):
            EmbeddedDriver().connect(fake_identity())

    def test_authentication_is_not_the_embedded_drivers_business(self):
        driver = EmbeddedDriver()

        # A process running inside the instance is already past authentication, so
        # there is no expired password for it to recover from.
        assert (
            driver.password_expired(RuntimeError("Password change required")) is False
        )


class TestTheNetworkDriversTimeout:
    def test_the_default_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("IRIS_CONNECT_TIMEOUT", "3")

        assert default_connect_timeout() == 3.0
        assert NetworkDriver().connect_timeout == 3.0

    def test_an_explicit_timeout_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("IRIS_CONNECT_TIMEOUT", "3")

        assert NetworkDriver(connect_timeout=0.5).connect_timeout == 0.5

    def test_with_nothing_set_there_is_still_a_bound(self, monkeypatch):
        monkeypatch.delenv("IRIS_CONNECT_TIMEOUT", raising=False)

        assert NetworkDriver().connect_timeout == 10.0


class TestTheNetworkDriversAvailability:
    def test_no_iris_module_is_the_reason(self, monkeypatch):
        stub_iris(monkeypatch, None)

        assert NetworkDriver().unavailable_because() == "iris module not importable"

    def test_a_dbapi_connect_is_enough(self, monkeypatch):
        stub_iris(monkeypatch, iris_with_dbapi(lambda **kwargs: object()))

        assert NetworkDriver().available() is True

    @pytest.mark.parametrize("attribute", ["createConnection", "connect"])
    def test_the_older_entry_points_are_accepted(self, monkeypatch, attribute):
        stub_iris(monkeypatch, types.SimpleNamespace(**{attribute: lambda *a, **k: 1}))

        assert NetworkDriver().available() is True

    def test_a_module_with_no_way_in_says_so(self, monkeypatch):
        stub_iris(monkeypatch, types.SimpleNamespace())

        assert "no usable connect method" in NetworkDriver().unavailable_because()


class TestTheNetworkDriverReachesTheInstance:
    def test_a_port_with_nothing_behind_it_is_reported_before_any_handshake(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            free_port = listener.getsockname()[1]

        identity = fake_identity(host="127.0.0.1", port=free_port)

        with pytest.raises(IRISUnreachable) as caught:
            NetworkDriver().connect(identity)

        message = str(caught.value)
        assert f"port {free_port} not reachable" in message
        assert "check IRIS is running" in message

    def test_the_dbapi_path_passes_the_whole_identity(self, monkeypatch):
        asked = {}
        stub_iris(monkeypatch, iris_with_dbapi(lambda **kw: asked.update(kw) or "conn"))
        driver = NetworkDriver()
        monkeypatch.setattr(driver, "_precheck", lambda identity: None)

        assert driver.connect(fake_identity()) == "conn"
        assert asked == {
            "hostname": "fake-host",
            "port": 51972,
            "namespace": "USER",
            "username": "_SYSTEM",
            "password": "SYS",
            # Shared memory would reach an instance on this host, not the one asked for.
            "sharedmemory": False,
        }

    def test_the_create_connection_path_passes_them_positionally(self, monkeypatch):
        seen = []
        stub_iris(
            monkeypatch,
            types.SimpleNamespace(
                createConnection=lambda *args: seen.append(args) or "conn"
            ),
        )
        driver = NetworkDriver()
        monkeypatch.setattr(driver, "_precheck", lambda identity: None)

        assert driver.connect(fake_identity()) == "conn"
        assert seen == [("fake-host", 51972, "USER", "_SYSTEM", "SYS")]

    def test_the_plain_connect_path_is_the_last_resort(self, monkeypatch):
        asked = {}
        stub_iris(
            monkeypatch,
            types.SimpleNamespace(connect=lambda **kw: asked.update(kw) or "conn"),
        )
        driver = NetworkDriver()
        monkeypatch.setattr(driver, "_precheck", lambda identity: None)

        assert driver.connect(fake_identity()) == "conn"
        assert asked["hostname"] == "fake-host"

    def test_a_handshake_that_never_returns_is_given_up_on(self, monkeypatch):
        """The driver's own connect has no timeout, so the session imposes one."""
        import threading

        release = threading.Event()

        def never(**kwargs):
            release.wait(30)
            return "too late"

        stub_iris(monkeypatch, iris_with_dbapi(never))
        driver = NetworkDriver(connect_timeout=0.05)
        monkeypatch.setattr(driver, "_precheck", lambda identity: None)

        try:
            with pytest.raises(HandshakeTimeout) as caught:
                driver.connect(fake_identity())
        finally:
            release.set()

        assert "did not complete within" in str(caught.value)

    def test_a_refused_handshake_arrives_as_a_session_error(self, monkeypatch):
        def refuse(**kwargs):
            raise RuntimeError("Access Denied")

        stub_iris(monkeypatch, iris_with_dbapi(refuse))
        driver = NetworkDriver()
        monkeypatch.setattr(driver, "_precheck", lambda identity: None)

        with pytest.raises(SessionError) as caught:
            driver.connect(fake_identity())

        assert "fake-host:51972/USER" in str(caught.value)


class TestTheExpiredPasswordAFreshContainerShipsWith:
    @pytest.mark.parametrize(
        "text",
        ["Password change required", "the password has expired"],
    )
    def test_the_state_is_recognised_from_the_failure(self, text):
        assert NetworkDriver().password_expired(RuntimeError(text)) is True

    def test_an_unrelated_failure_is_not_that_state(self):
        assert NetworkDriver().password_expired(RuntimeError("Access Denied")) is False


def test_the_precheck_timeout_is_short_enough_to_be_a_precheck():
    # A precheck that can hang for the connect budget is not a precheck.
    assert drivers_module.PRECHECK_TIMEOUT <= default_connect_timeout() / 2


def test_the_install_dir_can_come_from_either_variable(monkeypatch):
    monkeypatch.delenv("IRISINSTALLDIR", raising=False)
    monkeypatch.setenv("ISC_PACKAGE_INSTALLDIR", "/usr/irissys")

    assert EmbeddedDriver()._install_dir() == "/usr/irissys"

    monkeypatch.setenv("IRISINSTALLDIR", "/opt/iris")
    assert EmbeddedDriver()._install_dir() == "/opt/iris"


def test_importing_iris_reports_rather_than_raises(monkeypatch, caplog):
    """A missing driver package is a driver-selection fact, not a crash."""
    import builtins

    original = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "iris":
            raise ImportError("no module named iris")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.delitem(sys.modules, "iris", raising=False)

    with caplog.at_level(logging.DEBUG, logger=drivers_module.__name__):
        assert drivers_module._import_iris() is None

    assert "not importable" in caplog.text
