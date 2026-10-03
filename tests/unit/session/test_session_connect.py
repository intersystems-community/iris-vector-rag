"""A connect that fails costs the budget once, not once per caller.

Feature: 083-iris-session (FR-004, US1)
"""

import pytest

from iris_vector_rag.session import IRISSession
from iris_vector_rag.session.errors import (
    ConnectFailed,
    HandshakeTimeout,
    InBackoff,
    IRISUnreachable,
)
from tests.support.fake_driver import FakeDriver, fake_identity, fake_session


def test_acquiring_a_session_does_not_connect():
    driver = FakeDriver()

    session = IRISSession(identity=fake_identity(), driver=driver)

    assert driver.connects == 0
    assert session.is_connected is False


def test_the_first_statement_connects_once():
    session = fake_session(rows=[(1,)])

    session.execute("SELECT 1")
    session.execute("SELECT 1")

    assert session.driver.connects == 1


def test_a_failed_connect_is_named_and_opens_the_backoff_window():
    session = fake_session(connect_failures=1, backoff_seconds=60)

    with pytest.raises(ConnectFailed) as caught:
        session.execute("SELECT 1")

    assert "fake-host:51972/USER" in str(caught.value)
    assert session.in_backoff is True


def test_inside_the_window_the_next_caller_fails_immediately():
    driver = FakeDriver(connect_failures=5)
    session = IRISSession(identity=fake_identity(), driver=driver, backoff_seconds=60)

    with pytest.raises(ConnectFailed):
        session.execute("SELECT 1")
    with pytest.raises(InBackoff):
        session.execute("SELECT 1")

    # The second caller did not pay a connect attempt to learn what the first knew.
    assert driver.connects == 1


def test_the_window_expires():
    session = fake_session(connect_failures=1, backoff_seconds=0.0)

    with pytest.raises(ConnectFailed):
        session.execute("SELECT 1")

    assert session.in_backoff is False
    assert session.execute("SELECT 1") is not None
    assert session.driver.connects == 2


def test_a_handshake_timeout_opens_the_window_too():
    identity = fake_identity()
    driver = FakeDriver(
        connect_failures=1, connect_error=HandshakeTimeout(identity, "hung")
    )
    session = IRISSession(identity=identity, driver=driver, backoff_seconds=60)

    with pytest.raises(HandshakeTimeout):
        session.execute("SELECT 1")

    assert session.in_backoff is True


def test_an_unreachable_port_leaves_the_window_closed():
    # Two seconds of socket precheck is cheap, and the answer changes the moment
    # IRIS starts; refusing to look again would make the caller wait longer.
    identity = fake_identity()
    driver = FakeDriver(
        connect_failures=2, connect_error=IRISUnreachable(identity, "port shut")
    )
    session = IRISSession(identity=identity, driver=driver, backoff_seconds=60)

    with pytest.raises(IRISUnreachable):
        session.execute("SELECT 1")

    assert session.in_backoff is False
    with pytest.raises(IRISUnreachable):
        session.execute("SELECT 1")
    assert driver.connects == 2


def test_the_connect_budget_and_backoff_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("IRIS_CONNECT_TIMEOUT", "3")
    monkeypatch.setenv("IRIS_CONNECT_BACKOFF", "7")

    session = IRISSession(identity=fake_identity(), driver=FakeDriver())

    assert session.connect_timeout == 3.0
    assert session.backoff_seconds == 7.0
