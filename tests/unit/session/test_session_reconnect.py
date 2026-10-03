"""A dead connection is replaced once, and the caller writes no retry code.

Feature: 083-iris-session (FR-009, US2)
"""

import pytest

from iris_vector_rag.session import IRISSession
from iris_vector_rag.session.errors import ConnectionLost
from iris_vector_rag.testing.fake_driver import FakeDriver, fake_identity, fake_session


def test_a_dropped_connection_is_replaced_and_the_statement_succeeds():
    session = fake_session(rows=[(1,)], health_interval=3600, drop_after=1)

    session.execute("SELECT 1")  # runs, and leaves the connection at its limit
    result = session.execute("SELECT 1")  # would fail; reconnects and retries

    assert result.one() == (1,)
    assert session.driver.connects == 2


def test_only_one_replacement_per_operation():
    # Every connection this driver hands out is dead on arrival.
    session = fake_session(rows=[(1,)], health_interval=3600, drop_after=0)

    with pytest.raises(ConnectionLost) as caught:
        session.execute("SELECT 1")

    assert session.driver.connects == 2
    assert "fake-host:51972/USER" in str(caught.value)


def test_a_statement_error_is_not_retried_on_a_fresh_connection():
    # IRIS rejecting SQL will reject it again; retrying buys a reconnect and the
    # same failure.
    session = fake_session(
        rows=[(1,)], health_interval=3600, fail_statements=["SELECT bogus"]
    )

    with pytest.raises(ValueError, match="fake statement failure"):
        session.execute("SELECT bogus")

    assert session.driver.connects == 1


def test_a_reconnect_that_cannot_connect_reports_the_connect_failure():
    driver = FakeDriver(rows=[(1,)], drop_after=1)
    session = IRISSession(identity=fake_identity(), driver=driver, health_interval=3600)
    session.execute("SELECT 1")
    driver.connect_failures = 1

    with pytest.raises(ConnectionLost):
        session.execute("SELECT 1")


def test_the_replaced_connection_is_closed():
    session = fake_session(rows=[(1,)], health_interval=3600, drop_after=1)

    session.execute("SELECT 1")
    session.execute("SELECT 1")

    assert session.driver.closes == 1


def test_a_connection_the_caller_closed_is_replaced_without_a_round_trip():
    """The old entry points hand out the session's connection, and callers close it.

    Ten years of code says `conn = get_iris_dbapi_connection(); ...; conn.close()`.
    That connection is the session's now, so the close has to cost the caller nothing
    more than a reconnect on the next statement -- and noticing costs no probe,
    because a driver knows whether its own connection is closed (FR-009, US5).
    """
    session = fake_session(rows=[(1,)], health_interval=3600)

    session.execute("SELECT 1")
    session.connection_for_adapter().close()  # what an unmigrated caller does
    pings_before = session.driver.pings

    assert session.one("SELECT 1") == (1,)
    assert session.driver.connects == 2
    assert session.driver.pings == pings_before, "noticing a closed handle is free"
    assert session.open_connection_count == 1


def test_an_adapter_is_never_handed_a_closed_connection():
    session = fake_session(rows=[(1,)], health_interval=3600)

    first = session.connection_for_adapter()
    first.close()
    second = session.connection_for_adapter()

    assert second is not first
    assert second.closed is False
