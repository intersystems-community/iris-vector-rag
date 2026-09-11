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
