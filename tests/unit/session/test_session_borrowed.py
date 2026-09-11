"""A connection the caller owns is used, never closed, and never replaced.

Feature: 083-iris-session (FR-010, US1)
"""

import pytest

from iris_vector_rag.session import IRISSession
from iris_vector_rag.session.errors import CallerConnectionBroken
from iris_vector_rag.testing.fake_driver import FakeConnection, FakeDriver


def borrowed(**kwargs):
    driver = FakeDriver(**kwargs)
    connection = FakeConnection(driver)
    return IRISSession.from_connection(connection), driver, connection


def test_the_caller_connection_is_the_one_used():
    session, driver, connection = borrowed(rows=[(42,)])

    assert session.execute("SELECT x").one() == (42,)
    assert connection.statements_run == 1
    assert driver.connects == 0


def test_a_borrowed_session_reports_itself_connected_without_probing():
    session, driver, _ = borrowed()

    assert session.is_connected is True
    assert driver.pings == 0


def test_closing_the_session_does_not_close_the_caller_connection():
    session, driver, connection = borrowed()

    session.close()

    assert connection.closed is False
    assert driver.closes == 0


def test_a_broken_caller_connection_is_reported_not_replaced():
    session, driver, connection = borrowed(rows=[(1,)], drop_after=0)

    with pytest.raises(CallerConnectionBroken) as caught:
        session.execute("SELECT 1")

    assert driver.connects == 0
    assert "supplied" in str(caught.value).lower()


def test_a_borrowed_session_never_probes_on_a_cadence():
    session, driver, _ = borrowed(rows=[(1,)])

    for _ in range(5):
        session.execute("SELECT 1")

    assert driver.pings == 0
