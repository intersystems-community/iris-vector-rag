"""Statements are safe to run and safe to read.

Feature: 083-iris-session (FR-011, FR-012, US3)
"""

import pytest

from tests.support.fake_driver import fake_session


def test_values_survive_the_cursor():
    session = fake_session(columns=("doc_id", "text"), rows=[("d1", "hello")])

    rows = session.execute("SELECT doc_id, text FROM RAG.SourceDocuments")

    assert session.driver.open_cursors == 0  # the cursor is already gone
    assert rows.rows == (("d1", "hello"),)
    assert rows.columns == ("doc_id", "text")
    assert rows.dicts() == [{"doc_id": "d1", "text": "hello"}]


def test_parameters_are_passed_through():
    session = fake_session(rows=[(1,)])

    session.execute("SELECT TOP 1 x FROM t WHERE id = ?", ("abc",))

    assert session.driver.parameters[-1] == ("abc",)


def test_a_trailing_semicolon_is_stripped():
    # IRIS rejects it, and callers keep writing it.
    session = fake_session(rows=[(1,)])

    session.execute("SELECT 1;")

    assert session.driver.statements[-1] == "SELECT 1"


def test_the_cursor_is_released_when_the_statement_fails():
    session = fake_session(rows=[(1,)], fail_statements=["SELECT bad"])

    with pytest.raises(ValueError):
        session.execute("SELECT bad")

    assert session.driver.open_cursors == 0


def test_one_returns_none_for_no_rows():
    session = fake_session(rows=[])

    assert session.one("SELECT TOP 1 x FROM empty_table") is None


def test_one_returns_the_row():
    session = fake_session(rows=[(7,)])

    assert session.one("SELECT COUNT(*) FROM t") == (7,)


def test_execute_many_reports_the_batch_and_releases_its_cursor():
    session = fake_session()

    affected = session.execute_many(
        "INSERT INTO t (a, b) VALUES (?, ?)", [(1, 2), (3, 4)]
    )

    assert affected == 2
    assert session.driver.open_cursors == 0
    assert session.driver.parameters[-1] == [(1, 2), (3, 4)]


def test_execute_many_with_nothing_to_do_runs_nothing():
    session = fake_session()

    assert session.execute_many("INSERT INTO t VALUES (?)", []) == 0
    assert session.driver.statements == []


def test_a_write_reports_the_driver_rowcount():
    session = fake_session()

    result = session.execute("UPDATE t SET a = 1")

    assert result.rowcount == 1
    assert list(result) == []


def test_the_adapter_accessor_hands_over_the_driver_connection():
    session = fake_session()

    connection = session.connection_for_adapter()

    assert connection is session.driver.connections[0]


def test_the_session_is_a_context_manager():
    session = fake_session(rows=[(1,)])

    with session as opened:
        opened.execute("SELECT 1")

    assert session.is_connected is False
    assert session.driver.closes == 1
