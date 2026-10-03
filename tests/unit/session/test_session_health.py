"""Health is a deadline, not a tax on every statement.

Feature: 083-iris-session (FR-008, SC-002, US2)
"""

from tests.support.fake_driver import fake_session


def test_a_hundred_operations_inside_one_interval_pay_for_one_probe():
    session = fake_session(rows=[(1,)], health_interval=3600)

    for _ in range(100):
        session.execute("SELECT 1")

    # The connect established health; nothing after it needed to ask again.
    assert session.driver.pings == 0
    assert session.driver.connects == 1
    assert len(session.driver.statements) == 100


def test_one_probe_once_the_interval_has_elapsed():
    session = fake_session(rows=[(1,)], health_interval=0.0)

    session.execute("SELECT 1")
    session.execute("SELECT 1")
    session.execute("SELECT 1")

    # First statement connected; the two after it were each past the deadline.
    assert session.driver.pings == 2


def test_a_failed_probe_replaces_the_connection_before_the_statement_runs():
    session = fake_session(rows=[(1,)], health_interval=0.0)
    session.execute("SELECT 1")
    first = session.driver.connections[0]
    first.dropped = True

    session.execute("SELECT 1")

    assert session.driver.connects == 2
    assert session.is_connected is True


def test_is_connected_never_probes():
    session = fake_session(rows=[(1,)], health_interval=0.0)
    session.execute("SELECT 1")
    probes_before = session.driver.pings

    for _ in range(10):
        assert session.is_connected is True

    assert session.driver.pings == probes_before


def test_the_health_interval_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("IRIS_HEALTH_INTERVAL", "12")

    assert fake_session().health_interval == 12.0


def test_the_health_interval_default_is_thirty_seconds(monkeypatch):
    monkeypatch.delenv("IRIS_HEALTH_INTERVAL", raising=False)

    assert fake_session().health_interval == 30.0
