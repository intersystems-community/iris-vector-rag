"""End-to-end gates for the IRIS session, against the real instance.

No mocks. Every test here talks to the container the project owns, because the
failures this feature exists to remove -- an unbounded handshake, a probe on every
call, a row read after its cursor died -- only show up against a real driver.

Feature: 083-iris-session
"""

import os
import time

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.engine import IRISVectorEngine
from iris_vector_rag.session import IRISSession, SessionIdentity, session_for
from iris_vector_rag.session.errors import IRISUnreachable, SessionError
from iris_vector_rag.session.registry import open_sessions, reset_sessions_for_tests


@pytest.fixture
def session():
    """A live session on the project's own instance, cleaned up afterwards."""
    reset_sessions_for_tests()
    live = session_for()
    yield live
    reset_sessions_for_tests()


@pytest.fixture
def unreachable():
    """An identity nothing is listening on."""
    return SessionIdentity(
        host="127.0.0.1", port=59999, namespace="USER", username="_SYSTEM"
    )


class TestUserStory1OneWayIn:
    """Every entry point yields the same session with the same protections."""

    def test_a_session_runs_a_statement_against_the_real_instance(self, session):
        assert session.one("SELECT 1") == (1,)
        assert session.is_connected is True
        assert session.driver_name in ("network", "embedded")

    def test_every_entry_point_shares_one_session(self, session):
        config = ConfigurationManager()
        manager = ConnectionManager(config)
        engine = IRISVectorEngine.from_config()

        assert manager.session is session
        assert engine.session is session
        assert len(open_sessions()) == 1

    def test_the_managers_connection_is_the_sessions_connection(self, session):
        connection = ConnectionManager().get_connection()

        assert connection is session.connection_for_adapter()

        cursor = connection.cursor()
        try:
            cursor.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
        finally:
            cursor.close()

    def test_asking_who_it_is_costs_no_connection(self):
        reset_sessions_for_tests()
        fresh = session_for()

        assert fresh.identity.port == int(os.environ.get("IRIS_PORT", "51972"))
        assert fresh.driver_name
        assert fresh.is_connected is False

    def test_an_unreachable_instance_fails_in_seconds_not_forever(self, unreachable):
        doomed = IRISSession(identity=unreachable)

        started = time.monotonic()
        with pytest.raises(IRISUnreachable) as caught:
            doomed.execute("SELECT 1")
        elapsed = time.monotonic() - started

        assert elapsed < 5.0, f"the socket precheck took {elapsed:.1f}s"
        assert "127.0.0.1:59999/USER" in str(caught.value)

    def test_a_refused_socket_does_not_lock_the_next_caller_out(self, unreachable):
        # The cheap precheck cost nothing, so there is nothing to back off from:
        # an instance that comes up a second later must be reachable.
        doomed = IRISSession(identity=unreachable)

        for _ in range(2):
            with pytest.raises(SessionError):
                doomed.execute("SELECT 1")

        assert doomed.in_backoff is False

    def test_the_engine_reports_the_dimension_the_project_configured(self):
        engine = IRISVectorEngine.from_config()

        expected = int(ConfigurationManager().get_embedding_config()["dimension"])

        assert engine.embedding_dimension == expected

    def test_a_caller_connection_is_used_and_left_open(self, session):
        borrowed = IRISSession.from_connection(session.connection_for_adapter())

        assert borrowed.one("SELECT 1") == (1,)

        borrowed.close()

        assert session.one("SELECT 1") == (1,)  # still usable: nothing was closed


def _count_pings(session, monkeypatch):
    """Count real probes without changing what they do."""
    counted = []
    probe = session.driver.ping

    def counting_ping(connection):
        counted.append(1)
        return probe(connection)

    monkeypatch.setattr(session.driver, "ping", counting_ping)
    return counted


def _terminate_server_job(session):
    """Kill this connection's job inside IRIS. The real death, not a close()."""
    import subprocess

    from iris_vector_rag.session.drivers import NetworkDriver

    job = session.one("SELECT $JOB")[0]
    container = NetworkDriver()._container_publishing(session.identity.port)
    if not container:
        pytest.skip("no local container publishing this port; cannot kill a job")
    script = f"Do ##class(%SYSTEM.Process).Terminate({job},2)\nHalt\n"
    subprocess.run(
        ["docker", "exec", "-i", container, "iris", "session", "IRIS", "-U", "%SYS"],
        input=script.encode(),
        capture_output=True,
        timeout=30,
    )


class TestUserStory2HealthOnACadence:
    """At most one probe per interval, and a dead connection replaced silently."""

    def test_a_hundred_operations_inside_one_interval_pay_for_no_probe(
        self, session, monkeypatch
    ):
        session.execute("SELECT 1")  # connect, which establishes health
        probes = _count_pings(session, monkeypatch)

        for _ in range(100):
            assert session.one("SELECT 1") == (1,)

        assert (
            probes == []
        ), f"{len(probes)} probes inside one {session.health_interval}s interval"

    def test_a_probe_happens_once_the_interval_has_elapsed(self, monkeypatch):
        eager = IRISSession(identity=SessionIdentity.from_env(), health_interval=0.0)
        eager.execute("SELECT 1")
        probes = _count_pings(eager, monkeypatch)

        eager.execute("SELECT 1")
        eager.execute("SELECT 1")

        assert len(probes) == 2
        eager.close()

    def test_a_job_killed_inside_iris_is_replaced_without_caller_retry_code(
        self, session
    ):
        session.execute("SELECT 1")
        _terminate_server_job(session)

        # No retry loop here on purpose: this is the caller's whole experience.
        assert session.one("SELECT 1") == (1,)
        assert session.is_connected is True

    def test_asking_whether_it_is_connected_costs_no_round_trip(
        self, session, monkeypatch
    ):
        session.execute("SELECT 1")
        probes = _count_pings(session, monkeypatch)

        for _ in range(20):
            assert session.is_connected is True

        assert probes == []


class TestUserStory3StatementsSafeToRunAndRead:
    """Rows outlive their cursor, failures leak nothing, threads do not share."""

    def test_values_are_readable_after_the_session_is_closed(self, session):
        rows = session.execute(
            "SELECT TOP 3 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = ?",
            ("RAG",),
        )
        session.close()

        # The driver's row objects die with their cursor; these are plain tuples.
        assert len(rows) == 3
        assert all(isinstance(row[0], str) for row in rows)
        assert rows.dicts()[0]["TABLE_NAME"] == rows[0][0]

    def test_a_rejected_statement_leaves_the_connection_usable(self, session):
        before = session.connection_for_adapter()

        for _ in range(20):
            with pytest.raises(Exception) as caught:
                session.execute("SELECT * FROM RAG.NoSuchTableHere")
            assert not isinstance(
                caught.value, SessionError
            ), "IRIS rejecting a statement is not a session failure"

        assert session.connection_for_adapter() is before  # never reconnected
        assert session.one("SELECT 1") == (1,)

    def test_a_trailing_semicolon_is_accepted_by_the_session(self, session):
        # IRIS rejects it; callers keep writing it.
        assert session.one("SELECT 1;") == (1,)

    def test_each_thread_talks_to_its_own_server_job(self, session):
        import threading

        jobs = []
        barrier = threading.Barrier(4)

        def run():
            barrier.wait()
            jobs.append(session.one("SELECT $JOB")[0])

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(jobs) == 4
        assert len(set(jobs)) == 4, f"threads shared a connection: {jobs}"
        session.close_all()

    def test_a_batch_write_and_read_round_trips(self, session):
        session.execute("DROP TABLE IF EXISTS RAG.SessionE2E")
        session.execute(
            "CREATE TABLE RAG.SessionE2E (doc_id VARCHAR(64), note VARCHAR(200))"
        )
        try:
            written = session.execute_many(
                "INSERT INTO RAG.SessionE2E (doc_id, note) VALUES (?, ?)",
                [("d1", "one"), ("d2", "two"), ("d3", "three")],
            )
            session.connection_for_adapter().commit()

            assert written == 3
            assert session.one("SELECT COUNT(*) FROM RAG.SessionE2E") == (3,)
            assert session.execute(
                "SELECT note FROM RAG.SessionE2E WHERE doc_id = ?", ("d2",)
            ).one() == ("two",)
        finally:
            session.execute("DROP TABLE IF EXISTS RAG.SessionE2E")
            session.connection_for_adapter().commit()


class TestUserStory4DriverChoiceIsADecision:
    """Three environments, three outcomes, and the choice is on the record."""

    def test_this_process_reports_which_driver_it_chose(self, caplog):
        import logging as _logging

        from iris_vector_rag.session.drivers import select_driver

        identity = SessionIdentity.from_env()
        with caplog.at_level(_logging.INFO, logger="iris_vector_rag.session.drivers"):
            driver = select_driver(identity)

        assert driver.name in ("embedded", "network")
        assert driver.name in caplog.text
        assert identity.label in caplog.text

    def test_the_chosen_driver_is_the_one_that_actually_connects(self):
        from iris_vector_rag.session.drivers import select_driver

        identity = SessionIdentity.from_env()
        driver = select_driver(identity)

        connection = driver.connect(identity)
        try:
            assert driver.ping(connection) is True
        finally:
            driver.close(connection)

    def test_neither_driver_usable_names_both_and_dials_nothing(self, monkeypatch):
        from iris_vector_rag.session.drivers import (
            EmbeddedDriver,
            NetworkDriver,
            select_driver,
        )
        from iris_vector_rag.session.errors import NoUsableDriver

        monkeypatch.setattr(EmbeddedDriver, "available", lambda self: False)
        monkeypatch.setattr(NetworkDriver, "available", lambda self: False)

        with pytest.raises(NoUsableDriver) as caught:
            select_driver(SessionIdentity.from_env())

        assert "embedded" in str(caught.value)
        assert "network" in str(caught.value)

    def test_a_substituted_driver_reaches_no_instance_at_all(self):
        from iris_vector_rag.testing.fake_driver import FakeDriver

        # Nothing is listening on this port. A session given a driver never asks.
        nowhere = SessionIdentity(
            host="127.0.0.1", port=59999, namespace="USER", username="_SYSTEM"
        )
        session = IRISSession(identity=nowhere, driver=FakeDriver(rows=[(1,)]))

        assert session.one("SELECT 1") == (1,)
        assert session.driver_name == "fake"


class TestUserStory5TheOldNamesStillWork:
    """Superseded entry points keep working against the real instance, and warn once.

    The migration this feature asks for is a rename, not a rewrite: the names 0.14
    shipped still hand out a connection, and it is the session's -- so a caller who
    upgrades and changes nothing gets the bounded handshake and the health cadence
    for free, plus one warning naming what to move to.
    """

    def test_every_superseded_name_answers_from_the_one_session(self, session):
        import warnings as _warnings

        from iris_vector_rag.common.connection_manager import LegacyConnectionManager
        from iris_vector_rag.common.iris_connection import get_iris_connection
        from iris_vector_rag.common.iris_connection_manager import IRISConnectionManager
        from iris_vector_rag.common.iris_dbapi_connector import (
            get_iris_dbapi_connection,
        )

        expected = session.connection_for_adapter()
        calls = {
            "get_iris_connection": get_iris_connection,
            "get_iris_dbapi_connection": get_iris_dbapi_connection,
            "IRISConnectionManager": lambda: IRISConnectionManager().get_connection(),
            "LegacyConnectionManager": lambda: LegacyConnectionManager().connect(),
        }

        for name, call in calls.items():
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                connection = call()
            deprecations = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]

            assert connection is expected, f"{name} did not answer from the session"
            assert len(deprecations) == 1, f"{name} warned {len(deprecations)} times"
            assert "session" in str(deprecations[0].message).lower()

        # Four entry points, one connection, one session.
        assert len(open_sessions()) == 1
        assert session.one("SELECT 1") == (1,)

    def test_the_api_pool_serves_concurrent_requests_from_sessions(self, session):
        """What the HTTP API does at startup, and what its request threads then do.

        `fastapi` is not installed in the test environment, so this exercises the
        layer the API depends on: the pool it builds at startup, used the way its
        routes use it -- many threads at once, each running a statement.

        Twelve requests over a capacity of three, on a Community Edition licence that
        refuses the sixth connection: the pool has to hold the total down, which is
        the whole of what is left of its job now the session owns the connections.
        """
        import threading

        from iris_vector_rag.common.connection_pool import _session_pool

        identity = SessionIdentity.from_env()
        pool = _session_pool(
            host=identity.host,
            port=identity.port,
            namespace=identity.namespace,
            username=identity.username,
            password=identity.password,
            pool_size=2,
            max_overflow=1,
        )

        answers: list[object] = []
        failures: list[BaseException] = []
        lock = threading.Lock()

        def serve_one_request():
            try:
                with pool.get_connection() as connection:
                    cursor = connection.cursor()
                    try:
                        cursor.execute("SELECT TOP 1 1 FROM %Dictionary.CompiledClass")
                        row = cursor.fetchone()
                        value = row[0] if row else None
                    finally:
                        cursor.close()
                with lock:
                    answers.append(value)
            except BaseException as exc:  # recorded, not raised in a thread
                with lock:
                    failures.append(exc)

        threads = [threading.Thread(target=serve_one_request) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert failures == [], failures
        assert len(answers) == 12
        assert {int(a) for a in answers} == {1}

        stats = pool.get_stats()
        assert stats["in_use_connections"] == 0
        assert stats["total_acquisitions"] == 12
        assert stats["capacity"] == 3

        # Every request came from the one session for this identity, and the licence
        # never saw more connections than the pool advertises -- nor, now the requests
        # have finished, any at all.
        assert pool.session is session
        assert session.open_connection_count == 0

        # A request in flight is what "connected" reports on.
        with pool.get_connection():
            assert pool.get_stats()["connected"] is True
            assert pool.get_stats()["open_connections"] == 1

    def test_the_removed_names_are_removed(self):
        """0.15 deletes the duplicates outright: they had no caller to break."""
        import importlib

        from iris_vector_rag.common import connection_pool, iris_connection

        assert not hasattr(iris_connection, "IRISConnectionPool")
        assert not hasattr(connection_pool, "get_pool")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("iris_vector_rag.common.iris_connector")
