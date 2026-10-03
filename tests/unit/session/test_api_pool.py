"""The HTTP API's pool bounds connections, not just callers.

The pool the API keeps used to hold a free list of its own, and that list is what
kept the number of open connections at `pool_size`. The session gives each thread
its own connection instead, so a burst of short-lived request threads would open one
connection per thread -- and IRIS Community Edition refuses the sixth with "Unable to
allocate a license". The pool's remaining job is therefore to hold the total down to
the capacity it advertises.

Feature: 083-iris-session (FR-017, FR-018, US5)
"""

import threading

import pytest

from iris_vector_rag.common.connection_pool import _session_pool
from tests.support.fake_driver import FakeDriver, fake_identity


@pytest.fixture
def pool_over_one_session(monkeypatch):
    """A pool whose session never touches a socket."""
    import iris_vector_rag.common.connection_pool as pool_module
    from iris_vector_rag.session import IRISSession

    session = IRISSession(
        identity=fake_identity(),
        driver=FakeDriver(rows=[(1,)]),
        health_interval=3600,
    )
    monkeypatch.setattr(pool_module, "session_for", lambda *a, **k: session)

    def build(**overrides):
        parameters = {
            "host": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": "_SYSTEM",
            "password": "SYS",
        }
        parameters.update(overrides)
        return _session_pool(**parameters)

    return build, session


def _serve(pool, requests, workers):
    """`requests` requests over `workers` fresh threads, as a server would."""
    failures = []
    lock = threading.Lock()

    def one_request():
        try:
            with pool.get_connection() as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                finally:
                    cursor.close()
        except BaseException as exc:  # recorded: a thread cannot raise at the caller
            with lock:
                failures.append(exc)

    threads = [threading.Thread(target=one_request) for _ in range(requests)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert failures == [], failures


class TestConnectionsStayWithinCapacity:
    def test_a_burst_of_fresh_threads_does_not_outrun_the_capacity(
        self, pool_over_one_session
    ):
        build, session = pool_over_one_session
        pool = build(pool_size=2, max_overflow=1)

        _serve(pool, requests=12, workers=12)

        # 12 threads asked; the licence only ever sees the pool's capacity.
        assert session.open_connection_count <= pool.get_stats()["capacity"]
        assert pool.get_stats()["total_acquisitions"] == 12
        assert pool.get_stats()["in_use_connections"] == 0
        # Every request thread has finished, so nothing is held any more: a licence
        # counts connections, and a thread that has exited cannot use one.
        assert session.open_connection_count == 0
        assert pool.get_stats()["connected"] is False

    def test_a_thread_that_keeps_serving_keeps_its_connection(
        self, pool_over_one_session
    ):
        """Within capacity nothing is thrown away: that is the point of a session."""
        build, session = pool_over_one_session
        pool = build(pool_size=4, max_overflow=0)

        for _ in range(5):
            with pool.get_connection() as connection:
                assert connection is session.connection_for_adapter()

        assert session.driver.connects == 1
        assert session.open_connection_count == 1

    def test_the_pool_reports_what_the_session_has_open(self, pool_over_one_session):
        build, session = pool_over_one_session
        pool = build(pool_size=2)

        with pool.get_connection():
            serving = pool.get_stats()

        idle = pool.get_stats()

        assert serving["open_connections"] == 1
        assert serving["in_use_connections"] == 1
        assert idle["open_connections"] == session.open_connection_count == 1
        assert idle["in_use_connections"] == 0


class TestReleaseIsHonest:
    def test_releasing_more_often_than_acquiring_does_not_free_extra_slots(
        self, pool_over_one_session
    ):
        build, _ = pool_over_one_session
        pool = build(pool_size=1, max_overflow=0, pool_timeout=1)

        connection = pool.acquire()
        pool.release(connection)
        pool.release(connection)

        # The slot is free exactly once.
        assert pool.acquire() is not None
        with pytest.raises(TimeoutError):
            pool.acquire()
