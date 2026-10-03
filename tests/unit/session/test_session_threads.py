"""Two threads never share one connection, and a forked child inherits none.

Feature: 083-iris-session (FR-003, US3)
"""

import threading

from iris_vector_rag.testing.fake_driver import fake_session


def test_each_thread_gets_its_own_connection():
    session = fake_session(rows=[(1,)], health_interval=3600)
    seen = []
    barrier = threading.Barrier(4)

    def run():
        barrier.wait()
        session.execute("SELECT 1")
        seen.append(id(session.connection_for_adapter()))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(seen)) == 4
    assert session.driver.connects == 4


def test_the_session_says_how_many_connections_it_has_open():
    """A caller bounded by a licence has to be able to ask (FR-017)."""
    session = fake_session(rows=[(1,)], health_interval=3600)
    assert session.open_connection_count == 0

    session.execute("SELECT 1")
    assert session.open_connection_count == 1

    session.close_all()
    assert session.open_connection_count == 0


def test_a_thread_that_exits_takes_its_connection_with_it():
    """A licence counts connections, and a finished thread cannot use one (FR-017).

    A server thread that serves a request and exits without closing anything used to
    leave its connection open for the life of the process: five such threads exhaust
    a Community Edition licence while nothing at all is running.
    """
    import gc

    session = fake_session(rows=[(1,)], health_interval=3600)

    def run():
        session.execute("SELECT 1")

    for _ in range(8):
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()

    gc.collect()

    assert session.open_connection_count == 0
    assert session.driver.connects == 8
    assert all(connection.closed for connection in session.driver.connections)


def test_a_live_threads_connection_is_not_reaped():
    """Only the connections of threads that have finished go back."""
    import gc

    session = fake_session(rows=[(1,)], health_interval=3600)
    running = threading.Event()
    release = threading.Event()

    def run():
        session.execute("SELECT 1")
        running.set()
        release.wait(timeout=10)

    thread = threading.Thread(target=run)
    thread.start()
    running.wait(timeout=10)

    session.execute("SELECT 1")  # this thread's own
    gc.collect()

    assert session.open_connection_count == 2

    release.set()
    thread.join()
    gc.collect()

    assert session.open_connection_count == 1


def test_no_thread_observes_a_half_built_session():
    session = fake_session(rows=[(1,)], health_interval=3600)
    errors = []

    def run():
        try:
            for _ in range(20):
                session.execute("SELECT 1")
        except Exception as exc:  # pragma: no cover - the point is that it is empty
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(session.driver.statements) == 80


def test_a_connection_from_another_process_is_not_reused(monkeypatch):
    session = fake_session(rows=[(1,)], health_interval=3600)
    session.execute("SELECT 1")
    inherited = session.driver.connections[0]

    monkeypatch.setattr("os.getpid", lambda: 999999)
    session.execute("SELECT 1")

    assert session.driver.connects == 2
    assert session.connection_for_adapter() is not inherited


def test_closing_in_one_thread_leaves_another_threads_connection_alone():
    session = fake_session(rows=[(1,)], health_interval=3600)
    session.execute("SELECT 1")
    main_connection = session.connection_for_adapter()

    def run():
        session.execute("SELECT 1")
        session.close()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert main_connection.closed is False
    assert session.is_connected is True
