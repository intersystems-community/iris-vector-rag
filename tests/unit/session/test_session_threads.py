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
