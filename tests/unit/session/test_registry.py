"""One session per identity per process.

Feature: 083-iris-session (FR-002, FR-020)
"""

import pytest

from iris_vector_rag.session import SessionIdentity, session_for
from iris_vector_rag.session.registry import open_sessions, reset_sessions_for_tests
from iris_vector_rag.testing.fake_driver import FakeDriver


def identity(**overrides):
    fields = {
        "host": "localhost",
        "port": 51972,
        "namespace": "USER",
        "username": "_SYSTEM",
        "password": "SYS",
    }
    fields.update(overrides)
    return SessionIdentity(**fields)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_sessions_for_tests()
    yield
    reset_sessions_for_tests()


def test_the_same_identity_yields_the_same_session():
    first = session_for(identity=identity(), driver=FakeDriver())
    second = session_for(identity=identity(), driver=FakeDriver())

    assert first is second


def test_a_different_namespace_yields_a_different_session():
    first = session_for(identity=identity(namespace="USER"), driver=FakeDriver())
    second = session_for(identity=identity(namespace="RAG"), driver=FakeDriver())

    assert first is not second
    assert len(open_sessions()) == 2


def test_a_different_password_does_not_yield_a_second_session():
    first = session_for(identity=identity(password="SYS"), driver=FakeDriver())
    second = session_for(identity=identity(password="other"), driver=FakeDriver())

    assert first is second


def test_the_first_callers_cadence_is_the_sessions_and_a_later_one_is_told(caplog):
    """The second caller's `health_interval` cannot win, so it is not lost silently.

    Two callers share one session, and the pool's `pool_recycle` arrives as a
    `health_interval`. Whichever asked first decided; a later, different request
    would otherwise be dropped with nothing said (FR-002).
    """
    import logging

    first = session_for(identity=identity(), driver=FakeDriver(), health_interval=30.0)

    with caplog.at_level(logging.WARNING, logger="iris_vector_rag.session.registry"):
        second = session_for(
            identity=identity(), driver=FakeDriver(), health_interval=45.0
        )

    assert second is first
    assert first.health_interval == 30.0
    assert "45" in caplog.text and "30" in caplog.text


def test_the_ignored_cadence_is_reported_once_not_per_call(caplog):
    """The pool asks on every request; a per-request warning is not a warning.

    `IRISConnectionPool.session` resolves the session on each acquire and release, so
    a caller whose `pool_recycle` lost the race would otherwise get two lines of log
    per served request -- which is how a real warning gets filtered out (FR-002).
    """
    import logging

    session_for(identity=identity(), driver=FakeDriver(), health_interval=30.0)

    with caplog.at_level(logging.WARNING, logger="iris_vector_rag.session.registry"):
        for _ in range(20):
            session_for(identity=identity(), health_interval=45.0)

    assert caplog.text.count("ignoring the requested") == 1


def test_the_same_cadence_asked_twice_says_nothing(caplog):
    import logging

    session_for(identity=identity(), driver=FakeDriver(), health_interval=45.0)

    with caplog.at_level(logging.WARNING, logger="iris_vector_rag.session.registry"):
        session_for(identity=identity(), driver=FakeDriver(), health_interval=45.0)

    assert caplog.text == ""


def test_acquiring_a_session_opens_no_connection():
    driver = FakeDriver()

    session = session_for(identity=identity(), driver=driver)

    assert driver.connects == 0
    assert session.is_connected is False


def test_reset_closes_what_it_clears():
    driver = FakeDriver()
    session = session_for(identity=identity(), driver=driver)
    session.execute("SELECT 1")
    assert session.is_connected is True

    reset_sessions_for_tests()

    assert driver.closes == 1
    assert open_sessions() == {}


def test_a_borrowed_connection_session_is_not_registered():
    from iris_vector_rag.session import IRISSession
    from iris_vector_rag.testing.fake_driver import FakeConnection

    borrowed = IRISSession.from_connection(FakeConnection(FakeDriver()))

    assert open_sessions() == {}
    assert borrowed is not session_for(identity=identity(), driver=FakeDriver())
