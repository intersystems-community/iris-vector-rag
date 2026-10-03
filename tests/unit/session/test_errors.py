"""Every session failure names the instance and stays catchable as ConnectionError.

Feature: 083-iris-session (FR-019)
"""

import pytest

from iris_vector_rag.session import SessionIdentity
from iris_vector_rag.session.errors import (
    CallerConnectionBroken,
    ConnectFailed,
    ConnectionLost,
    HandshakeTimeout,
    InBackoff,
    IRISUnreachable,
    NoUsableDriver,
    PasswordExpired,
    SessionError,
)

IDENTITY = SessionIdentity(
    host="localhost",
    port=51972,
    namespace="USER",
    username="_SYSTEM",
    password="SYS",
)

ERRORS = [
    IRISUnreachable,
    HandshakeTimeout,
    ConnectFailed,
    InBackoff,
    NoUsableDriver,
    PasswordExpired,
    ConnectionLost,
    CallerConnectionBroken,
]


def test_only_the_cheap_precheck_failure_leaves_the_backoff_window_closed():
    # The socket precheck costs two seconds and answers a question that changes
    # the moment IRIS starts. Everything past it cost a whole connect budget.
    assert IRISUnreachable.opens_backoff is False
    assert HandshakeTimeout.opens_backoff is True
    assert ConnectFailed.opens_backoff is True


@pytest.mark.parametrize("error_type", ERRORS)
def test_every_error_names_the_instance_and_namespace(error_type):
    error = error_type(IDENTITY, "because reasons")

    assert "localhost:51972/USER" in str(error)
    assert "because reasons" in str(error)
    assert error.identity is IDENTITY


@pytest.mark.parametrize("error_type", ERRORS)
def test_every_error_is_still_a_connection_error(error_type):
    # Call sites written against the previous connection module catch
    # ConnectionError; none of them should have to change.
    assert issubclass(error_type, SessionError)
    assert issubclass(error_type, ConnectionError)


@pytest.mark.parametrize("error_type", ERRORS)
def test_no_error_leaks_the_password(error_type):
    assert "SYS" not in str(error_type(IDENTITY, "detail"))


# Verbatim from intersystems-irispython against IRIS 2026.1, collected by killing a
# connection's server-side job while a statement was in flight (Feature 083 E2E).
LOST_CONNECTION_MESSAGES = [
    "<COMMUNICATION LINK ERROR> Connection closed",
    "<COMMUNICATION LINK ERROR> Failed to receive message; Details: "
    "Error code: 60 Error message: Communication timed out",
    "<COMMUNICATION LINK ERROR> Failed to send message",
]

# Statements IRIS understood and rejected. Retrying these on a fresh connection buys
# the same failure at the price of a handshake.
STATEMENT_MESSAGES = [
    "[SQLCODE: <-30>:<Table or view not found>] Table 'RAG.BOGUS' not found",
    "[SQLCODE: <-1>:<Invalid SQL statement>]",
    "[SQLCODE: <-104>:<Field validation failed>]",
]


@pytest.mark.parametrize("message", LOST_CONNECTION_MESSAGES)
def test_the_drivers_own_transport_failures_are_recognised(message):
    from iris_vector_rag.session.errors import looks_like_lost_connection

    assert looks_like_lost_connection(RuntimeError(message)) is True


@pytest.mark.parametrize("message", STATEMENT_MESSAGES)
def test_a_statement_iris_rejected_is_not_a_lost_connection(message):
    from iris_vector_rag.session.errors import looks_like_lost_connection

    assert looks_like_lost_connection(ValueError(message)) is False


def test_the_failure_modes_are_distinct_types():
    assert len({e.__name__ for e in ERRORS}) == len(ERRORS)
    for error_type in ERRORS:
        others = [o for o in ERRORS if o is not error_type]
        assert not any(issubclass(error_type, other) for other in others)


def test_an_os_error_is_a_transport_failure_whatever_it_says():
    """The transport failing is the transport failing; the text is the driver's."""
    from iris_vector_rag.session.errors import looks_like_lost_connection

    assert looks_like_lost_connection(OSError("errno 54")) is True
    assert looks_like_lost_connection(BrokenPipeError()) is True
    assert looks_like_lost_connection(EOFError()) is True


def test_a_session_error_is_never_retried_on_a_fresh_connection():
    """The session already classified it; a reconnect would only cost a reconnect."""
    from iris_vector_rag.session.errors import (
        InBackoff,
        IRISUnreachable,
        looks_like_lost_connection,
    )
    from iris_vector_rag.testing.fake_driver import fake_identity

    assert looks_like_lost_connection(IRISUnreachable(fake_identity(), "no")) is False
    assert looks_like_lost_connection(InBackoff(fake_identity(), "later")) is False
