"""The IRIS session: one object that owns one instance-and-namespace relationship.

    from iris_vector_rag.session import session_for

    session = session_for(config_manager)
    rows = session.execute("SELECT TOP 5 doc_id FROM RAG.SourceDocuments")

Feature: 083-iris-session
"""

from iris_vector_rag.session.drivers import (
    Driver,
    EmbeddedDriver,
    NetworkDriver,
    select_driver,
)
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
from iris_vector_rag.session.identity import SessionIdentity
from iris_vector_rag.session.registry import (
    close_all_sessions,
    open_sessions,
    session_for,
)
from iris_vector_rag.session.results import Rows
from iris_vector_rag.session.session import IRISSession

__all__ = [
    # the two names most callers need
    "session_for",
    "IRISSession",
    # identity and results
    "SessionIdentity",
    "Rows",
    # drivers
    "Driver",
    "EmbeddedDriver",
    "NetworkDriver",
    "select_driver",
    # errors
    "SessionError",
    "IRISUnreachable",
    "HandshakeTimeout",
    "ConnectFailed",
    "InBackoff",
    "NoUsableDriver",
    "PasswordExpired",
    "ConnectionLost",
    "CallerConnectionBroken",
    # registry
    "open_sessions",
    "close_all_sessions",
]
