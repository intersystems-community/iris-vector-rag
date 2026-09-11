"""One session per identity per process.

Two callers asking for the same instance-and-namespace get the same session, so the
health cadence, the back-off window and the driver choice are shared facts rather than
per-caller guesses. Acquiring a session never connects.

Feature: 083-iris-session
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Any, Dict, Optional, Tuple

from iris_vector_rag.session.identity import SessionIdentity
from iris_vector_rag.session.session import IRISSession

logger = logging.getLogger(__name__)

_sessions: Dict[Tuple[Any, ...], IRISSession] = {}
_lock = threading.RLock()


def session_for(
    config_manager: Optional[Any] = None,
    *,
    identity: Optional[SessionIdentity] = None,
    driver: Optional[Any] = None,
    health_interval: Optional[float] = None,
) -> IRISSession:
    """The session for this identity, creating it on first ask.

    Does not connect. The first statement does that.
    """
    resolved = SessionIdentity.resolve(config_manager, identity)
    with _lock:
        existing = _sessions.get(resolved.key)
        if existing is not None:
            return existing
        session = IRISSession(
            identity=resolved,
            driver=driver,
            health_interval=health_interval,
        )
        _sessions[resolved.key] = session
        logger.debug("opened a session for %s", resolved.label)
        return session


def open_sessions() -> Dict[Tuple[Any, ...], IRISSession]:
    """A copy of the registry, for diagnostics and tests."""
    with _lock:
        return dict(_sessions)


def close_all_sessions() -> None:
    """Close every registered session's connections and forget them."""
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        try:
            session.close_all()
        except Exception as exc:  # a teardown path never raises
            logger.debug("closing %s failed: %s", session.identity.label, exc)


def reset_sessions_for_tests() -> None:
    """Close and forget every session. Tests own the process; they may do this."""
    close_all_sessions()


atexit.register(close_all_sessions)
