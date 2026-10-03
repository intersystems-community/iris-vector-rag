"""The distinct ways talking to IRIS fails.

Every one of them names the instance and namespace, and every one of them is still
a builtin ``ConnectionError`` so that the call sites written against the previous
connection module keep catching what they always caught.

Feature: 083-iris-session (FR-019)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from iris_vector_rag.session.identity import SessionIdentity


_REFUSAL_MARKERS = (
    "refused",
    "unreachable",
    "not reachable",
    "timed out",
    "timeout",
    "communication link",
    "name or service not known",
)


def _unconfigured_and_unreachable(identity: object, detail: str) -> bool:
    return bool(getattr(identity, "from_defaults", False)) and any(
        marker in detail.lower() for marker in _REFUSAL_MARKERS
    )


def _default_hint(identity: object) -> str:
    return (
        f"No connection settings were configured, so the defaults were used: "
        f"{getattr(identity, 'host', '?')}:{getattr(identity, 'port', '?')} as user "
        f"{getattr(identity, 'username', '?')}. Set IRIS_HOST and IRIS_PORT, or set "
        f"IRIS_CONTAINER_NAME to the container to use"
    )


class SessionError(ConnectionError):
    """A session could not do what was asked of it.

    ``opens_backoff`` says whether this failure means "do not try again for a
    while". Only the failures that cost a whole connect budget do; the two-second
    socket precheck answers a question that changes the moment IRIS starts, so
    refusing to ask it again would make a caller wait longer than the truth.
    """

    opens_backoff: bool = False
    # Whether a failure of this kind, on settings nobody configured, earns the hint.
    hints_defaults: bool = False

    def __init__(self, identity: "SessionIdentity", detail: str = "") -> None:
        self.identity = identity
        self.detail = detail
        if self.hints_defaults and _unconfigured_and_unreachable(identity, detail):
            detail = f"{detail}. {_default_hint(identity)}" if detail else detail
        label = getattr(identity, "label", str(identity))
        message = (
            f"IRIS session {label}: {detail}" if detail else f"IRIS session {label}"
        )
        super().__init__(message)


class IRISUnreachable(SessionError):
    """The port is not accepting connections."""

    hints_defaults = True
    opens_backoff = False


class HandshakeTimeout(SessionError):
    """TCP connected but the driver handshake never completed in budget."""

    hints_defaults = True
    opens_backoff = True


class ConnectFailed(SessionError):
    """The driver refused the connection for a reason of its own."""

    hints_defaults = True
    opens_backoff = True


class InBackoff(SessionError):
    """A previous attempt burned the connect budget; this one fails immediately."""

    opens_backoff = False


class NoUsableDriver(SessionError):
    """Neither the embedded nor the network driver can be used in this process."""

    opens_backoff = False


class PasswordExpired(SessionError):
    """The account password is expired; the library recognises this and does not clear it."""

    opens_backoff = False


class ConnectionLost(SessionError):
    """An owned connection died and its replacement failed too."""

    opens_backoff = False


class CallerConnectionBroken(SessionError):
    """A connection the caller supplied failed; the session replaces nothing."""

    opens_backoff = False


_LOST_MARKERS = (
    "connection is closed",
    "connection closed",
    "connection reset",
    "not connected",
    "no connection",
    "connection lost",
    # The driver's own wording for a dead transport: "<COMMUNICATION LINK ERROR>
    # Failed to receive message" is what a killed server-side job looks like.
    "communication link",
    "failed to receive message",
    "failed to send message",
    "broken pipe",
    "socket",
    "server closed",
    "connection refused",
)


def looks_like_lost_connection(error: BaseException) -> bool:
    """Is this the connection dying rather than the statement being wrong?

    A statement that IRIS rejected must not be retried on a fresh connection: it
    would fail identically and cost a reconnect. Only transport-level failures are
    worth replacing a connection for.
    """
    if isinstance(error, (ConnectionLost, CallerConnectionBroken)):
        return True
    if isinstance(error, SessionError):
        return False
    if isinstance(error, (BrokenPipeError, ConnectionResetError, EOFError)):
        return True
    if isinstance(error, OSError):
        return True
    text = str(error).lower()
    return any(marker in text for marker in _LOST_MARKERS)


def wrap_connect_failure(
    identity: "SessionIdentity", error: BaseException, detail: Optional[str] = None
) -> SessionError:
    """Name a driver's connect failure in the session's own vocabulary."""
    if isinstance(error, SessionError):
        return error
    return ConnectFailed(identity, detail or f"{type(error).__name__}: {error}")
