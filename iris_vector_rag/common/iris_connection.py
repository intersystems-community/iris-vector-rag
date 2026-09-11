"""
Deprecated IRIS connection entry point, kept callable for 0.15.

Since 0.15 the session owns connections: `get_iris_connection()` resolves its
arguments into a `SessionIdentity` and hands out the session's connection, so the
socket precheck, the bounded handshake, the health cadence and the reconnect apply
to callers that have not migrated yet. It keeps its signature and its validation
errors, warns once per call site, and is removed in 0.16.

    from iris_vector_rag.session import session_for
    connection = session_for(config_manager).connection_for_adapter()

`detect_iris_edition`, `auto_detect_iris_port` and `_get_iris_dbapi_module` are not
deprecated: they answer questions about the environment, not about connections.

Feature: 051-simplify-iris-connection
Feature: 083-iris-session (deprecation)
"""

import logging
import os
import re
import subprocess
import warnings
from typing import Any, Dict, Optional, Tuple

from iris_vector_rag.common.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Module-level edition cache (session-wide)
_edition_cache: Optional[Tuple[str, int]] = None


def _get_iris_dbapi_module():
    """
    Import the IRIS connectivity module.

    Prefers iris-embedded-python-wrapper (iris.dbapi) which provides
    a unified connection API for both embedded and external modes.
    Falls back to intersystems-irispython (iris.connect) if wrapper unavailable.
    """
    try:
        import iris

        if hasattr(iris, "dbapi") and hasattr(iris.dbapi, "connect"):
            return iris
        if hasattr(iris, "connect") or hasattr(iris, "createConnection"):
            return iris
        logger.warning(
            "iris module imported from %s but has no usable connect method",
            getattr(iris, "__file__", "unknown"),
        )
    except Exception as e:
        logger.error("Failed to import iris module: %s: %s", type(e).__name__, e)

    logger.error(
        "InterSystems IRIS Python SDK not available. "
        "Install with: pip install intersystems-irispython>=5.1.2"
    )
    return None


def auto_detect_iris_port() -> Optional[int]:
    """Auto-detect running IRIS instance and its SuperServer port."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "iris" in line.lower() and "1972" in line:
                    match = re.search(r"0\.0\.0\.0:(\d+)->1972/tcp", line)
                    if match:
                        return int(match.group(1))
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["iris", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                if "status:" in line and "running" in line:
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "SuperServers:" in lines[j]:
                            match = re.search(r"SuperServers:\s+(\d+)", lines[j])
                            if match:
                                return int(match.group(1))
    except Exception:
        pass
    return None


def _validate_connection_params(h: str, p: int, n: str, u: str, pwd: str) -> None:
    if not h or not h.strip():
        raise ValidationError("host", h, "non-empty string", "Host cannot be empty")
    if not isinstance(p, int) or p < 1 or p > 65535:
        raise ValidationError("port", p, "1-65535", f"Invalid port: {p}")
    if not n or not n.strip() or not re.match(r"^[A-Za-z0-9_]+$", n):
        raise ValidationError(
            "namespace", n, "alphanumeric + underscores", f"Invalid namespace: {n}"
        )


def _get_connection_params_from_env() -> Dict[str, Any]:
    port_env = os.environ.get("IRIS_PORT")
    port = int(port_env) if port_env else (auto_detect_iris_port() or 1972)
    return {
        "host": os.environ.get("IRIS_HOST", "localhost"),
        "port": port,
        "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
        "username": os.environ.get(
            "IRIS_USER", os.environ.get("IRIS_USERNAME", "_SYSTEM")
        ),
        "password": os.environ.get("IRIS_PASSWORD", "SYS"),
    }


def detect_iris_edition() -> Tuple[str, int]:
    global _edition_cache
    if _edition_cache is not None:
        return _edition_cache
    backend_mode = os.environ.get("IRIS_BACKEND_MODE", "").lower()
    if backend_mode in ("community", "enterprise"):
        max_connections = 1 if backend_mode == "community" else 999
        _edition_cache = (backend_mode, max_connections)
        return _edition_cache
    _edition_cache = ("community", 1)
    return _edition_cache


def get_iris_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    namespace: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """The session's connection for these parameters. Deprecated since 0.15.

    The socket precheck, the bounded handshake, the expired-password recovery and
    the module-level cache this function used to own are the session's now, so a
    caller that has not migrated still gets all of them -- and shares one
    connection with the callers that have, which a second cache would not.
    """
    warnings.warn(
        "get_iris_connection() is deprecated and is removed in 0.16. Use "
        "iris_vector_rag.session.session_for(...).connection_for_adapter(), or "
        "session_for(...).execute(sql) to skip the cursor entirely.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _session_connection(host, port, namespace, username, password)


def _session_connection(
    host: Optional[str],
    port: Optional[int],
    namespace: Optional[str],
    username: Optional[str],
    password: Optional[str],
) -> Any:
    """Resolve these parameters to a session and hand out its connection.

    Kept separate from the deprecated name so the library's own remaining callers
    do not raise a warning their users cannot act on.
    """
    from iris_vector_rag.session import SessionIdentity, session_for

    env = _get_connection_params_from_env()
    h = env["host"] if host is None else host
    p = env["port"] if port is None else port
    n = env["namespace"] if namespace is None else namespace
    u = env["username"] if username is None else username
    pwd = env["password"] if password is None else password

    # Validated here rather than in the identity so that the parameter names in
    # the error are the ones the caller passed (ValidationError.parameter_name).
    _validate_connection_params(h, p, n, u, pwd)

    identity = SessionIdentity(
        host=h, port=int(p), namespace=n, username=u, password=pwd
    )
    return session_for(identity=identity).connection_for_adapter()
