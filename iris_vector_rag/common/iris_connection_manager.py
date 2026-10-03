"""
Deprecated IRIS connection manager, kept importable for 0.15.

Every name here hands out the session's connection now. `get_iris_connection` stays
re-exported from `iris_connection` because tests and fixtures import it from this
module; `get_iris_connection_old`, which ignored its own arguments and had no
caller, is gone.

Removed in 0.16. Use `iris_vector_rag.session.session_for(...)`.

Feature: 083-iris-session (FR-015, FR-016)
"""

import logging
import warnings
from typing import Any, Dict, Optional

from iris_vector_rag.common.iris_connection import (  # noqa: F401 - re-exported
    _session_connection,
    get_iris_connection,
)

logger = logging.getLogger(__name__)

_REPLACEMENT = (
    "Use iris_vector_rag.session.session_for(...).connection_for_adapter(); "
    "removed in 0.16."
)


def _from_config(config: Optional[Dict[str, Any]]) -> Any:
    """The session's connection for a plain connection-parameter mapping."""
    config = config or {}
    return _session_connection(
        config.get("hostname") or config.get("host"),
        config.get("port"),
        config.get("namespace"),
        config.get("username"),
        config.get("password"),
    )


class IRISConnectionManager:
    """Deprecated: one session already owns the connection this handed out."""

    def __init__(self, prefer_dbapi: bool = True):
        warnings.warn(
            f"IRISConnectionManager is deprecated. {_REPLACEMENT}",
            DeprecationWarning,
            stacklevel=2,
        )
        self.prefer_dbapi = prefer_dbapi
        self._connection = None

    def get_connection(self, config: Optional[Dict[str, Any]] = None) -> Any:
        if self._connection is not None:
            return self._connection
        self._connection = _from_config(config)
        return self._connection

    def close(self):
        """Forget the connection without closing it: the session owns it."""
        self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_iris_dbapi_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """Deprecated: the session's connection for these parameters."""
    warnings.warn(
        f"get_iris_dbapi_connection() is deprecated. {_REPLACEMENT}",
        DeprecationWarning,
        stacklevel=2,
    )
    return _from_config(config)
