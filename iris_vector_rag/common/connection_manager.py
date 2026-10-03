"""
The ODBC/JDBC-era connection manager, renamed and kept working for 0.15.

Renamed from `ConnectionManager` to `LegacyConnectionManager` so that the name
`ConnectionManager` identifies one class: `iris_vector_rag.core.connection`'s, which
is what storage and the pipelines are handed (SC-001).

Its JDBC and ODBC branches are gone with it. Neither reached IRIS: the JDBC branch
imported a name that does not exist and the ODBC branch went through a third
`get_iris_connection`, now deleted. Every connection type resolves to the session,
which is the only thing that connects.

Removed in 0.16. Use `iris_vector_rag.session.session_for(...)`.

Feature: 083-iris-session (FR-015, SC-001)
"""

import logging
import os
import warnings
from contextlib import contextmanager
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

VALID_CONNECTION_TYPES = ("jdbc", "odbc", "dbapi")


class LegacyConnectionManager:
    """Deprecated: a cursor-and-execute wrapper over what the session now owns."""

    def __init__(self, connection_type: str = "dbapi"):
        warnings.warn(
            "LegacyConnectionManager (formerly ConnectionManager) is deprecated and "
            "is removed in 0.16. Use iris_vector_rag.session.session_for(...), whose "
            "execute() returns rows that outlive their cursor.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.connection_type = connection_type.lower()
        if self.connection_type not in VALID_CONNECTION_TYPES:
            raise ValueError(
                f"Invalid connection type: {connection_type}. "
                f"Must be one of {', '.join(VALID_CONNECTION_TYPES)}"
            )
        self._session = None

    @property
    def session(self):
        if self._session is None:
            from iris_vector_rag.session import session_for

            self._session = session_for()
        return self._session

    def connect(self):
        """The session's connection for this thread."""
        return self.session.connection_for_adapter()

    def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """Rows for a SELECT, an empty list for anything else."""
        rows = self.session.execute(query, tuple(params) if params else None)
        return list(rows)

    def execute_many(self, query: str, params_list: List[List[Any]]) -> None:
        self.session.execute_many(query, [tuple(params) for params in params_list])
        self.connect().commit()

    @contextmanager
    def cursor(self):
        """A driver cursor, closed on the way out."""
        cursor = self.connect().cursor()
        try:
            yield cursor
        finally:
            try:
                cursor.close()
            except Exception:  # pragma: no cover - a cursor that cannot close
                pass

    def close(self):
        """Forget the session. The connection belongs to it, not to this wrapper."""
        self._session = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Global instance, kept for callers that configured a connection type at startup.
_global_connection_manager: Optional[LegacyConnectionManager] = None


def get_connection_manager(
    connection_type: Optional[str] = None,
) -> LegacyConnectionManager:
    """Deprecated: get or create the global legacy manager."""
    global _global_connection_manager

    if connection_type:
        return LegacyConnectionManager(connection_type)

    if _global_connection_manager is None:
        default_type = os.environ.get("RAG_CONNECTION_TYPE", "dbapi")
        _global_connection_manager = LegacyConnectionManager(default_type)

    return _global_connection_manager


def set_global_connection_type(connection_type: str):
    """Deprecated: the session chooses the driver and says which it chose."""
    global _global_connection_manager
    _global_connection_manager = LegacyConnectionManager(connection_type)
    logger.info("Set global connection type to: %s", connection_type)
