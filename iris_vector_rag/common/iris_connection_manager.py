"""
IRIS Connection Manager - Unified to use iris_connection.py.
"""

import logging
import warnings
from typing import Any, Dict, Optional

from iris_vector_rag.common.iris_connection import get_iris_connection, detect_iris_edition

logger = logging.getLogger(__name__)


class IRISConnectionManager:
    """
    Deprecated: Use get_iris_connection() directly.
    """

    def __init__(self, prefer_dbapi: bool = True):
        warnings.warn(
            "IRISConnectionManager is deprecated. Use get_iris_connection() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.prefer_dbapi = prefer_dbapi
        self._connection = None

    def get_connection(self, config: Optional[Dict[str, Any]] = None) -> Any:
        if self._connection is not None:
            return self._connection
        
        if config:
            self._connection = get_iris_connection(
                host=config.get("hostname") or config.get("host"),
                port=config.get("port"),
                namespace=config.get("namespace"),
                username=config.get("username"),
                password=config.get("password")
            )
        else:
            self._connection = get_iris_connection()
            
        return self._connection

    def close(self):
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


def get_iris_dbapi_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    return get_iris_connection(
        host=config.get("hostname") or config.get("host") if config else None,
        port=config.get("port") if config else None,
        namespace=config.get("namespace") if config else None,
        username=config.get("username") if config else None,
        password=config.get("password") if config else None
    )

def get_iris_connection_old(config: Optional[Dict[str, Any]] = None, prefer_dbapi: bool = True) -> Any:
    return get_iris_connection()
