"""
Connector for InterSystems IRIS using Python DBAPI.
Unified to use iris_connection.py for consistency and UV compatibility.
"""

import logging
from typing import Any, Dict, Optional

from iris_vector_rag.common.iris_connection import get_iris_connection, auto_detect_iris_port

logger = logging.getLogger(__name__)


def get_iris_dbapi_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Unified entry point for IRIS DBAPI connections.
    """
    if config:
        # Map parameters to get_iris_connection expected names if needed
        return get_iris_connection(
            host=config.get("hostname") or config.get("host"),
            port=config.get("port"),
            namespace=config.get("namespace"),
            username=config.get("username"),
            password=config.get("password")
        )
    return get_iris_connection()


def get_iris_dbapi_module():
    """
    Access the IRIS DBAPI module.
    """
    from iris_vector_rag.common.iris_connection import _get_iris_dbapi_module
    return _get_iris_dbapi_module()


# Backward compatibility
irisdbapi = get_iris_dbapi_module()
