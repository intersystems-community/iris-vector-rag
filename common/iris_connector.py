"""
Clean IRIS Database Connector
Uses DBAPI primarily, JDBC only as emergency fallback.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_iris_connection(
    config: Optional[Dict[str, Any]] = None, prefer_dbapi: bool = True
) -> Any:
    """
    Get an IRIS database connection, preferring DBAPI by default.

    Args:
        config: Optional configuration dictionary
        prefer_dbapi: Whether to prefer DBAPI over JDBC (default: True)

    Returns:
        Database connection object

    Raises:
        Exception: If both DBAPI and JDBC fail
    """
    # Always try DBAPI first
    try:
        from common.iris_dbapi_connector import get_iris_dbapi_connection

        conn = get_iris_dbapi_connection()

        # Validate the connection handle
        if conn is None:
            raise IRISConnectionError("DBAPI connection returned NULL handle")

        logger.info("✅ Using DBAPI connection")
        return conn
    except Exception as dbapi_error:
        logger.error(f"❌ DBAPI connection failed: {dbapi_error}")

        if not prefer_dbapi:
            # Only fall back to JDBC if explicitly requested
            try:
                from common.iris_connection_manager import get_iris_jdbc_connection

                conn = get_iris_jdbc_connection(config)

                # Validate the connection handle
                if conn is None:
                    raise IRISConnectionError("JDBC connection returned NULL handle")

                logger.warning(
                    "⚠️ Falling back to JDBC connection - this indicates a DBAPI problem!"
                )
                return conn
            except Exception as jdbc_error:
                logger.error(f"❌ JDBC fallback also failed: {jdbc_error}")
                raise IRISConnectionError(
                    f"Both DBAPI and JDBC connections failed. DBAPI: {dbapi_error}, JDBC: {jdbc_error}"
                )
        else:
            # If DBAPI fails and we prefer it, this is a critical error
            raise IRISConnectionError(
                f"DBAPI connection failed and fallback disabled: {dbapi_error}"
            )


class IRISConnectionError(Exception):
    """Custom exception for IRIS connection errors."""
