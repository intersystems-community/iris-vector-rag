"""
Simple IRIS client for database operations
"""

import logging
from typing import Any, Dict, List, Optional

from .iris_connection_manager import IRISConnectionManager

logger = logging.getLogger(__name__)


class IRISClient:
    """Simple IRIS database client"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize IRIS client with configuration"""
        self.config = config or {}
        self.connection_manager = IRISConnectionManager()
        self._connection = None

    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self._connection = self.connection_manager.get_connection(self.config)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IRIS: {e}")
            return False

    def query(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        if not self._connection:
            if not self.connect():
                raise ConnectionError("No database connection available")

        try:
            cursor = self._connection.cursor()
            cursor.execute(sql, params or ())

            # Get column names
            columns = (
                [desc[0] for desc in cursor.description] if cursor.description else []
            )

            # Fetch results
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))

            return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """Execute non-query SQL statement"""
        if not self._connection:
            if not self.connect():
                raise ConnectionError("No database connection available")

        try:
            cursor = self._connection.cursor()
            cursor.execute(sql, params or ())
            self._connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Execute failed: {e}")
            self._connection.rollback()
            raise

    def close(self):
        """Close database connection"""
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
