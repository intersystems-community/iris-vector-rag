"""
Simplified Connection Manager that provides a working DBAPI-like interface.

This module creates a DBAPI-compatible connection that works around the
circular import issues in intersystems_iris by using JDBC under the hood
but presenting a DBAPI-like interface for testing.
"""
import logging
from typing import Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class SimplifiedCursor:
    """A simplified cursor that mimics DBAPI cursor interface."""
    
    def __init__(self, jdbc_cursor):
        self._jdbc_cursor = jdbc_cursor
        self._closed = False
    
    def execute(self, query: str, params: Optional[Tuple] = None):
        """Execute a query."""
        if self._closed:
            raise RuntimeError("Cursor is closed")
        
        if params:
            self._jdbc_cursor.execute(query, params)
        else:
            self._jdbc_cursor.execute(query)
    
    def fetchone(self) -> Optional[Tuple]:
        """Fetch one row."""
        if self._closed:
            raise RuntimeError("Cursor is closed")
        
        result = self._jdbc_cursor.fetchone()
        return result
    
    def fetchall(self) -> List[Tuple]:
        """Fetch all rows."""
        if self._closed:
            raise RuntimeError("Cursor is closed")
        
        return self._jdbc_cursor.fetchall()
    
    def fetchmany(self, size: int = 1) -> List[Tuple]:
        """Fetch many rows."""
        if self._closed:
            raise RuntimeError("Cursor is closed")
        
        return self._jdbc_cursor.fetchmany(size)
    
    def close(self):
        """Close the cursor."""
        if not self._closed:
            self._jdbc_cursor.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimplifiedConnection:
    """A simplified connection that mimics DBAPI connection interface."""
    
    def __init__(self, jdbc_connection):
        self._jdbc_connection = jdbc_connection
        self._closed = False
    
    def cursor(self) -> SimplifiedCursor:
        """Create a cursor."""
        if self._closed:
            raise RuntimeError("Connection is closed")
        
        jdbc_cursor = self._jdbc_connection.cursor()
        return SimplifiedCursor(jdbc_cursor)
    
    def commit(self):
        """Commit the transaction."""
        if self._closed:
            raise RuntimeError("Connection is closed")
        
        self._jdbc_connection.commit()
    
    def rollback(self):
        """Rollback the transaction."""
        if self._closed:
            raise RuntimeError("Connection is closed")
        
        self._jdbc_connection.rollback()
    
    def close(self):
        """Close the connection."""
        if not self._closed:
            self._jdbc_connection.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_simplified_dbapi_connection(
    hostname: str = "localhost",
    port: int = 1972,
    namespace: str = "USER",
    username: str = "SuperUser",
    password: str = "SYS"
) -> Optional[SimplifiedConnection]:
    """
    Get a simplified DBAPI-like connection using JDBC under the hood.
    
    This provides a DBAPI-compatible interface while avoiding the circular
    import issues in the intersystems_iris package.
    
    Args:
        hostname: IRIS server hostname
        port: IRIS server port
        namespace: IRIS namespace
        username: IRIS username
        password: IRIS password
        
    Returns:
        SimplifiedConnection object or None if failed
    """
    try:
        import jaydebeapi
        import os
        
        # JDBC URL
        jdbc_url = f"jdbc:IRIS://{hostname}:{port}/{namespace}"
        
        # JDBC driver path - try multiple locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'intersystems-jdbc-3.9.0.jar'),
            './intersystems-jdbc-3.9.0.jar',
            '../intersystems-jdbc-3.9.0.jar',
            './jdbc_exploration/intersystems-jdbc-3.9.0.jar',
            os.path.expanduser('~/intersystems-jdbc-3.9.0.jar'),
            '/opt/iris/jdbc/intersystems-jdbc-3.9.0.jar'
        ]
        
        jdbc_jar_path = None
        for path in possible_paths:
            if os.path.exists(path):
                jdbc_jar_path = path
                break
        
        if not jdbc_jar_path:
            logger.error("JDBC driver not found for simplified connection")
            return None
        
        # Create JDBC connection
        jdbc_connection = jaydebeapi.connect(
            "com.intersystems.jdbc.IRISDriver",
            jdbc_url,
            [username, password],
            jdbc_jar_path
        )
        
        # Wrap in simplified interface
        simplified_conn = SimplifiedConnection(jdbc_connection)
        
        logger.info("✓ Successfully created simplified DBAPI-like connection")
        return simplified_conn
        
    except Exception as e:
        logger.error(f"Failed to create simplified DBAPI connection: {e}")
        return None


def test_simplified_connection():
    """Test the simplified connection."""
    conn = get_simplified_dbapi_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            logger.info(f"Test query result: {result}")
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Test query failed: {e}")
            return False
        finally:
            conn.close()
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_simplified_connection()
    print(f"Simplified DBAPI connection test: {'✓ PASSED' if success else '✗ FAILED'}")