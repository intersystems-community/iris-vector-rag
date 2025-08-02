# common/connection_singleton.py
"""
Connection singleton module for managing shared IRIS database connections.
Provides thread-safe singleton pattern for database connections.
"""

import threading
from typing import Optional
from unittest.mock import Mock

# Global connection instance
_shared_connection = None
_connection_lock = threading.Lock()

def get_shared_iris_connection():
    """
    Get the shared IRIS database connection.
    Returns a mock connection for testing purposes.
    """
    global _shared_connection
    
    with _connection_lock:
        if _shared_connection is None:
            # Create mock connection for testing
            _shared_connection = Mock()
            _shared_connection.execute = Mock()
            _shared_connection.fetchall = Mock(return_value=[])
            _shared_connection.fetchone = Mock(return_value=None)
            _shared_connection.commit = Mock()
            _shared_connection.rollback = Mock()
            _shared_connection.close = Mock()
            
        return _shared_connection

def reset_shared_connection():
    """
    Reset the shared connection (useful for testing).
    """
    global _shared_connection
    
    with _connection_lock:
        if _shared_connection:
            try:
                _shared_connection.close()
            except:
                pass  # Ignore errors during cleanup
        _shared_connection = None

def is_connection_active() -> bool:
    """
    Check if there's an active shared connection.
    """
    global _shared_connection
    return _shared_connection is not None