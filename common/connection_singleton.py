#!/usr/bin/env python3
"""
Connection Singleton Manager

Ensures all components use the same database connection to prevent
transaction isolation issues during testing and operation.
"""

import threading
from typing import Optional, Any, Dict
from common.iris_connection_manager import IRISConnectionManager

class ConnectionSingleton:
    """Singleton connection manager to ensure consistent database connections."""
    
    _instance = None
    _lock = threading.Lock()
    _connection_manager = None
    _connection = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConnectionSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_connection(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get the singleton database connection."""
        if self._connection is None:
            with self._lock:
                if self._connection is None:
                    if self._connection_manager is None:
                        self._connection_manager = IRISConnectionManager(config)
                    self._connection = self._connection_manager.get_connection()
                    # Enable autocommit to ensure immediate visibility of changes
                    try:
                        self._connection.autocommit = True
                    except AttributeError:
                        # Some drivers might not support autocommit attribute
                        pass
        return self._connection
    
    def reset_connection(self):
        """Reset the connection (useful for testing)."""
        with self._lock:
            if self._connection:
                try:
                    self._connection.close()
                except:
                    pass
            self._connection = None
            self._connection_manager = None

# Global singleton instance
_connection_singleton = ConnectionSingleton()

def get_shared_iris_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a shared IRIS database connection.
    
    This ensures all components use the same connection to prevent
    transaction isolation issues.
    
    Args:
        config: Optional configuration dictionary for connection parameters.
        
    Returns:
        A shared database connection object.
    """
    return _connection_singleton.get_connection(config)

def reset_shared_connection():
    """Reset the shared connection (useful for testing)."""
    _connection_singleton.reset_connection()