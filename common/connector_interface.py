"""
Unified interface for IRIS database connectors.
Supports both DBAPI and JDBC implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

class IRISConnectorInterface(ABC):
    """Abstract interface for IRIS database connectors."""
    
    @abstractmethod
    def cursor(self):
        """Return a database cursor."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[List] = None):
        """Execute a query with optional parameters."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass
    
    @abstractmethod
    def commit(self):
        """Commit the current transaction."""
        pass

class DBAPIConnectorWrapper(IRISConnectorInterface):
    """Wrapper for DBAPI connections to implement standard interface."""
    
    def __init__(self, connection):
        self.connection = connection
    
    def cursor(self):
        return self.connection.cursor()
    
    def execute_query(self, query: str, params: Optional[List] = None):
        cursor = self.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor
    
    def close(self):
        self.connection.close()
    
    def commit(self):
        self.connection.commit()

class JDBCConnectorWrapper(IRISConnectorInterface):
    """Wrapper for JDBC connections to implement standard interface."""
    
    def __init__(self, connection):
        self.connection = connection
    
    def cursor(self):
        return self.connection.cursor()
    
    def execute_query(self, query: str, params: Optional[List] = None):
        cursor = self.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor
    
    def close(self):
        self.connection.close()
    
    def commit(self):
        self.connection.commit()