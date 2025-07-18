"""
Factory for creating standardized database connections.
Implements user preference for DBAPI as default.
"""

import os
import logging
from typing import Dict, Any, Optional
from .connector_interface import IRISConnectorInterface, DBAPIConnectorWrapper, JDBCConnectorWrapper

logger = logging.getLogger(__name__)

class ConnectionFactory:
    """Factory for creating IRIS database connections."""
    
    @staticmethod
    def create_connection(connection_type: str = "dbapi", **config) -> IRISConnectorInterface:
        """
        Create an IRIS database connection.
        
        Args:
            connection_type: "dbapi" (default) or "jdbc"
            **config: Connection configuration parameters
            
        Returns:
            IRISConnectorInterface implementation
        """
        if connection_type == "dbapi":
            return ConnectionFactory._create_dbapi_connection(**config)
        elif connection_type == "jdbc":
            return ConnectionFactory._create_jdbc_connection(**config)
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")
    
    @staticmethod
    def _create_dbapi_connection(**config) -> IRISConnectorInterface:
        """Create DBAPI connection (user preference)."""
        from .iris_dbapi_connector import get_iris_dbapi_connection
        
        # Set environment variables if provided in config
        if config:
            for key, value in config.items():
                env_key = f"IRIS_{key.upper()}"
                os.environ[env_key] = str(value)
        
        connection = get_iris_dbapi_connection()
        return DBAPIConnectorWrapper(connection)
    
    @staticmethod
    def _create_jdbc_connection(**config) -> IRISConnectorInterface:
        """Create JDBC connection (enterprise/legacy)."""
        from .iris_connector import get_real_iris_connection
        
        connection = get_real_iris_connection(config)
        return JDBCConnectorWrapper(connection)
    
    @staticmethod
    def from_environment() -> IRISConnectorInterface:
        """Create connection using environment variables with DBAPI default."""
        return ConnectionFactory.create_connection("dbapi")