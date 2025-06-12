"""
IRIS Connection Manager - DBAPI First Architecture

This module provides a unified connection manager that prioritizes DBAPI connections
over JDBC, making DBAPI the default for all RAG operations.

Connection Priority:
1. DBAPI (intersystems-irispython package) - DEFAULT
2. JDBC (fallback for specific use cases)
3. Mock (for testing without database)
"""

import os
import logging
from typing import Optional, Any, Dict, Union

logger = logging.getLogger(__name__)

class IRISConnectionManager:
    """
    Unified connection manager for IRIS database with DBAPI-first approach.
    """
    
    def __init__(self, prefer_dbapi: bool = True):
        """
        Initialize connection manager.
        
        Args:
            prefer_dbapi: Whether to prefer DBAPI over JDBC (default: True)
        """
        self.prefer_dbapi = prefer_dbapi
        self._connection = None
        self._connection_type = None
        
    def get_connection(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a database connection, preferring DBAPI by default.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Database connection object
        """
        if self._connection is not None:
            return self._connection
            
        if self.prefer_dbapi:
            # Try DBAPI first
            try:
                self._connection = self._get_dbapi_connection(config)
                self._connection_type = "DBAPI"
                logger.info("✓ Connected using DBAPI (intersystems-irispython)")
                return self._connection
            except Exception as e:
                # Only log as warning if it's a real failure, not just missing module
                if "module 'iris' has no attribute 'connect'" in str(e):
                    logger.debug(f"DBAPI not available (iris module issue): {e}")
                else:
                    logger.warning(f"DBAPI connection failed: {e}")
                logger.info("Falling back to JDBC connection...")
                
        # Fallback to JDBC
        try:
            self._connection = self._get_jdbc_connection(config)
            self._connection_type = "JDBC"
            logger.info("✓ Connected using JDBC")
            return self._connection
        except Exception as e:
            logger.error(f"All connection methods failed. JDBC error: {e}")
            raise ConnectionError("Failed to establish database connection with both DBAPI and JDBC")
    
    def _get_dbapi_connection(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get DBAPI connection using intersystems-irispython package.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            DBAPI connection object
        """
        try:
            import irisnative
            
            # Get connection parameters
            conn_params = self._get_connection_params(config)
            
            # Create an IRIS native connection first
            iris_native_connection = irisnative.createConnection(
                conn_params["hostname"],
                conn_params["port"],
                conn_params["namespace"],
                conn_params["username"],
                conn_params["password"]
            )
            
            # The iris_native_connection object itself should be DBAPI compliant
            connection = iris_native_connection
            
            logger.debug(f"DBAPI connection established to {conn_params['hostname']}:{conn_params['port']}")
            return connection
            
        except ImportError as e:
            raise ImportError(f"intersystems-irispython package not available. Install with: pip install intersystems-irispython. Error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to create DBAPI connection: {e}")
    
    def _get_jdbc_connection(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get JDBC connection as fallback.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            JDBC connection object
        """
        try:
            import jaydebeapi
            
            # Get connection parameters
            conn_params = self._get_connection_params(config)
            
            # JDBC URL
            jdbc_url = f"jdbc:IRIS://{conn_params['hostname']}:{conn_params['port']}/{conn_params['namespace']}"
            
            # JDBC driver path - try multiple locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'intersystems-jdbc-3.8.4.jar'),
                './intersystems-jdbc-3.8.4.jar',
                '../intersystems-jdbc-3.8.4.jar',
                './jdbc_exploration/intersystems-jdbc-3.8.4.jar',
                os.path.expanduser('~/intersystems-jdbc-3.8.4.jar'),
                '/opt/iris/jdbc/intersystems-jdbc-3.8.4.jar'
            ]
            
            jdbc_jar_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    jdbc_jar_path = path
                    break
            
            if not jdbc_jar_path:
                # List all attempted paths for debugging
                attempted_paths = '\n  - '.join(possible_paths)
                raise FileNotFoundError(
                    f"JDBC driver not found. Attempted paths:\n  - {attempted_paths}\n"
                    f"Download from: https://github.com/intersystems-community/iris-driver-distribution/raw/main/JDBC/JDK18/intersystems-jdbc-3.8.4.jar"
                )
            
            # Create JDBC connection
            connection = jaydebeapi.connect(
                "com.intersystems.jdbc.IRISDriver",
                jdbc_url,
                [conn_params["username"], conn_params["password"]],
                jdbc_jar_path
            )
            
            logger.debug(f"JDBC connection established to {jdbc_url} using driver at {jdbc_jar_path}")
            return connection
            
        except ImportError:
            raise ImportError("jaydebeapi package not available. Install with: pip install jaydebeapi")
        except Exception as e:
            raise ConnectionError(f"Failed to create JDBC connection: {e}")
    
    def _get_connection_params(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get connection parameters from config or environment variables.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Dictionary of connection parameters
        """
        # Default parameters
        params = {
            "hostname": os.environ.get("IRIS_HOST", "localhost"),
            "port": int(os.environ.get("IRIS_PORT", "1972")),
            "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
            "username": os.environ.get("IRIS_USERNAME", "SuperUser"),
            "password": os.environ.get("IRIS_PASSWORD", "SYS")
        }
        
        # Override with config if provided
        if config:
            params.update(config)
        
        return params
    
    def get_connection_type(self) -> Optional[str]:
        """
        Get the type of the current connection.
        
        Returns:
            Connection type string ("DBAPI", "JDBC", or None)
        """
        return self._connection_type
    
    def close(self):
        """Close the current connection."""
        if self._connection:
            try:
                self._connection.close()
                logger.debug(f"Closed {self._connection_type} connection")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._connection = None
                self._connection_type = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for backward compatibility
def get_iris_connection(config: Optional[Dict[str, Any]] = None, prefer_dbapi: bool = True) -> Any:
    """
    Get an IRIS database connection, preferring DBAPI by default.
    
    Args:
        config: Optional configuration dictionary
        prefer_dbapi: Whether to prefer DBAPI over JDBC (default: True)
        
    Returns:
        Database connection object
    """
    manager = IRISConnectionManager(prefer_dbapi=prefer_dbapi)
    return manager.get_connection(config)


def get_dbapi_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a DBAPI connection specifically.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        DBAPI connection object
    """
    manager = IRISConnectionManager(prefer_dbapi=True)
    return manager._get_dbapi_connection(config)


def test_connection() -> bool:
    """
    Test database connectivity.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with IRISConnectionManager() as manager:
            conn = manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection manager
    logging.basicConfig(level=logging.INFO)
    
    print("Testing IRIS Connection Manager...")
    
    # Test connection
    if test_connection():
        print("✓ Connection test passed")
    else:
        print("✗ Connection test failed")
    
    # Test connection types
    try:
        with IRISConnectionManager(prefer_dbapi=True) as manager:
            conn = manager.get_connection()
            print(f"✓ Connection established using: {manager.get_connection_type()}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")