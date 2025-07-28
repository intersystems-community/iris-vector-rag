"""
IRIS Connection Manager - DBAPI First Architecture with JDBC Fallback

This module provides a unified connection manager that prioritizes DBAPI connections
and falls back to JDBC if the DBAPI connection fails. It includes smart
environment detection to ensure IRIS packages are available.

Connection Priority:
1. DBAPI (intersystems-irispython package)
2. JDBC (jaydebeapi package) - as a fallback
3. Mock (for testing without database)

Environment Priority:
1. UV environment (.venv) if available and has IRIS packages
2. Current environment if it has IRIS packages
3. System Python as fallback
"""

import os
import logging
from typing import Optional, Any, Dict
from dataclasses import dataclass

from common.db_driver_utils import DriverType, get_driver_type as _get_driver_type, get_driver_capabilities as _get_driver_capabilities

logger = logging.getLogger(__name__)

@dataclass
class ConnectionInfo:
    """Metadata about an IRIS database connection."""
    connection: Any
    driver_type: DriverType
    hostname: str
    port: int
    namespace: str
    username: str
    capabilities: Dict[str, bool]

# Smart environment detection
def _detect_best_iris_environment():
    """Detect the best environment for IRIS connections."""
    try:
        from .environment_manager import EnvironmentManager
        env_manager = EnvironmentManager()
        return env_manager.ensure_iris_available()
    except ImportError:
        # Fallback if environment manager not available
        return True

class IRISConnectionManager:
    """
    Unified connection manager for IRIS database with DBAPI-first approach.
    
    Provides automatic driver detection, fallback mechanisms, and connection
    metadata for optimizing SQL generation based on driver capabilities.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize connection manager.
        """
        from iris_rag.config.manager import ConfigurationManager
        self.config_manager = config_manager or ConfigurationManager()
        self._connection = None
        self._connection_type = None
        self._connection_info = None
        # Driver capabilities are now handled by the db_driver_utils module
        
    def get_connection(self) -> Any:
        """
        Get a database connection, preferring DBAPI and falling back to JDBC.
        
        This method maintains backward compatibility by returning just the connection object.
        Use get_connection_info() for enhanced metadata.
            
        Returns:
            Database connection object
        """
        if self._connection is not None:
            return self._connection
        
        # Get full connection info and extract just the connection
        connection_info = self.get_connection_info()
        return connection_info.connection
    
    def get_connection_info(self) -> ConnectionInfo:
        """
        Get detailed connection information including driver type and capabilities.
        
        Returns:
            ConnectionInfo object with connection metadata
        """
        if self._connection_info is not None:
            return self._connection_info
            
        config = {}
        if hasattr(self.config_manager, 'get_database_config'):
            config = self.config_manager.get_database_config()
        elif isinstance(self.config_manager, dict):
            config = self.config_manager
        
        # Ensure config is a dictionary, even if get_database_config returns None
        if config is None:
            config = {}
            
        # Get connection parameters
        conn_params = self._get_connection_params(config)
        
        # Try DBAPI first
        try:
            connection = self._get_dbapi_connection(config)
            self._connection = connection
            self._connection_type = "DBAPI"
            
            self._connection_info = ConnectionInfo(
                connection=connection,
                driver_type=DriverType.DBAPI,
                hostname=conn_params['hostname'],
                port=conn_params['port'],
                namespace=conn_params['namespace'],
                username=conn_params['username'],
                capabilities=_get_driver_capabilities(DriverType.DBAPI)
            )
            
            logger.info("✓ Connected using DBAPI (intersystems-irispython)")
            return self._connection_info
            
        except Exception as dbapi_error:
            logger.warning(f"DBAPI connection failed: {dbapi_error}. Attempting JDBC fallback.")
            
            # Fallback to JDBC
            try:
                connection = self._get_jdbc_connection(config)
                self._connection = connection
                self._connection_type = "JDBC"
                
                self._connection_info = ConnectionInfo(
                    connection=connection,
                    driver_type=DriverType.JDBC,
                    hostname=conn_params['hostname'],
                    port=conn_params['port'],
                    namespace=conn_params['namespace'],
                    username=conn_params['username'],
                    capabilities=_get_driver_capabilities(DriverType.JDBC)
                )
                
                logger.info("✓ Connected using JDBC (jaydebeapi)")
                logger.warning("⚠️  JDBC driver has limitations with vector operations")
                return self._connection_info
                
            except Exception as jdbc_error:
                logger.error(f"JDBC connection failed after DBAPI failure: {jdbc_error}")
                raise ConnectionError(f"Failed to establish database connection with both DBAPI and JDBC. DBAPI error: {dbapi_error}, JDBC error: {jdbc_error}")
    
    def get_driver_type(self) -> DriverType:
        """
        Get the type of driver currently in use.
        
        Returns:
            DriverType enum value
        """
        return self.get_connection_info().driver_type
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get the capabilities of the current driver.
        
        Returns:
            Dictionary of capability flags
        """
        return self.get_connection_info().capabilities
    
    def supports_vector_operations(self) -> bool:
        """
        Check if the current driver supports vector operations.
        
        Returns:
            True if vector operations are supported
        """
        return self.get_capabilities().get("vector_operations", False)
    
    def _get_dbapi_connection(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get DBAPI connection using proper iris import as per official documentation.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            DBAPI connection object
        """
        try:
            # Use the proper import as per InterSystems documentation
            import iris
            
            # Get connection parameters
            conn_params = self._get_connection_params(config)
            
            # Create an IRIS connection using the official approach.
            # IRIS DBAPI requires either a connectionstr or individual parameters
            
            # Try connection string format first (recommended approach)
            try:
                connectionstr = f"{conn_params['hostname']}:{conn_params['port']}/{conn_params['namespace']}"
                connection = iris.connect(
                    connectionstr=connectionstr,
                    username=conn_params['username'],
                    password=conn_params['password']
                )
            except Exception as conn_str_error:
                # Fallback to individual parameters if connection string fails
                logger.debug(f"Connection string approach failed: {conn_str_error}, trying individual parameters")
                connection = iris.connect(
                    hostname=conn_params['hostname'],
                    port=conn_params['port'],
                    namespace=conn_params['namespace'],
                    username=conn_params['username'],
                    password=conn_params['password']
                )
            
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
            jdbc_url = f"jdbc:IRIS://{conn_params['hostname']}:{conn_params['port']}/{conn_params['namespace']}?SSL=false"
            
            # JDBC driver path - try multiple locations
            possible_paths = [
                './intersystems-jdbc-3.10.3.jar',
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
                    f"Download from: https://github.com/intersystems-community/iris-driver-distribution/raw/main/JDBC/JDK18/intersystems-jdbc-3.9.0.jar"
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
        
        # Override with config if provided, but only if the value is not None
        if config:
            for key, value in config.items():
                if value is not None:
                    params[key] = value
        
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
def get_iris_dbapi_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get DBAPI connection specifically.
    """
    manager = IRISConnectionManager()
    return manager._get_dbapi_connection(config)

def get_iris_jdbc_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get JDBC connection specifically.
    """
    manager = IRISConnectionManager()
    return manager._get_jdbc_connection(config)

def get_iris_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get an IRIS database connection.
    
    This function attempts to connect using the DBAPI first, and falls back
    to JDBC if the DBAPI connection fails. It will raise a ConnectionError
    if both methods fail.
    
    Args:
        config: Optional configuration dictionary for connection parameters.
        
    Returns:
        A database connection object.
    """
    manager = IRISConnectionManager(config)
    return manager.get_connection()


def get_dbapi_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a DBAPI connection specifically.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        DBAPI connection object
    """
    manager = IRISConnectionManager()
    return manager._get_dbapi_connection(config)


# Global connection manager instance for enhanced functionality
_global_connection_manager = IRISConnectionManager()

def get_connection_info() -> ConnectionInfo:
    """
    Get detailed connection information including driver type and capabilities.
    
    Returns:
        ConnectionInfo object with connection metadata
    """
    return _global_connection_manager.get_connection_info()

def get_driver_type() -> DriverType:
    """
    Get the type of driver currently in use.
    
    Returns:
        DriverType enum value
    """
    return _global_connection_manager.get_driver_type()

def supports_vector_operations() -> bool:
    """
    Check if the current driver supports vector operations.
    
    Returns:
        True if vector operations are supported
    """
    return _global_connection_manager.supports_vector_operations()

def get_driver_capabilities() -> Dict[str, bool]:
    """
    Get the capabilities of the current driver.
    
    Returns:
        Dictionary of capability flags
    """
    return _global_connection_manager.get_capabilities()


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
        with IRISConnectionManager() as manager:
            conn = manager.get_connection()
            print(f"✓ Connection established using: {manager.get_connection_type()}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")