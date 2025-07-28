"""
Direct IRIS DBAPI connection that bypasses circular import issues.

This module attempts to directly import and use the IRIS DBAPI components
without triggering the circular import in the intersystems_iris package.
"""
import logging
import importlib.util
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_direct_iris_dbapi_connection(
    hostname: str = "localhost",
    port: int = 1972,
    namespace: str = "USER",
    username: str = "SuperUser",
    password: str = "SYS"
) -> Optional[Any]:
    """
    Get IRIS DBAPI connection by directly importing the working components.
    
    This approach tries to bypass the circular import by importing only
    the specific modules we need in the correct order.
    """
    try:
        # Approach 1: Try to import just the connect function directly
        try:
            # Import the specific module that contains the connect function
            spec = importlib.util.find_spec("intersystems_iris.dbapi._DBAPI")
            if spec and spec.loader:
                dbapi_module = importlib.util.module_from_spec(spec)
                
                # Try to load it without triggering the circular import
                # by setting up the module structure first
                sys.modules["intersystems_iris.dbapi._DBAPI"] = dbapi_module
                spec.loader.exec_module(dbapi_module)
                
                if hasattr(dbapi_module, 'connect'):
                    conn = dbapi_module.connect(
                        hostname=hostname,
                        port=port,
                        namespace=namespace,
                        username=username,
                        password=password
                    )
                    logger.info("✓ Successfully connected using direct DBAPI import")
                    return conn
                    
        except Exception as e:
            logger.debug(f"Direct import approach failed: {e}")
        
        # Approach 2: Try to manually patch the circular import
        try:
            # Create a minimal mock for the problematic reference
            import types
            
            # Create mock modules to break the circular dependency
            mock_descriptor = types.ModuleType("intersystems_iris.dbapi._Descriptor")
            mock_descriptor._Descriptor = type("_Descriptor", (), {})
            sys.modules["intersystems_iris.dbapi._Descriptor"] = mock_descriptor
            
            # Now try to import the parameter module
            import intersystems_iris.dbapi._Parameter
            
            # Then import the main DBAPI
            import intersystems_iris.dbapi
            
            conn = intersystems_iris.dbapi.connect(
                hostname=hostname,
                port=port,
                namespace=namespace,
                username=username,
                password=password
            )
            logger.info("✓ Successfully connected using patched DBAPI import")
            return conn
            
        except Exception as e:
            logger.debug(f"Patched import approach failed: {e}")
        
        # Approach 3: Try to use the embedded IRIS connection
        try:
            import intersystems_iris._IRISEmbedded as iris_embedded
            
            # Try to create a connection using the embedded interface
            if hasattr(iris_embedded, 'connect'):
                conn = iris_embedded.connect(
                    hostname=hostname,
                    port=port,
                    namespace=namespace,
                    username=username,
                    password=password
                )
                logger.info("✓ Successfully connected using embedded IRIS interface")
                return conn
                
        except Exception as e:
            logger.debug(f"Embedded interface approach failed: {e}")
            
        logger.error("All direct DBAPI connection approaches failed")
        return None
        
    except Exception as e:
        logger.error(f"Failed to establish direct IRIS DBAPI connection: {e}")
        return None


def test_direct_connection():
    """Test the direct connection approach."""
    conn = get_direct_iris_dbapi_connection()
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
    success = test_direct_connection()
    print(f"Direct DBAPI connection test: {'✓ PASSED' if success else '✗ FAILED'}")