"""
Fixed IRIS DBAPI connector that works around circular import issues.

This module provides a working DBAPI connection by carefully managing
the import order to avoid the circular dependency in intersystems_iris.
"""
import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_fixed_iris_dbapi_connection(
    hostname: str = "localhost",
    port: int = 1972,
    namespace: str = "USER",
    username: str = "SuperUser",
    password: str = "SYS"
) -> Optional[Any]:
    """
    Get IRIS DBAPI connection with circular import fix.
    
    This function works around the circular import issue by:
    1. Temporarily removing problematic modules from sys.modules
    2. Importing in the correct order
    3. Restoring the module state
    
    Args:
        hostname: IRIS server hostname
        port: IRIS server port
        namespace: IRIS namespace
        username: IRIS username
        password: IRIS password
        
    Returns:
        IRIS DBAPI connection or None if failed
    """
    try:
        # Step 1: Clean up any partially loaded modules
        modules_to_clean = [
            'intersystems_iris',
            'intersystems_iris.dbapi',
            'intersystems_iris.dbapi._DBAPI',
            'intersystems_iris.dbapi._Parameter',
            'intersystems_iris.dbapi._Descriptor',
            'intersystems_iris._BufferWriter',
            'intersystems_iris._ListWriter',
            'iris'
        ]
        
        # Save original modules
        saved_modules = {}
        for module_name in modules_to_clean:
            if module_name in sys.modules:
                saved_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]
        
        # Step 2: Try to import in correct order
        try:
            # Import the base module first
            import intersystems_iris
            
            # Import DBAPI submodules in dependency order
            import intersystems_iris.dbapi._Descriptor
            import intersystems_iris.dbapi._Parameter
            import intersystems_iris.dbapi._DBAPI
            import intersystems_iris.dbapi
            
            # Now try the connection
            conn = intersystems_iris.dbapi.connect(
                hostname=hostname,
                port=port,
                namespace=namespace,
                username=username,
                password=password
            )
            
            logger.info("✓ Successfully connected using fixed IRIS DBAPI")
            return conn
            
        except Exception as import_error:
            logger.warning(f"Fixed import approach failed: {import_error}")
            
            # Restore original modules
            for module_name, module in saved_modules.items():
                sys.modules[module_name] = module
            
            return None
            
    except Exception as e:
        logger.error(f"Failed to establish IRIS DBAPI connection: {e}")
        return None


def test_fixed_connection():
    """Test the fixed connection approach."""
    conn = get_fixed_iris_dbapi_connection()
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
    success = test_fixed_connection()
    print(f"Fixed DBAPI connection test: {'✓ PASSED' if success else '✗ FAILED'}")