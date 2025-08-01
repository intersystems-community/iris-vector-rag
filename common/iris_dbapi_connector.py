"""
Connector for InterSystems IRIS using Python DBAPI
"""
import os
import logging

logger = logging.getLogger(__name__)

def _get_iris_dbapi_module():
    """
    Attempts to import and return the appropriate IRIS DBAPI module.
    
    Based on PyPI documentation for intersystems-irispython package:
    - The main import is 'import iris'
    - DBAPI functionality is accessed through the iris module
    - The package provides both native connections and DBAPI interface
    
    Returns:
        The IRIS DBAPI module if successfully imported, None otherwise.
    """
    try:
        import iris as iris_dbapi
        # Check if iris_dbapi module has _DBAPI submodule with connect method
        if hasattr(iris_dbapi, '_DBAPI') and hasattr(iris_dbapi._DBAPI, 'connect'):
            # The _DBAPI submodule provides the DBAPI interface
            logger.info("Successfully imported 'iris' module with DBAPI interface")
            return iris_dbapi._DBAPI
        elif hasattr(iris_dbapi, 'connect'):
            # The iris_dbapi module itself provides the DBAPI interface
            logger.info("Successfully imported 'iris' module with DBAPI interface")
            return iris_dbapi
        else:
            logger.warning("'iris' module imported but doesn't appear to have DBAPI interface (no 'connect' method)")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import 'iris' module (circular import issue): {e}")
        
        # Fallback to direct iris import for older installations
        try:
            import iris
            if hasattr(iris, 'connect'):
                logger.info("Successfully imported 'iris' module with DBAPI interface (fallback)")
                return iris
            else:
                logger.warning("'iris' module imported but doesn't appear to have DBAPI interface (no 'connect' method)")
        except ImportError as e2:
            logger.warning(f"Failed to import 'iris' module as fallback: {e2}")
    
    # All import attempts failed
    logger.error(
        "InterSystems IRIS DBAPI module could not be imported. "
        "The 'iris' module was found but doesn't have the expected 'connect' method. "
        "Please ensure the 'intersystems-irispython' package is installed correctly. "
        "DBAPI connections will not be available."
    )
    return None

def get_iris_dbapi_connection():
    """
    Establishes a connection to InterSystems IRIS using DBAPI.

    Reads connection parameters from environment variables:
    - IRIS_HOST
    - IRIS_PORT
    - IRIS_NAMESPACE
    - IRIS_USER
    - IRIS_PASSWORD
    - IRIS_CONNECTION_STRING (if provided, overrides individual parameters)

    Returns:
        A DBAPI connection object or None if connection fails.
    """
    # Get the DBAPI module using lazy loading to avoid circular imports
    irisdbapi = get_iris_dbapi_module()
    
    if not irisdbapi:
        logger.error("Cannot create DBAPI connection: InterSystems IRIS DBAPI module is not available.")
        return None

    # For the iris module, we need to create a native connection first, then use it for DBAPI
    host = os.environ.get("IRIS_HOST", "localhost")
    port = int(os.environ.get("IRIS_PORT", 1972))
    namespace = os.environ.get("IRIS_NAMESPACE", "USER")
    user = os.environ.get("IRIS_USER", "_SYSTEM")
    password = os.environ.get("IRIS_PASSWORD", "SYS")

    try:
        logger.info(f"Attempting IRIS connection to {host}:{port}/{namespace} as user {user}")
        
        # Use the correct connection parameters format
        conn = irisdbapi.connect(
            hostname=host,
            port=port,
            namespace=namespace,
            username=user,
            password=password
        )
        
        # Validate the connection handle
        if conn is None:
            logger.error("DBAPI connection failed: _handle is NULL")
            return None
        
        # Test the connection with a simple query
        try:
            cursor = conn.cursor()
            if cursor is None:
                logger.error("DBAPI connection failed: cursor is NULL")
                conn.close()
                return None
            
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            if result is None:
                logger.error("DBAPI connection failed: test query returned NULL")
                conn.close()
                return None
                
        except Exception as test_e:
            logger.error(f"DBAPI connection validation failed: {test_e}")
            try:
                conn.close()
            except:
                pass
            return None
        
        logger.info("Successfully connected to IRIS using DBAPI interface.")
        return conn
        
    except Exception as e:
        logger.error(f"DBAPI connection failed: {e}")
        return None

# Lazy-loaded DBAPI module - initialized only when needed
_cached_irisdbapi = None

def get_iris_dbapi_module():
    """
    Get the IRIS DBAPI module with lazy loading to avoid circular imports.
    
    This function caches the module after first successful import to avoid
    repeated import attempts.
    
    Returns:
        The IRIS DBAPI module if available, None otherwise.
    """
    global _cached_irisdbapi
    
    if _cached_irisdbapi is None:
        _cached_irisdbapi = _get_iris_dbapi_module()
    
    return _cached_irisdbapi

# For backward compatibility, provide irisdbapi as a property-like access
@property
def irisdbapi():
    """Backward compatibility property for accessing the IRIS DBAPI module."""
    return get_iris_dbapi_module()

# Make irisdbapi available as module attribute through __getattr__
def __getattr__(name):
    """Module-level attribute access for backward compatibility."""
    if name == 'irisdbapi':
        return get_iris_dbapi_module()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if __name__ == '__main__':
    # Basic test for the connection
    # Ensure environment variables are set (e.g., in a .env file or system-wide)
    # Example:
    # export IRIS_HOST="your_iris_host"
    # export IRIS_PORT="1972"
    # export IRIS_NAMESPACE="USER"
    # export IRIS_USER="your_user"
    # export IRIS_PASSWORD="your_password"
    logging.basicConfig(level=logging.INFO)
    connection = get_iris_dbapi_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT %Version FROM %SYSTEM.Version")
            version = cursor.fetchone()
            logger.info(f"IRIS Version (DBAPI): {version[0]}")
            cursor.close()
        except Exception as e:
            logger.error(f"Error during DBAPI test query: {e}")
        finally:
            connection.close()
            logger.info("DBAPI connection closed.")
    else:
        logger.warning("DBAPI connection could not be established for testing.")