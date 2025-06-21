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
    # Try primary import: iris module
    try:
        import iris
        # Check if iris module has dbapi functionality
        if hasattr(iris, 'connect'):
            # The iris module itself provides the DBAPI interface
            logger.info("Successfully imported 'iris' module with DBAPI interface")
            return iris
        else:
            logger.warning("'iris' module imported but doesn't appear to have DBAPI interface (no 'connect' method)")
    except ImportError as e:
        logger.warning(f"Failed to import 'iris' module: {e}")
    
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
    # Get the DBAPI module just-in-time
    irisdbapi = _get_iris_dbapi_module()
    
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
        logger.info("Successfully connected to IRIS using DBAPI interface.")
        return conn
        
    except Exception as e:
        logger.error(f"DBAPI connection failed: {e}")
        return None

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