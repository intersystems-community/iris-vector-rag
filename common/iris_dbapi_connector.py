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
    
    # Fallback attempts for older or alternative package structures
    try:
        import intersystems_iris.dbapi._DBAPI as irisdbapi_alt
        logger.info("Successfully imported 'intersystems_iris.dbapi._DBAPI' (working fallback)")
        return irisdbapi_alt
    except ImportError as e2:
        logger.warning(f"Failed to import 'intersystems_iris.dbapi._DBAPI': {e2}")
    
    try:
        import intersystems_iris.dbapi as irisdbapi_alt2
        logger.info("Successfully imported 'intersystems_iris.dbapi' (fallback)")
        return irisdbapi_alt2
    except ImportError as e3:
        logger.warning(f"Failed to import 'intersystems_iris.dbapi': {e3}")
    
    try:
        import irisnative.dbapi as irisdbapi_native
        logger.info("Successfully imported 'irisnative.dbapi' (fallback)")
        return irisdbapi_native
    except ImportError as e4:
        logger.warning(f"Failed to import 'irisnative.dbapi': {e4}")
    
    # All import attempts failed
    logger.error(
        "InterSystems IRIS DBAPI module could not be imported. "
        "Checked 'iris' (primary), 'intersystems_iris.dbapi._DBAPI', "
        "'intersystems_iris.dbapi', and 'irisnative.dbapi'. "
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
        
        # Try different connection approaches to handle SSL issues
        try:
            # First try: Use the working connection format with SSL disabled
            conn = irisdbapi.connect(host, port, namespace, user, password, ssl=False)
            logger.info("Successfully connected to IRIS using DBAPI interface (SSL disabled).")
            return conn
        except Exception as ssl_error:
            logger.warning(f"SSL disabled connection failed: {ssl_error}")
            
            # Second try: Use the original working format
            conn = irisdbapi.connect(host, port, namespace, user, password)
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