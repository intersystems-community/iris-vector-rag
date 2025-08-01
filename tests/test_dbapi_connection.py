import os
import logging
import sys

# Add the project root to the Python path to allow importing from common
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from common.connection_manager import get_connection_manager, set_global_connection_type
    from common.iris_dbapi_connector import _get_iris_dbapi_module # Import the function
except ImportError as e:
    logging.error(f"Failed to import necessary modules. Ensure you are in the project root or have set PYTHONPATH. Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dbapi_connection():
    """
    Tests the DBAPI connection through the ConnectionManager.
    """
    logger.info("Starting DBAPI connection test...")

    # Get the DBAPI module lazily within the test function to avoid circular imports
    irisdbapi = _get_iris_dbapi_module()

    if not irisdbapi: # This check remains the same
        logger.error(
            "InterSystems IRIS DBAPI module (expected 'iris' module or fallbacks) "
            "not found. Test cannot proceed."
        )
        logger.info(
            "Please ensure 'intersystems-irispython' is installed in your virtual environment "
            "and that the 'iris' module can be imported."
        )
        return
    
    logger.info(f"DBAPI module available: {irisdbapi}")

    # Explicitly set to use DBAPI
    logger.info("Setting global connection type to 'dbapi'")
    set_global_connection_type("dbapi")

    manager = None
    try:
        logger.info("Attempting to get connection manager...")
        # The get_connection_manager will now use the globally set "dbapi" type
        manager = get_connection_manager()
        
        logger.info(f"Connection type selected: {manager.connection_type.upper()}")

        # Test basic connection and query
        logger.info("Attempting to execute a simple query: SELECT 1 as test_value")
        test_result = manager.execute("SELECT 1 as test_value")

        if test_result:
            logger.info(f"Successfully executed query. Result: {test_result[0][0]}")
        else:
            logger.warning("Query executed but returned no results or an empty list.")

        # Test with context manager
        logger.info("Testing connection manager as a context manager...")
        with get_connection_manager() as cm_context: # Should pick up global "dbapi"
             logger.info(f"Context manager connection type: {cm_context.connection_type.upper()}")
             res = cm_context.execute("SELECT 2 as test_value")
             logger.info(f"Context manager query 'SELECT 2' result: {res}")

        # Test cursor usage
        logger.info("Testing cursor usage...")
        with manager.cursor() as cursor:
            cursor.execute("SELECT 3 as test_value")
            result = cursor.fetchone()
            if result:
                logger.info(f"Successfully fetched with cursor. Result: {result[0]}")
            else:
                logger.warning("Cursor query returned no result.")
        
        logger.info("DBAPI connection test completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the DBAPI connection test: {e}", exc_info=True)
        if manager and manager.connection_type != "dbapi":
            logger.info(f"Note: ConnectionManager may have fallen back to {manager.connection_type.upper()} due to DBAPI failure.")
    finally:
        if manager:
            logger.info("Closing connection manager...")
            manager.close()
        logger.info("DBAPI connection test finished.")

if __name__ == "__main__":
    logger.info("-----------------------------------------------------")
    logger.info(" MAKE SURE IRIS ENVIRONMENT VARIABLES ARE SET:       ")
    logger.info(" - IRIS_HOST                                         ")
    logger.info(" - IRIS_PORT                                         ")
    logger.info(" - IRIS_NAMESPACE                                    ")
    logger.info(" - IRIS_USER                                         ")
    logger.info(" - IRIS_PASSWORD                                     ")
    logger.info(" OR IRIS_CONNECTION_STRING                           ")
    logger.info(" AND intersystems_iris.dbapi is installed.           ")
    logger.info("-----------------------------------------------------")
    test_dbapi_connection()