import logging
import os
import sys

# Add common to sys.path to find iris_connector
project_root = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(project_root, 'common')
if common_path not in sys.path:
    sys.path.insert(0, common_path)

try:
    from iris_connector import get_iris_connection, IRISConnectionError
except ImportError as e:
    print(f"CRITICAL: Error importing from common.iris_connector. Ensure 'common' is in PYTHONPATH: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger("test_dbapi_connection")

def test_dbapi_simple_query():
    logger.info("--- Starting test_dbapi_simple_query ---")
    
    # Connection details will be taken from environment variables or defaults
    # by get_iris_connection (e.g., IRIS_HOST=localhost, IRIS_PORT=1972 for in-container)
    db_connection = None
    cursor = None

    try:
        logger.info("Attempting to get DB-API connection via common.iris_connector...")
        # Default connection_config in get_iris_connection uses env vars or defaults
        # which should be suitable for running inside the single container.
        db_connection = get_iris_connection() 
        
        if not db_connection:
            raise IRISConnectionError("Failed to get a DB-API connection from iris_connector.")

        logger.info(f"DB-API Connection successful. Type: {type(db_connection)}")
        
        cursor = db_connection.cursor()
        logger.info(f"DB-API cursor obtained. Type: {type(cursor)}")

        test_query = "SELECT 1 AS TestValue, $HOROLOG AS CurrentTimestamp"
        logger.info(f"Executing test query: {test_query}")
        
        cursor.execute(test_query)
        logger.info("Test query executed successfully.")
        
        results = cursor.fetchall()
        logger.info(f"Raw results from query: {results}")
        
        if results and len(results) == 1:
            row = results[0]
            logger.info(f"Fetched row: {row}")
            
            # pyodbc returns Row objects where columns can be accessed by name or index.
            # intersystems_iris DB-API might behave similarly or return tuples.
            # Let's try accessing by index first, then by name if available.
            test_value = None
            current_timestamp = None
            
            if len(row) == 2: # Expecting two columns
                test_value = row[0]
                current_timestamp = row[1]
                logger.info(f"Accessed by index: TestValue='{test_value}', CurrentTimestamp='{current_timestamp}'")
            elif hasattr(row, "TestValue") and hasattr(row, "CurrentTimestamp"):
                 test_value = row.TestValue
                 current_timestamp = row.CurrentTimestamp
                 logger.info(f"Accessed by attribute: TestValue='{test_value}', CurrentTimestamp='{current_timestamp}'")
            else:
                logger.error("❌ FAILURE: Could not access results by index or attribute name.")
                return


            if test_value == 1:
                logger.info(f"✅ SUCCESS: TestValue is 1 as expected.")
            else:
                logger.error(f"❌ FAILURE: TestValue is '{test_value}', expected 1.")

            if current_timestamp and isinstance(current_timestamp, str) and len(current_timestamp) > 0:
                logger.info(f"✅ SUCCESS: CurrentTimestamp is a non-empty string: '{current_timestamp}'.")
            else:
                logger.error(f"❌ FAILURE: CurrentTimestamp is invalid: '{current_timestamp}'.")
        else:
            logger.error(f"❌ FAILURE: Did not retrieve exactly one row. Results: {results}")

    except IRISConnectionError as e:
        logger.error(f"DB-API Connection Error during test: {e}", exc_info=True)
    except Exception as e:
        # For intersystems_iris, dbapi exceptions are often subclasses of dbapi.Error
        # Check if it's a DBAPI error to print SQLCODE etc.
        if hasattr(e, 'sqlcode') and hasattr(e, 'message'):
             logger.error(f"DB-API Error occurred: SQLCODE: {e.sqlcode}, Message: {e.message}", exc_info=True)
        else:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        
    finally:
        if cursor: 
            cursor.close()
            logger.info("DB-API cursor closed.")
        if db_connection: 
            db_connection.close()
            logger.info("DB-API connection closed.")

if __name__ == "__main__":
    logger.info("Starting DB-API connection test...")
    test_dbapi_simple_query()
    logger.info("Finished DB-API connection test.")
