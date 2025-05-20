import iris # This should work if intersystems_irispython-3.2.0 is installed
import os
import logging

# Configure basic logging to see output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_specific_wheel_test():
    # Connection details (should match what your Docker container setup uses)
    db_host = os.environ.get("IRIS_HOST", "localhost")
    # Ensure the IRIS Docker container (e.g., 'iris_benchmark_container' from tests) is running
    # and its port 1972 is mapped to host port 51773 (or your test config).
    db_port = int(os.environ.get("IRIS_PORT", 51773)) 
    db_namespace = os.environ.get("IRIS_NAMESPACE", "USER")
    # The user/pass "test"/"test" is created by the conftest.py setup.
    # If running this standalone, ensure the user exists or use SuperUser/SYS.
    db_user = os.environ.get("IRIS_USERNAME", "test") 
    db_password = os.environ.get("IRIS_PASSWORD", "test")

    connection_url_info = f"host='{db_host}', port={db_port}, namespace='{db_namespace}', user='{db_user}'"
    logger.info(f"Attempting to connect to IRIS using: {connection_url_info} via specific wheel driver")

    connection = None
    cursor = None

    try:
        connection = iris.connect(
            hostname=db_host,
            port=db_port,
            namespace=db_namespace,
            username=db_user,
            password=db_password
        )
        logger.info(f"DBAPI connection obtained: {type(connection)}")
        
        cursor = connection.cursor()
        logger.info("DBAPI cursor obtained.")

        # Define a sample fully inlined SQL query (problematic one)
        top_k = 1
        dummy_vector_768 = [0.1] * 768 
        vector_str = f"[{','.join(map(str, dummy_vector_768))}]"

        test_sql = f"""
            SELECT TOP {top_k} doc_id 
            FROM SourceDocuments 
            WHERE embedding IS NOT NULL 
            ORDER BY VECTOR_COSINE(embedding, TO_VECTOR('{vector_str}', 'DOUBLE', 768)) DESC
        """
        
        logger.info(f"Executing SQL (fully inlined):\n{test_sql[:300]}... (vector part truncated)")

        cursor.execute(test_sql) # No parameters passed
        logger.info("SQL query executed successfully.")
        
        results = cursor.fetchall()
        if results:
            logger.info(f"Query returned {len(results)} row(s). First row: {results[0]}")
        else:
            logger.info("Query returned no results (this is okay if the table is empty or no matches).")

    except iris.dbapi.Error as e: # Catching the specific DBAPI error from this driver
        logger.error(f"IRIS DBAPIError occurred: {e}")
        # The 'iris.dbapi.Error' object often has 'sqlcode' and 'message' attributes directly
        if hasattr(e, 'sqlcode') and hasattr(e, 'message'):
             logger.error(f"  DBAPI Error Details: SQLCODE: {e.sqlcode}, Message: {e.message}")
        elif e.args:
             logger.error(f"  DBAPI Error args: {e.args}")
             
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        
    finally:
        if cursor:
            cursor.close()
            logger.info("DBAPI cursor closed.")
        if connection:
            connection.close()
            logger.info("DBAPI connection closed.")

if __name__ == "__main__":
    # Important: This script must be run in a clean virtual environment
    # where ONLY the specific intersystems_irispython-3.2.0 wheel is installed.
    # Ensure the IRIS Docker container is running and accessible.
    # The SourceDocuments table should exist.
    run_specific_wheel_test()
