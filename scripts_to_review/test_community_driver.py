import sqlalchemy
import os
import logging

# Configure basic logging to see output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    # Connection details (should match what your Docker container setup uses)
    # Taken from common/iris_connector.py and conftest.py
    # Ensure the IRIS Docker container (e.g., iris_benchmark_container) is running
    # and port 51773 is mapped to its 1972.
    # The user/pass "test"/"test" is created by the conftest.py setup.
    # If running this standalone, ensure the user exists or use SuperUser/SYS.
    
    # Default to environment variable if set, otherwise use common test values
    # For this standalone test, we'll hardcode the typical testcontainer URL
    # but ideally, this would come from a shared config or env var.
    # The conftest_real_pmc.py uses IRIS_CONNECTION_URL which is set to
    # f"iris://test:test@localhost:{host_port}/{namespace}"
    # where host_port is often 51773 and namespace is USER.
    
    # Let's use the same credentials and target as the main test suite.
    # Ensure your IRIS container (e.g., 'iris_benchmark_container' from tests) is running.
    db_host = os.environ.get("IRIS_HOST", "localhost")
    db_port = int(os.environ.get("IRIS_PORT", 51773)) # Default test port from conftest
    db_namespace = os.environ.get("IRIS_NAMESPACE", "USER")
    db_user = os.environ.get("IRIS_USERNAME", "test") # User created by test setup
    db_password = os.environ.get("IRIS_PASSWORD", "test") # Password for 'test' user

    connection_url = f"iris://{db_user}:{db_password}@{db_host}:{db_port}/{db_namespace}"
    
    logger.info(f"Attempting to connect to IRIS using URL: {connection_url} via sqlalchemy-iris")

    engine = None
    sqla_connection = None
    raw_dbapi_connection = None
    cursor = None

    try:
        engine = sqlalchemy.create_engine(connection_url)
        logger.info("SQLAlchemy engine created.")
        
        sqla_connection = engine.connect()
        logger.info("SQLAlchemy connection established.")
        
        raw_dbapi_connection = sqla_connection.connection
        logger.info(f"Raw DBAPI connection obtained: {type(raw_dbapi_connection)}")
        
        cursor = raw_dbapi_connection.cursor()
        logger.info("DBAPI cursor obtained.")

        # Define a sample fully inlined SQL query (problematic one)
        # Using a simple vector for testing
        top_k = 1
        # Example vector string (ensure it's a valid format for TO_VECTOR)
        # For a 768-dim vector, this would be very long. Using a 3-dim example for brevity.
        # Adjust dimensions if your TO_VECTOR expects a specific size (e.g., 768)
        # For the actual test, we need a 768-dim vector. Let's create a dummy one.
        dummy_vector_768 = [0.1] * 768 
        vector_str = f"[{','.join(map(str, dummy_vector_768))}]"

        # Note: The table SourceDocuments and its embedding column must exist
        # and have at least one row with a non-null embedding for this query to potentially return data.
        # The goal here is to see if the *query executes without parsing errors*, not necessarily to get data.
        test_sql = f"""
            SELECT TOP {top_k} doc_id 
            FROM SourceDocuments 
            WHERE embedding IS NOT NULL 
            ORDER BY VECTOR_COSINE(embedding, TO_VECTOR('{vector_str}', 'DOUBLE', 768)) DESC
        """
        
        logger.info(f"Executing SQL (fully inlined):\n{test_sql[:300]}... (vector part truncated)") # Log part of the SQL

        cursor.execute(test_sql) # No parameters passed
        logger.info("SQL query executed successfully.")
        
        results = cursor.fetchall()
        if results:
            logger.info(f"Query returned {len(results)} row(s). First row: {results[0]}")
        else:
            logger.info("Query returned no results (this is okay if the table is empty or no matches).")

    except sqlalchemy.exc.DBAPIError as e:
        logger.error(f"SQLAlchemy DBAPIError occurred: {e}")
        if hasattr(e.orig, 'sqlcode') and hasattr(e.orig, 'message'):
             logger.error(f"  Underlying DBAPI Error: SQLCODE: {e.orig.sqlcode}, Message: {e.orig.message}")
        elif hasattr(e.orig, 'args'):
             logger.error(f"  Underlying DBAPI Error args: {e.orig.args}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        
    finally:
        if cursor:
            cursor.close()
            logger.info("DBAPI cursor closed.")
        if raw_dbapi_connection: # SQLAlchemy's connection.connection doesn't need explicit close if sqla_connection is closed
            pass
        if sqla_connection:
            sqla_connection.close()
            logger.info("SQLAlchemy connection closed.")
        if engine:
            engine.dispose()
            logger.info("SQLAlchemy engine disposed.")

if __name__ == "__main__":
    # Important: This script must be run in the venv_community_test environment
    # where sqlalchemy-iris and its dependencies (including the community IRIS driver)
    # are installed.
    # Also, ensure the IRIS Docker container (e.g., 'iris_benchmark_container') is running
    # and accessible via localhost:51773 (or as configured).
    # The SourceDocuments table should exist.
    run_test()
