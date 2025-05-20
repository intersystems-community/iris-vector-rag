import pyodbc
import os
import logging
import time # Import the time module

# Configure basic logging to see output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pyodbc_test():
    # Connection details
    db_host = os.environ.get("IRIS_HOST", "localhost")
    # Ensure the IRIS Docker container (e.g., 'iris_benchmark_container' from tests) is running
    # and its port 1972 is mapped to host port 51773 (or your test config).
    db_port = int(os.environ.get("IRIS_PORT", 51773)) 
    db_namespace = os.environ.get("IRIS_NAMESPACE", "USER")
    # The user/pass "test"/"test" is created by the conftest.py setup.
    # If running this standalone, ensure the user exists or use SuperUser/SYS.
    db_user = os.environ.get("IRIS_USERNAME", "test") 
    db_password = os.environ.get("IRIS_PASSWORD", "test")

    # DSN-less connection string
    # The DRIVER name must exactly match what's in odbcinst.ini
    # Our odbcinst_docker.ini uses [InterSystems ODBC]
    driver_name = "InterSystems ODBC" 
    
    conn_str = (
        f"DRIVER={{{driver_name}}};"
        f"SERVER={db_host};"
        f"PORT={db_port};"
        f"DATABASE={db_namespace};"
        f"UID={db_user};"
        f"PWD={db_password};"
        # Adding common options that might be needed
        f"Protocol=TCP;"
    )
    
    logger.info(f"Attempting to connect to IRIS using pyodbc with connection string: DRIVER={{{driver_name}}};SERVER={db_host};... (credentials masked)")

    # Add a delay to allow IRIS container to fully initialize on the Docker network
    # Especially important when services are starting up together with docker-compose up
    initial_wait_time = 1 # seconds
    logger.info(f"Waiting for {initial_wait_time} seconds for IRIS service to be ready...")
    time.sleep(initial_wait_time)

    connection = None
    cursor = None

    try:
        connection = pyodbc.connect(conn_str, autocommit=True)
        logger.info(f"pyodbc connection obtained: {type(connection)}")
        
        cursor = connection.cursor()
        logger.info("pyodbc cursor obtained.")

        # Diagnostic: Query INFORMATION_SCHEMA.ROUTINES to find the procedure
        try:
            logger.info("Querying INFORMATION_SCHEMA.ROUTINES for relevant procedures...")
            # Looking for names containing 'CompileVectorSearchUtilsAppClass' or 'VectorSearchUtils' or 'SearchSourceDocuments'
            # ROUTINE_NAME is often the procedure name, SPECIFIC_NAME can be more detailed for class methods
            # Using a more targeted query based on user feedback, focusing on SPECIFIC_NAME
            # SPECIFIC_NAME often holds the ClassMethod name for projected methods.
            # ROUTINE_NAME would be the SQL-friendly procedure name.
            # Broader diagnostic query
            # Diagnostic query looking for the explicitly named SP and the compilation SP
            # Diagnostic query looking for the explicitly named SPs after prefixing
            # Diagnostic query looking for the new explicitly named SP and the compilation SP
            # Diagnostic query looking for RAG.MinimalEcho
            target_sp_name_test = "MinimalEcho"
            schema_query = f"""
                SELECT ROUTINE_SCHEMA, ROUTINE_NAME, SPECIFIC_NAME, DATA_TYPE, ROUTINE_DEFINITION
                FROM INFORMATION_SCHEMA.ROUTINES 
                WHERE ROUTINE_TYPE = 'PROCEDURE' AND 
                      UPPER(ROUTINE_SCHEMA) = 'RAG' AND
                      UPPER(ROUTINE_NAME) = UPPER('{target_sp_name_test}')
            """
            logger.info(f"Executing diagnostic query for RAG.{target_sp_name_test}:\n{schema_query}")
            cursor.execute(schema_query)
            found_routines = cursor.fetchall()
            if found_routines:
                logger.info(f"Found RAG.{target_sp_name_test} in INFORMATION_SCHEMA.ROUTINES:")
                for routine in found_routines:
                    logger.info(f"  Schema: {routine.ROUTINE_SCHEMA}, Routine Name: {routine.ROUTINE_NAME}, Specific Name: {routine.SPECIFIC_NAME}, Returns: {routine.DATA_TYPE}, Definition Preview: {str(routine.ROUTINE_DEFINITION)[:100]}...")
            else:
                logger.warning(f"RAG.{target_sp_name_test} NOT found by diagnostic query in INFORMATION_SCHEMA.ROUTINES.")
        except Exception as e_schema_query:
            logger.error(f"Error querying INFORMATION_SCHEMA.ROUTINES for RAG.{target_sp_name_test}: {e_schema_query}", exc_info=True)

        # Test calling RAG.MinimalEcho (no parameters, returns a result set)
        expected_output_int = 12345
        expected_column_name = "HardcodedNumericValue"
        
        # Standard CALL syntax for procedures returning result sets
        sql_proc_call = '{CALL "RAG"."MinimalEcho"()}'
        
        logger.info(f"Attempting to call Stored Procedure (expecting result set): {sql_proc_call}")
        
        cursor.execute(sql_proc_call) 
        logger.info(f"Stored Procedure RAG.{target_sp_name_test} (expecting result set) executed successfully via pyodbc.")
        
        results = cursor.fetchall()
        # Expecting one row, one column named "HardcodedNumericValue" containing an integer
        if results and len(results) == 1 and results[0] and hasattr(results[0], expected_column_name):
            returned_value = getattr(results[0], expected_column_name)
            logger.info(f"RAG.{target_sp_name_test} returned column '{expected_column_name}': {returned_value} (type: {type(returned_value)})")
            if returned_value == expected_output_int:
                logger.info(f"✅ SUCCESS: RAG.{target_sp_name_test} correctly returned the integer {expected_output_int} in column '{expected_column_name}'.")
            else:
                logger.error(f"❌ FAILURE: RAG.{target_sp_name_test} returned {returned_value}, expected {expected_output_int}.")
        elif results and len(results) == 1 and results[0] and len(results[0]) == 1:
            # Fallback to accessing by index if named attribute isn't found
            returned_value = results[0][0]
            logger.info(f"RAG.{target_sp_name_test} returned (by index): {returned_value} (type: {type(returned_value)})")
            if returned_value == expected_output_int:
                logger.info(f"✅ SUCCESS (by index): RAG.{target_sp_name_test} correctly returned the integer {expected_output_int}.")
            else:
                logger.error(f"❌ FAILURE (by index): RAG.{target_sp_name_test} returned {returned_value}, expected {expected_output_int}.")
        else:
            logger.error(f"❌ FAILURE: RAG.{target_sp_name_test} did not return the expected result set. Results: {results}")

    except pyodbc.Error as e:
        logger.error(f"pyodbc Error occurred during RAG.{target_sp_name_test} (expecting result set) test: {e}")
        # pyodbc errors are often tuples, e.g., (sqlstate, message)
        # Or sometimes more complex error objects.
        # For example, e.args[0] might be the SQLSTATE, e.args[1] the message.
        if len(e.args) > 1:
            logger.error(f"  SQLSTATE: {e.args[0]}")
            logger.error(f"  Message: {e.args[1]}")
             
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        
    finally:
        if cursor:
            cursor.close()
            logger.info("pyodbc cursor closed.")
        if connection:
            connection.close()
            logger.info("pyodbc connection closed.")

if __name__ == "__main__":
    # Important: This script must be run in a clean virtual environment
    # where ONLY pyodbc is installed (and unixODBC is configured with the IRIS driver).
    # Ensure the IRIS Docker container is running and accessible.
    # The SourceDocuments table should exist.
    run_pyodbc_test()
