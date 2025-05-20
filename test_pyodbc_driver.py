import pyodbc
import os
import logging
import time

# Configure basic logging to see output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_pyodbc_connection_string():
    db_host = os.environ.get("IRIS_HOST", "localhost")
    db_port = int(os.environ.get("IRIS_PORT", 51773)) 
    db_namespace = os.environ.get("IRIS_NAMESPACE", "USER")
    db_user = os.environ.get("IRIS_USERNAME", "test") 
    db_password = os.environ.get("IRIS_PASSWORD", "test")
    driver_name = "InterSystems ODBC"
    
    return (
        f"DRIVER={{{driver_name}}};"
        f"SERVER={db_host};"
        f"PORT={db_port};"
        f"DATABASE={db_namespace};"
        f"UID={db_user};"
        f"PWD={db_password};"
        f"Protocol=TCP;"
    )

def run_pyodbc_test_old_minimal_echo(): 
    logger.info("\n--- Starting run_pyodbc_test_old_minimal_echo (testing old RAG.MinimalEcho projection) ---")
    # ... (content of this function remains the same as before, kept for historical reference) ...
    conn_str = get_pyodbc_connection_string()
    logger.info(f"Attempting to connect to IRIS using pyodbc with connection string: {conn_str.split('PWD=')[0]}PWD=********;...")
    initial_wait_time = 1 
    logger.info(f"Waiting for {initial_wait_time} seconds for IRIS service to be ready...")
    time.sleep(initial_wait_time)
    connection = None
    cursor = None
    target_sp_name_test = "MinimalEcho" 
    try:
        connection = pyodbc.connect(conn_str, autocommit=True)
        logger.info(f"pyodbc connection obtained for old MinimalEcho test.")
        cursor = connection.cursor()
        logger.info("pyodbc cursor obtained for old MinimalEcho test.")
        logger.info(f"Attempting to call Stored Procedure (expecting result set from old MinimalEcho): {{CALL \"RAG\".\"{target_sp_name_test}\"()}}")
        try:
            cursor.execute('{CALL "RAG"."MinimalEcho"()}')
            logger.info(f"Stored Procedure RAG.{target_sp_name_test} executed.")
            results = cursor.fetchall()
            logger.info(f"Old RAG.MinimalEcho call results: {results}")
        except pyodbc.Error as e:
             logger.warning(f"pyodbc Error during old RAG.{target_sp_name_test} call (expected if SP removed): {e}")
             if len(e.args) > 1: logger.warning(f"  SQLSTATE: {e.args[0]}, Message: {e.args[1]}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in run_pyodbc_test_old_minimal_echo: {e}", exc_info=True)
    finally:
        if cursor: cursor.close()
        if connection: connection.close()
    logger.info("--- Finished run_pyodbc_test_old_minimal_echo ---")


def test_simple_echo_os_sp(): # Renamed function
    logger.info("\n--- Starting test_simple_echo_os_sp (testing RAG.SimpleEchoOS) ---") # Updated log message
    conn_str = get_pyodbc_connection_string()
    logger.info(f"Attempting to connect to IRIS using pyodbc: {conn_str.split('PWD=')[0]}PWD=********;...")

    connection = None
    cursor = None
    sp_name = "RAG.SimpleEchoOS" # Updated SP name
    input_string = "Hello ObjectScript SP!" # Changed input string slightly for clarity
    
    # The SP RAG.SimpleEchoOS uses "AS EchoedValue", so we expect this column name.
    expected_column_name = "EchoedValue" 

    try:
        connection = pyodbc.connect(conn_str, autocommit=True)
        logger.info(f"pyodbc connection obtained for {sp_name} test.")
        cursor = connection.cursor()
        logger.info(f"pyodbc cursor obtained for {sp_name} test.")

        # Revert to standard CALL syntax
        sql_call = f"{{CALL {sp_name}(?)}}"
        logger.info(f"Attempting to call Stored Procedure: {sql_call} with param: '{input_string}'")
        
        cursor.execute(sql_call, input_string)
        logger.info(f"Stored Procedure {sp_name} executed successfully.")
        
        results = cursor.fetchall()
        logger.info(f"Raw results from {sp_name}: {results}")
        
        if results and len(results) == 1 and results[0]:
            returned_value = None
            actual_column_name_found = None
            
            if hasattr(results[0], expected_column_name):
                returned_value = getattr(results[0], expected_column_name)
                actual_column_name_found = expected_column_name
            elif len(results[0]) == 1: # Fallback to index if named attribute isn't found
                returned_value = results[0][0]
                actual_column_name_found = f"Index 0 (Default)"
                logger.info(f"Accessed by index as column '{expected_column_name}' was not found by hasattr.")

            if returned_value is not None:
                logger.info(f"{sp_name} returned via column '{actual_column_name_found}': '{returned_value}' (type: {type(returned_value)})")
                if returned_value == input_string:
                    logger.info(f"✅ SUCCESS: {sp_name} correctly echoed the input string '{input_string}'.")
                else:
                    logger.error(f"❌ FAILURE: {sp_name} returned '{returned_value}', expected '{input_string}'.")
            else:
                logger.error(f"❌ FAILURE: {sp_name} result found, but could not extract value. Row: {results[0]}")
        else:
            logger.error(f"❌ FAILURE: {sp_name} did not return the expected single row result set. Results: {results}")

    except pyodbc.Error as e:
        logger.error(f"pyodbc Error occurred during {sp_name} test: {e}")
        if len(e.args) > 1: logger.error(f"  SQLSTATE: {e.args[0]}, Message: {e.args[1]}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in test_simple_echo_os_sp: {e}", exc_info=True)
    finally:
        if cursor: cursor.close(); logger.info(f"pyodbc cursor closed for {sp_name} test.")
        if connection: connection.close(); logger.info(f"pyodbc connection closed for {sp_name} test.")

if __name__ == "__main__":
    logger.info("Starting pyodbc driver tests...")
    # run_pyodbc_test_old_minimal_echo() 
    test_simple_echo_os_sp()      # Focus on the new ObjectScript SP test
    logger.info("Finished pyodbc driver tests.")
