import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from common
project_root = Path(__file__).resolve().parent.parent
# Ensure project_root is at the very beginning
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

# Minimal diagnostic prints, if still needed, can be re-added.
# print(f"DEBUG: sys.path: {sys.path}")

try:
    from common.connection_factory import ConnectionFactory
    from common.connector_interface import DBAPIConnectorWrapper # Import for isinstance check
except ImportError as e:
    print(f"Error: Could not import ConnectionFactory or DBAPIConnectorWrapper. Details: {e}")
    print("Ensure common.connection_factory.py exists and common/__init__.py is present.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_sql_from_file(sql_file_path: str):
    """
    Connects to the IRIS database and executes SQL commands from a given file.
    """
    if not Path(sql_file_path).is_file():
        logging.error(f"SQL file not found: {sql_file_path}")
        return False

    try:
        logging.info(f"Attempting to connect to IRIS database using DBAPI...")
        # Use create_connection and pass "dbapi" as a string
        connection_wrapper = ConnectionFactory.create_connection(connection_type="dbapi")
        # Assuming the wrapper has a 'get_native_connection' or similar, or is usable directly
        # For now, let's assume the wrapper itself provides cursor()
        # If the wrapper returns the raw connection, then:
        # native_connection = connection_wrapper.get_native_connection() # Example
        # cursor = native_connection.cursor()
        # Based on DBAPIConnectorWrapper in connection_factory.py, it should wrap the connection
        # and might expose cursor() directly or via the wrapped connection.
        # Let's assume the wrapper itself is the connection object for now,
        # or it has a .connection attribute.
        # The DBAPIConnectorWrapper takes the raw connection and should expose a cursor method.
        # The IRISConnectorInterface should define a cursor() method.
        # Let's assume connection_wrapper is an instance of IRISConnectorInterface
        
        # The IRISConnectorInterface is expected to provide a cursor() method.
        # The DBAPIConnectorWrapper(connection) should implement this.
        cursor = connection_wrapper.cursor()
        # The wrapper itself should handle commit/rollback via the interface
        # raw_connection = connection_wrapper.get_native_connection() # This was incorrect

        logging.info("Successfully connected to IRIS database.")

        with open(sql_file_path, 'r') as f:
            sql_script = f.read()

        # Split script into individual statements if necessary,
        # though many drivers/DBs can handle multi-statement strings.
        # For simplicity, assuming the script can be run as a whole or
        # that individual statements are separated by semicolons and
        # the driver handles it. If not, more complex parsing might be needed.
        # For ALTER TABLE, it's usually a single statement.
        
        logging.info(f"Executing SQL from file: {sql_file_path}")
        # Depending on the DBAPI driver, execute might not support multiple statements directly.
        # If the SQL file contains multiple statements separated by ';',
        # they might need to be executed one by one.
        # For a simple ALTER TABLE, this should be fine.
        
        # Splitting by semicolon for basic multi-statement support
        # This is a naive split and might fail for SQL with semicolons in strings or comments
        statements = [s.strip() for s in sql_script.split(';') if s.strip()]
        
        for i, statement in enumerate(statements):
            if statement.startswith('--'): # Skip SQL comments
                logging.info(f"Skipping comment: {statement[:100]}...")
                continue
            logging.info(f"Executing statement {i+1}/{len(statements)}: {statement[:100]}...") # Log first 100 chars
            cursor.execute(statement)
        
        connection_wrapper.commit() # Commit via the wrapper interface
        logging.info(f"Successfully executed SQL script: {sql_file_path}")
        return True

    except Exception as e:
        logging.error(f"Error executing SQL script {sql_file_path}: {e}")
        # Rollback should also be available on the wrapper if commit is
        # However, the IRISConnectorInterface does not define rollback.
        # For DBAPI, the underlying connection object on the wrapper would have rollback.
        # Let's access self.connection for rollback if it's a DBAPIConnectorWrapper
        if isinstance(connection_wrapper, DBAPIConnectorWrapper) and hasattr(connection_wrapper, 'connection'):
            try:
                logging.info("Attempting rollback on underlying DBAPI connection...")
                connection_wrapper.connection.rollback()
                logging.info("Rollback successful.")
            except Exception as re:
                logging.error(f"Error during rollback: {re}")
        elif hasattr(connection_wrapper, 'rollback'): # Check if wrapper itself has rollback (future-proofing)
             try:
                logging.info("Attempting rollback on wrapper...")
                connection_wrapper.rollback()
                logging.info("Rollback successful.")
             except Exception as re:
                logging.error(f"Error during wrapper rollback: {re}")
        return False
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection_wrapper' in locals() and connection_wrapper:
            connection_wrapper.close() # Close via the wrapper
            logging.info("Database connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute an SQL script on the IRIS database.")
    parser.add_argument("sql_file", help="Path to the .sql file to execute.")
    
    args = parser.parse_args()

    if execute_sql_from_file(args.sql_file):
        logging.info("SQL script execution completed successfully.")
        sys.exit(0)
    else:
        logging.error("SQL script execution failed.")
        sys.exit(1)