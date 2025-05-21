import sys
import os
import traceback

print(f"--- Python version ---")
print(sys.version)
print(f"--- sys.executable ---")
print(sys.executable)
print(f"--- sys.path ---")
for p_idx, p_val in enumerate(sys.path):
    print(f"  sys.path[{p_idx}]: {p_val}")
print(f"--- LD_LIBRARY_PATH ---")
print(os.environ.get('LD_LIBRARY_PATH'))
print(f"--- ODBCINI ---")
print(os.environ.get('ODBCINI'))
print(f"--- ODBCSYSINI ---")
print(os.environ.get('ODBCSYSINI'))

print(f"--- Attempting import pyodbc ---")
try:
    import pyodbc
    print("--- pyodbc imported successfully ---")
    print(f"pyodbc version: {pyodbc.version if hasattr(pyodbc, 'version') else 'N/A'}")
    print(f"pyodbc module location: {pyodbc.__file__ if hasattr(pyodbc, '__file__') else 'N/A'}")
except ImportError as e_imp:
    print(f"--- ImportError for pyodbc ---")
    print(e_imp)
    traceback.print_exc()
    sys.exit(1) # Exit if pyodbc cannot be imported
except ModuleNotFoundError as e_mod:
    print(f"--- ModuleNotFoundError for pyodbc ---")
    print(e_mod)
    traceback.print_exc()
    sys.exit(1) # Exit if pyodbc cannot be imported
except Exception as e_other:
    print(f"--- Other Exception during pyodbc import ---")
    print(e_other)
    traceback.print_exc()
    sys.exit(1) # Exit if pyodbc cannot be imported

# Original content of test_pyodbc_vector_ops.py starts here
import json # Ensure json is imported as it's used later

# --- Connection Parameters ---
# Adjust these as per your IRIS ODBC DSN configuration
# Common DSN names: 'IRIS', 'IRIS ODBC', 'IRISAPP'
# Or use a full connection string.
# Example DSN based on your odbcinst_docker.ini or local odbc.ini
DSN_NAME = os.getenv("ODBC_DSN", "IRIS") 
UID = os.getenv("IRIS_UID", "SuperUser")
PWD = os.getenv("IRIS_PWD", "SYS")

# For Docker, hostname might be 'localhost' if port is mapped, or container name.
IRIS_HOSTNAME = os.getenv("IRIS_HOSTNAME", "iris-db") # Changed to iris-db for docker-compose service name
IRIS_PORT = os.getenv("IRIS_PORT", "1972")
IRIS_NAMESPACE = os.getenv("IRIS_NAMESPACE", "USER")
# Find your driver name from odbcinst.ini, e.g., "InterSystems ODBC" or "InterSystems IRIS ODBC 32"
# Or, if you installed the InterSystems IRIS client package, it might be something like:
# "/usr/local/intersystems/iris/mgr/lib/libirisodbcu35.so" (Linux example)
# On macOS, it might be in /Library/ODBC/ or similar if installed via package.
# If using Homebrew for unixODBC, drivers might be registered differently.
IRIS_ODBC_DRIVER = os.getenv("IRIS_ODBC_DRIVER", "{InterSystems ODBC}") 

# Connection string (alternative to DSN if DSN is problematic)
# Ensure the DRIVER name matches exactly what's in your odbcinst.ini or the full path to the driver library.
CONN_STR = f"DRIVER={IRIS_ODBC_DRIVER};SERVER={IRIS_HOSTNAME};PORT={IRIS_PORT};DATABASE={IRIS_NAMESPACE};UID={UID};PWD={PWD};"

TABLE_NAME = "RAG.PyODBCTestTable"
VECTOR_DIM = 3

def main():
    conn = None
    cursor = None
    
    # Decide whether to use DSN or Connection String
    # Setting use_dsn = True as odbc.ini is configured with Host=iris-db
    use_dsn = True 

    if use_dsn:
        connection_method = f"DSN={DSN_NAME};UID={UID};PWD={PWD}"
        print(f"Attempting to connect to IRIS using pyodbc with DSN: {DSN_NAME}...")
    else:
        connection_method = CONN_STR
        print(f"Attempting to connect to IRIS using pyodbc with Connection String: {CONN_STR}...")
        print("Ensure IRIS_ODBC_DRIVER, IRIS_HOSTNAME, IRIS_PORT, IRIS_NAMESPACE are correctly set or updated in the script.")

    try:
        conn = pyodbc.connect(connection_method, autocommit=False) # Explicit autocommit setting
        cursor = conn.cursor()
        print("Successfully connected to IRIS via pyodbc.")

        # 1. Create test table
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
            conn.commit()
        except pyodbc.Error as e_drop:
            print(f"Note: Could not drop table (might not exist): {e_drop}")
            try: conn.rollback() 
            except pyodbc.Error: pass # May fail if connection is already bad

        # Test with a simple table first (no VECTOR type)
        simple_table_name = f"{TABLE_NAME}_Simple"
        create_simple_table_sql = f"CREATE TABLE {simple_table_name} (id INTEGER PRIMARY KEY, value1 FLOAT, value2 FLOAT, value3 FLOAT)"
        print(f"Executing: {create_simple_table_sql}")
        cursor.execute(create_simple_table_sql)
        conn.commit()
        print(f"Simple table {simple_table_name} created successfully.")

        # Insert data into simple table
        print(f"\nInserting data into simple table...")
        test_id = 1
        test_vector_list = [0.1, 0.2, 0.3]
        insert_simple_sql = f"INSERT INTO {simple_table_name} (id, value1, value2, value3) VALUES (?, ?, ?, ?)"
        params_insert = (test_id, test_vector_list[0], test_vector_list[1], test_vector_list[2])
        print(f"Executing: {insert_simple_sql} with params {params_insert}")
        cursor.execute(insert_simple_sql, params_insert)
        conn.commit()
        print("Data inserted successfully into simple table.")

        # Query the simple table
        print(f"\nQuerying simple table...")
        select_simple_sql = f"SELECT id, value1, value2, value3 FROM {simple_table_name} WHERE id = ?"
        print(f"Executing: {select_simple_sql} with param {test_id}")
        cursor.execute(select_simple_sql, (test_id,))
        row = cursor.fetchone()
        
        if row:
            print("\nQuery Results from simple table:")
            print(f"  ID: {row.id}")
            print(f"  value1: {row.value1} (type: {type(row.value1)})")
            print(f"  value2: {row.value2} (type: {type(row.value2)})")
            print(f"  value3: {row.value3} (type: {type(row.value3)})")
            
            # Check if values match what we inserted
            values_match = (
                abs(row.value1 - test_vector_list[0]) < 1e-6 and
                abs(row.value2 - test_vector_list[1]) < 1e-6 and
                abs(row.value3 - test_vector_list[2]) < 1e-6
            )
            
            if values_match:
                print("SUCCESS: Retrieved values match what was inserted.")
            else:
                print(f"WARNING: Retrieved values don't match what was inserted.")
                
            # Now try to create a table with VECTOR type
            print(f"\nAttempting to create table with VECTOR type...")
            vector_table_name = f"{TABLE_NAME}_Vector"
            create_vector_table_sql = f"CREATE TABLE {vector_table_name} (id INTEGER PRIMARY KEY, embedding VECTOR(DOUBLE, {VECTOR_DIM}))"
            print(f"Executing: {create_vector_table_sql}")
            cursor.execute(create_vector_table_sql)
            conn.commit()
            print(f"Vector table {vector_table_name} created successfully.")
            
            # Clean up the vector table
            print(f"\nCleaning up: Dropping vector table {vector_table_name}...")
            cursor.execute(f"DROP TABLE IF EXISTS {vector_table_name}")
            conn.commit()
            print(f"Vector table {vector_table_name} dropped successfully.")
        else:
            print("ERROR: No rows returned from select query.")

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"\npyodbc Error occurred:")
        print(f"  SQLSTATE: {sqlstate}")
        # ex.args can have more than one element, print all
        for i, arg in enumerate(ex.args):
            print(f"  args[{i}]: {arg}")
        return 1 # Indicate failure
        
    finally:
        if cursor:
            try:
                print(f"\nCleaning up: Dropping tables...")
                # Ensure table drop is attempted even if prior ops failed but connection is open
                if conn and not getattr(conn, 'closed', True): # Check if conn is not None and not closed
                    cursor.execute(f"DROP TABLE IF EXISTS {simple_table_name}")
                    cursor.execute(f"DROP TABLE IF EXISTS {vector_table_name}")
                    conn.commit()
                    print("Table dropped successfully.")
            except pyodbc.Error as e_cleanup: print(f"Error during cleanup: {e_cleanup}")
        if conn:
            conn.close()
            print("Database connection closed.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\nPyODBC vector operations test completed.")
    else:
        print("\nPyODBC vector operations test FAILED.")
    sys.exit(exit_code)
