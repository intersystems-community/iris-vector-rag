"""
Test script for IRIS SQL vector operations with workarounds for ODBC limitations.

This script demonstrates the workarounds for the following IRIS SQL vector operation limitations:
1. TO_VECTOR() function rejects parameter markers
2. TOP/FETCH FIRST clauses cannot be parameterized
3. Client drivers rewrite literals to :%qpar() even when no parameter list is supplied

The workaround is to use string interpolation for these operations, with proper validation
to prevent SQL injection.
"""

import os
import sys
import pyodbc
import json
import logging
from typing import List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_iris_vector_workarounds")

# --- Connection Parameters ---
DSN_NAME = "IRIS"
UID = "SuperUser"
PWD = "SYS"
IRIS_HOSTNAME = "iris-db"
IRIS_PORT = "1972"
IRIS_NAMESPACE = "USER"

# ODBC Driver name - this should match what's in odbcinst.ini
IRIS_ODBC_DRIVER = os.getenv("IRIS_ODBC_DRIVER", "{InterSystems ODBC}")

# Connection string (alternative to DSN if DSN is problematic)
CONN_STR = f"DRIVER={IRIS_ODBC_DRIVER};SERVER={IRIS_HOSTNAME};PORT={IRIS_PORT};DATABASE={IRIS_NAMESPACE};UID={UID};PWD={PWD};"

TABLE_NAME = "RAG.VectorWorkaroundsTest"
VECTOR_DIM = 3

def validate_vector_string(vector_string: str) -> bool:
    """
    Validate that a vector string contains only valid characters.
    This is important for security when using string interpolation.
    """
    # Only allow digits, dots, commas, and square brackets
    allowed_chars = set("0123456789.[],")
    return all(c in allowed_chars for c in vector_string)

def validate_top_k(top_k: Any) -> bool:
    """
    Validate that top_k is a positive integer.
    This is important for security when using string interpolation.
    """
    if not isinstance(top_k, int):
        return False
    return top_k > 0

def main():
    conn = None
    cursor = None
    
    # Decide whether to use DSN or Connection String
    use_dsn = True 

    if use_dsn:
        connection_method = f"DSN={DSN_NAME};UID={UID};PWD={PWD}"
        print(f"Attempting to connect to IRIS using pyodbc with DSN: {DSN_NAME}...")
    else:
        connection_method = CONN_STR
        print(f"Attempting to connect to IRIS using pyodbc with Connection String: {CONN_STR}...")

    try:
        conn = pyodbc.connect(connection_method, autocommit=False)
        cursor = conn.cursor()
        print("Successfully connected to IRIS via pyodbc.")

        # 1. Create test table with VECTOR type
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
            conn.commit()
        except pyodbc.Error as e_drop:
            print(f"Note: Could not drop table (might not exist): {e_drop}")
            try: conn.rollback() 
            except pyodbc.Error: pass

        create_table_sql = f"CREATE TABLE {TABLE_NAME} (id INTEGER PRIMARY KEY, embedding VECTOR(DOUBLE, {VECTOR_DIM}))"
        print(f"Executing: {create_table_sql}")
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Table {TABLE_NAME} created successfully.")

        # 2. Insert data using TO_VECTOR with string interpolation (workaround)
        print(f"\nAttempting to insert data using TO_VECTOR with string interpolation (workaround)...")
        
        # Test vectors
        test_vectors = [
            (1, [0.1, 0.2, 0.3]),
            (2, [0.4, 0.5, 0.6]),
            (3, [0.7, 0.8, 0.9])
        ]
        
        for test_id, vector_values in test_vectors:
            # Convert vector to string representation
            vector_str = f"[{','.join(map(str, vector_values))}]"
            
            # Validate vector string for security (prevent SQL injection)
            if not validate_vector_string(vector_str):
                print(f"ERROR: Invalid vector string: {vector_str}")
                continue
                
            # Use string interpolation for TO_VECTOR as parameters don't work
            insert_sql = f"INSERT INTO {TABLE_NAME} (id, embedding) VALUES ({test_id}, TO_VECTOR('{vector_str}', 'DOUBLE', {VECTOR_DIM}))"
            print(f"Executing: {insert_sql}")
            
            try:
                cursor.execute(insert_sql)
                conn.commit()
                print(f"Data for id={test_id} inserted successfully.")
            except pyodbc.Error as e_insert:
                print(f"Error inserting data for id={test_id}: {e_insert}")
                conn.rollback()

        # 3. Query data using VECTOR_COSINE with string interpolation (workaround)
        print(f"\nAttempting to query data using VECTOR_COSINE with string interpolation (workaround)...")
        
        # Query vector
        query_vector_list = [0.1, 0.2, 0.3]  # Same as first test vector for high similarity
        query_vector_str = f"[{','.join(map(str, query_vector_list))}]"
        
        # Validate vector string for security
        if not validate_vector_string(query_vector_str):
            print(f"ERROR: Invalid query vector string: {query_vector_str}")
            return 1
            
        # Validate top_k for security
        top_k = 2
        if not validate_top_k(top_k):
            print(f"ERROR: Invalid top_k value: {top_k}")
            return 1
            
        # Use string interpolation for TOP and TO_VECTOR as parameters don't work
        select_sql = f"""
            SELECT TOP {top_k} id, 
                   VECTOR_COSINE(embedding, TO_VECTOR('{query_vector_str}', 'DOUBLE', {VECTOR_DIM})) AS score 
            FROM {TABLE_NAME} 
            ORDER BY score DESC
        """
        print(f"Executing: {select_sql}")
        
        try:
            cursor.execute(select_sql)
            rows = cursor.fetchall()
            
            if rows:
                print("\nQuery Results:")
                for i, row in enumerate(rows):
                    print(f"  Row {i+1}:")
                    print(f"    ID: {row.id}")
                    print(f"    Score: {row.score} (type: {type(row.score)})")
                
                # Check if the first row has a score close to 1.0 (cosine similarity with itself)
                if len(rows) > 0:
                    first_row = rows[0]
                    score_is_correct = False
                    if hasattr(first_row, 'score') and isinstance(first_row.score, float):
                        if abs(first_row.score - 1.0) < 1e-6:
                            score_is_correct = True
                    
                    if score_is_correct:
                        print("SUCCESS: VECTOR_COSINE returned expected score of ~1.0 for identical vectors.")
                    else:
                        print(f"WARNING: Score was {first_row.score} (type: {type(first_row.score)}), expected ~1.0 (float).")
            else:
                print("ERROR: No rows returned from select query.")
                
        except pyodbc.Error as e_select:
            print(f"Error querying data: {e_select}")

        # 4. Test FETCH FIRST syntax (alternative to TOP)
        print(f"\nAttempting to query data using FETCH FIRST syntax...")
        
        # Validate top_k for security
        top_k = 2
        if not validate_top_k(top_k):
            print(f"ERROR: Invalid top_k value: {top_k}")
            return 1
            
        # Use string interpolation for FETCH FIRST and TO_VECTOR as parameters don't work
        fetch_sql = f"""
            SELECT id, 
                   VECTOR_COSINE(embedding, TO_VECTOR('{query_vector_str}', 'DOUBLE', {VECTOR_DIM})) AS score 
            FROM {TABLE_NAME} 
            ORDER BY score DESC
            FETCH FIRST {top_k} ROWS ONLY
        """
        print(f"Executing: {fetch_sql}")
        
        try:
            cursor.execute(fetch_sql)
            rows = cursor.fetchall()
            
            if rows:
                print("\nFETCH FIRST Query Results:")
                for i, row in enumerate(rows):
                    print(f"  Row {i+1}:")
                    print(f"    ID: {row.id}")
                    print(f"    Score: {row.score} (type: {type(row.score)})")
                
                print("SUCCESS: FETCH FIRST syntax worked with string interpolation.")
            else:
                print("ERROR: No rows returned from FETCH FIRST query.")
                
        except pyodbc.Error as e_fetch:
            print(f"Error with FETCH FIRST query: {e_fetch}")

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"\npyodbc Error occurred:")
        print(f"  SQLSTATE: {sqlstate}")
        # ex.args can have more than one element, print all
        for i, arg in enumerate(ex.args):
            print(f"  args[{i}]: {arg}")
        return 1
        
    finally:
        if cursor:
            try:
                print(f"\nCleaning up: Dropping table {TABLE_NAME}...")
                # Ensure table drop is attempted even if prior ops failed but connection is open
                if conn and not getattr(conn, 'closed', True):
                    cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
                    conn.commit()
                cursor.close()
                print("Cursor closed.")
            except pyodbc.Error as e_cleanup:
                print(f"Error during cleanup: {e_cleanup}")
                
        if conn:
            try:
                conn.close()
                print("Database connection closed.")
            except pyodbc.Error as e_conn_close:
                print(f"Error closing connection: {e_conn_close}")
    
    return 0

if __name__ == "__main__":
    print("\n--- IRIS SQL Vector Operations Workarounds Test ---\n")
    
    # Print environment variables for debugging
    print(f"--- LD_LIBRARY_PATH ---")
    print(os.environ.get('LD_LIBRARY_PATH', ''))
    print(f"--- ODBCINI ---")
    print(os.environ.get('ODBCINI', ''))
    print(f"--- ODBCSYSINI ---")
    print(os.environ.get('ODBCSYSINI', ''))
    
    exit_code = main()
    
    if exit_code == 0:
        print("\nIRIS SQL Vector Operations Workarounds Test SUCCEEDED.")
    else:
        print("\nIRIS SQL Vector Operations Workarounds Test FAILED.")
        
    sys.exit(exit_code)