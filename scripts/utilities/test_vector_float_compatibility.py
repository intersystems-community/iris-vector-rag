import jaydebeapi
import numpy as np
import os
import sys

# --- Configuration ---
# IRIS Connection Details from environment variables
IRIS_HOST = os.environ.get("IRIS_HOST", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", 1972))
IRIS_NAMESPACE = os.environ.get("IRIS_NAMESPACE", "USER")
IRIS_USER = os.environ.get("IRIS_USER", "_SYSTEM")
IRIS_PASSWORD = os.environ.get("IRIS_PASSWORD", "SYS")
# Path to the InterSystems IRIS JDBC driver JAR file
IRIS_JDBC_DRIVER_PATH = os.environ.get("IRIS_JDBC_DRIVER_PATH")

TABLE_NAME = "TestVectorFloatCompat"
INDEX_NAME = f"{TABLE_NAME}_HNSW_EmbeddingIndex" # Ensuring index name is somewhat unique
VECTOR_DIM = 384
SAMPLE_DATA_COUNT = 5
TOP_K_RESULTS = 3

# --- Helper Functions ---

def get_iris_connection():
    """Establishes a connection to InterSystems IRIS."""
    if not IRIS_JDBC_DRIVER_PATH:
        print("ERROR: The environment variable IRIS_JDBC_DRIVER_PATH is not set.")
        print("Please set it to the path of your InterSystems IRIS JDBC driver JAR file.")
        print("e.g., export IRIS_JDBC_DRIVER_PATH=/path/to/intersystems-jdbc-XYZ.jar")
        sys.exit(1)
    
    if not os.path.exists(IRIS_JDBC_DRIVER_PATH):
        print(f"ERROR: JDBC driver not found at IRIS_JDBC_DRIVER_PATH: {IRIS_JDBC_DRIVER_PATH}")
        sys.exit(1)

    conn_string = f"jdbc:IRIS://{IRIS_HOST}:{IRIS_PORT}/{IRIS_NAMESPACE}"
    print(f"Attempting to connect to IRIS: {conn_string} as {IRIS_USER} using driver {IRIS_JDBC_DRIVER_PATH}")
    try:
        conn = jaydebeapi.connect(
            "com.intersystems.jdbc.IRISDriver",
            conn_string,
            [IRIS_USER, IRIS_PASSWORD],
            IRIS_JDBC_DRIVER_PATH
        )
        print("Successfully connected to IRIS.")
        return conn
    except Exception as e:
        print(f"Error connecting to IRIS: {e}")
        # Print more details if it's a ClassNotFoundException, often due to wrong JAR path
        if "java.lang.ClassNotFoundException" in str(e):
            print("This might be due to an incorrect IRIS_JDBC_DRIVER_PATH or the JAR file not being accessible.")
        raise

def cleanup_resources(cursor, conn):
    """Drops the test table and index if they exist."""
    print(f"Attempting to drop index {INDEX_NAME}...")
    try:
        # In IRIS, HNSW indexes are often tied to the table and might be dropped with the table.
        # Explicit drop is cleaner if supported directly.
        # If index name is unique: DROP INDEX IndexName
        # If associated with table: DROP INDEX TableName.IndexName or specific ALTER TABLE
        # For HNSW created with ON TABLE, DROP TABLE should handle it.
        # Let's try a direct DROP INDEX first.
        cursor.execute(f"DROP INDEX {INDEX_NAME}")
        conn.commit()
        print(f"Index {INDEX_NAME} dropped successfully.")
    except Exception as e_idx:
        error_msg = str(e_idx).lower()
        if "does not exist" in error_msg or "not found" in error_msg or "unknown index" in error_msg or "object named" in error_msg: # IRIS specific error for unknown index
            print(f"Index {INDEX_NAME} not found (normal if first run or already cleaned).")
        else:
            print(f"Warning: Could not drop index {INDEX_NAME} (may not exist or other issue): {e_idx}")

    print(f"Attempting to drop table {TABLE_NAME}...")
    try:
        cursor.execute(f"DROP TABLE {TABLE_NAME}")
        conn.commit()
        print(f"Table {TABLE_NAME} dropped successfully.")
    except Exception as e_tbl:
        error_msg = str(e_tbl).lower()
        if "does not exist" in error_msg or "not found" in error_msg or "unknown table" in error_msg:
            print(f"Table {TABLE_NAME} not found (normal if first run or already cleaned).")
        else:
            print(f"Warning: Could not drop table {TABLE_NAME} (may not exist or other issue): {e_tbl}")


def create_test_table(cursor, conn):
    """Creates a test table with a VECTOR(FLOAT, N) column."""
    sql = f"""
    CREATE TABLE {TABLE_NAME} (
        ID INT PRIMARY KEY,
        Description VARCHAR(255),
        Embedding VECTOR(FLOAT, {VECTOR_DIM})
    )
    """
    print(f"Creating table {TABLE_NAME} with VECTOR(FLOAT, {VECTOR_DIM}) column...")
    try:
        cursor.execute(sql)
        conn.commit()
        print(f"Table {TABLE_NAME} created successfully.")
    except Exception as e:
        if "name is not unique" in str(e) or "already exists" in str(e).lower():
            print(f"Table {TABLE_NAME} already exists. Skipping creation.")
        else:
            print(f"Error creating table {TABLE_NAME}: {e}")
            raise

def insert_sample_data(cursor, conn):
    """Inserts sample vector data into the test table."""
    print(f"Inserting {SAMPLE_DATA_COUNT} sample data rows into {TABLE_NAME}...")
    sample_data = []
    for i in range(1, SAMPLE_DATA_COUNT + 1):
        # Generate a float32 numpy array, then convert to list of Python floats, then to string for TO_VECTOR
        vector = np.random.rand(VECTOR_DIM).astype(np.float32).tolist()
        sample_data.append((i, f"Item {i}", str(vector))) # TO_VECTOR expects string like '[1.0,2.0,...]'

    sql = f"INSERT INTO {TABLE_NAME} (ID, Description, Embedding) VALUES (?, ?, TO_VECTOR(?))"
    try:
        cursor.executemany(sql, sample_data)
        conn.commit()
        print(f"Inserted {cursor.rowcount} rows into {TABLE_NAME}.")
        assert cursor.rowcount == SAMPLE_DATA_COUNT, f"Expected {SAMPLE_DATA_COUNT} rows to be inserted, but got {cursor.rowcount}"
    except Exception as e:
        print(f"Error inserting data into {TABLE_NAME}: {e}")
        raise

def test_vector_cosine_similarity(cursor, step_name="VECTOR_COSINE similarity search"):
    """Tests VECTOR_COSINE similarity search."""
    print(f"Testing {step_name}...")
    query_vector = np.random.rand(VECTOR_DIM).astype(np.float32).tolist()
    query_vector_str = str(query_vector)

    sql = f"""
    SELECT TOP {TOP_K_RESULTS} ID, Description
    FROM {TABLE_NAME}
    ORDER BY VECTOR_COSINE(Embedding, TO_VECTOR(?)) DESC
    """
    try:
        cursor.execute(sql, (query_vector_str,))
        results = cursor.fetchall()
        print(f"{step_name} results (top {TOP_K_RESULTS}):")
        for row in results:
            print(f"  ID: {row[0]}, Description: {row[1]}")
        
        assert len(results) > 0, f"{step_name} returned no results."
        assert len(results) <= TOP_K_RESULTS, f"{step_name} returned more than {TOP_K_RESULTS} results."
        print(f"{step_name} test passed.")
    except Exception as e:
        print(f"Error during {step_name}: {e}")
        raise

def create_hnsw_index(cursor, conn):
    """Creates an HNSW index on the VECTOR(FLOAT) column."""
    # For VECTOR(FLOAT, N), 'dims' in WITH clause might be optional as it's in the column def.
    # 'distance' must match the search function (COSINE for VECTOR_COSINE).
    sql = f"""
    CREATE HNSW INDEX {INDEX_NAME} ON {TABLE_NAME} (Embedding)
    WITH ('m' = '16', 'efConstruction' = '200', 'distance' = 'COSINE')
    """
    print(f"Creating HNSW Index {INDEX_NAME} on {TABLE_NAME}(Embedding)...")
    try:
        cursor.execute(sql)
        # DDL like CREATE INDEX might be auto-committed or require explicit commit depending on driver/DB.
        # For safety with jaydebeapi, explicit commit is good.
        conn.commit()
        print(f"HNSW Index {INDEX_NAME} created successfully.")
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "duplicate index name" in error_msg or "is not unique" in error_msg:
             print(f"HNSW Index {INDEX_NAME} already exists. Skipping creation.")
        else:
            print(f"Error creating HNSW index {INDEX_NAME}: {e}")
            raise

# --- Main Execution ---

def main():
    conn = None
    print("--- IRIS VECTOR(FLOAT) Compatibility Test Script ---")
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()

        print("\n--- Step 0: Initial cleanup of resources ---")
        cleanup_resources(cursor, conn)

        print("\n--- Step 1: Creating test table ---")
        create_test_table(cursor, conn)

        print("\n--- Step 2: Inserting sample data ---")
        insert_sample_data(cursor, conn)

        print("\n--- Step 3: Testing VECTOR_COSINE similarity (pre-index) ---")
        test_vector_cosine_similarity(cursor, "VECTOR_COSINE similarity search (pre-index)")

        print("\n--- Step 4: Creating HNSW index ---")
        create_hnsw_index(cursor, conn)

        print("\n--- Step 5: Verifying HNSW index (re-running search) ---")
        test_vector_cosine_similarity(cursor, "VECTOR_COSINE similarity search (post-index)")

        print("\n---------------------------------------------")
        print("All tests completed successfully!")
        print("---------------------------------------------")

    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"An error occurred during the test: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Re-raise the exception to ensure script exits with non-zero status on error
        # This helps CI/CD pipelines detect failures.
        # sys.exit(1) # Or just let it propagate
        raise
    finally:
        if conn:
            print("\n--- Final Step: Final cleanup of resources ---")
            # Ensure cursor is valid even if an error occurred mid-script
            try:
                if conn.jconn.isClosed(): # Check if underlying Java connection is closed
                     print("Connection was already closed. Skipping final cleanup.")
                else:
                    cursor = conn.cursor()
                    cleanup_resources(cursor, conn)
            except Exception as e_cleanup:
                print(f"Error during final cleanup: {e_cleanup}")
            
            try:
                if not conn.jconn.isClosed():
                    conn.close()
                    print("IRIS connection closed.")
            except Exception as e_close:
                 print(f"Error closing connection: {e_close}")


if __name__ == "__main__":
    main()