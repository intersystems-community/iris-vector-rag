import sys
import time
import logging
import os
import random
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection
# from common.utils import generate_embedding # Assuming this generates a list of floats - Not used

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
SCHEMA_NAME = "RAGTEST" # Using a test schema
ALTER_TABLE_NAME = f"{SCHEMA_NAME}.QuickTestAlterCol"
COMPARE_TABLE_NAME = f"{SCHEMA_NAME}.QuickTestComparePerf"
VECTOR_DIMENSION = 3 # Keep small for easy manual testing and generation
SAMPLE_VECTOR_COUNT = 5
QUERY_REPETITIONS = 10

def cleanup_table(cursor, table_name_full):
    try:
        cursor.execute(f"DROP TABLE {table_name_full}")
        logging.info(f"Table {table_name_full} dropped successfully.")
    except Exception as e:
        if "SQLCODE=-136" in str(e) or "does not exist" in str(e).lower(): # Table does not exist
            logging.info(f"Table {table_name_full} does not exist, no need to drop.")
        else:
            logging.warning(f"Could not drop table {table_name_full}: {e}")

def create_schema_if_not_exists(cursor, schema_name):
    try:
        cursor.execute(f"CREATE SCHEMA {schema_name}")
        logging.info(f"Schema {schema_name} created successfully.")
    except Exception as e:
        if "SQLCODE=-370" in str(e) or "already exists" in str(e).lower(): # Schema already exists
            logging.info(f"Schema {schema_name} already exists.")
        else:
            logging.error(f"Error creating schema {schema_name}: {e}")
            raise

def test_alter_table_feasibility(conn):
    logging.info(f"\n--- 1. Testing ALTER TABLE Feasibility ({ALTER_TABLE_NAME}) ---")
    alter_feasible = False
    with conn.cursor() as cursor:
        create_schema_if_not_exists(cursor, SCHEMA_NAME)
        cleanup_table(cursor, ALTER_TABLE_NAME)

        try:
            # Create table with VARCHAR column
            cursor.execute(f"""
                CREATE TABLE {ALTER_TABLE_NAME} (
                    id INT PRIMARY KEY,
                    embedding_text VARCHAR(MAX)
                )
            """)
            logging.info(f"Created table {ALTER_TABLE_NAME} with VARCHAR column.")

            # Insert sample data
            sample_vec_str = ','.join(map(str, [random.random() for _ in range(VECTOR_DIMENSION)]))
            cursor.execute(f"INSERT INTO {ALTER_TABLE_NAME} (id, embedding_text) VALUES (?, ?)", (1, f"[{sample_vec_str}]"))
            conn.commit()
            logging.info(f"Inserted sample data into {ALTER_TABLE_NAME}.")

            # Attempt ALTER TABLE
            # Note: VECTOR type in IRIS needs dimension and optionally type (FLOAT, DOUBLE, INTEGER, etc.)
            # The exact syntax for ALTER TABLE MODIFY COLUMN might vary or not be supported for this change.
            # Example: ALTER TABLE MyTable ALTER COLUMN MyVarcharCol VECTOR(FLOAT, 10)
            # We'll try the IRIS syntax.
            alter_sql = f"ALTER TABLE {ALTER_TABLE_NAME} ALTER embedding_text VECTOR({str(VECTOR_DIMENSION).upper()}, {VECTOR_DIMENSION})" # TYPE should be like FLOAT, DOUBLE
            # Correcting to use FLOAT as the type, not the dimension string
            alter_sql = f"ALTER TABLE {ALTER_TABLE_NAME} ALTER embedding_text VECTOR(FLOAT, {VECTOR_DIMENSION})"
            logging.info(f"Attempting: {alter_sql}")
            cursor.execute(alter_sql)
            conn.commit()
            logging.info(f"ALTER TABLE command executed successfully for {ALTER_TABLE_NAME}.")
            alter_feasible = True

            # Verify (optional, simple check)
            cursor.execute(f"SELECT embedding_text FROM {ALTER_TABLE_NAME} WHERE id = 1")
            row = cursor.fetchone()
            logging.info(f"Data after alter (raw): {row[0]}")
            if isinstance(row[0], (list, tuple)): # Native vector type often comes back as list/tuple
                 logging.info(f"Column type appears to be VECTOR post-alter.")
            elif isinstance(row[0], str) and row[0].startswith(f"%vector"): # IRIS internal representation
                 logging.info(f"Column type appears to be VECTOR post-alter (internal format: {row[0][:20]}...).")
            else:
                 logging.warning(f"Column type might not have changed as expected post-alter. Type: {type(row[0])}")


        except Exception as e:
            logging.error(f"ALTER TABLE test failed for {ALTER_TABLE_NAME}: {e}")
            conn.rollback()
            alter_feasible = False
        finally:
            cleanup_table(cursor, ALTER_TABLE_NAME)
            conn.commit()
    return alter_feasible

def setup_comparison_table(conn):
    logging.info(f"\n--- 2. Setting up Comparison Table ({COMPARE_TABLE_NAME}) ---")
    with conn.cursor() as cursor:
        create_schema_if_not_exists(cursor, SCHEMA_NAME)
        cleanup_table(cursor, COMPARE_TABLE_NAME)
        try:
            cursor.execute(f"""
                CREATE TABLE {COMPARE_TABLE_NAME} (
                    id INT PRIMARY KEY,
                    varchar_embedding VARCHAR(MAX),
                    vector_embedding VECTOR(FLOAT, {VECTOR_DIMENSION})
                )
            """)
            conn.commit()
            logging.info(f"Table {COMPARE_TABLE_NAME} created successfully.")

            # Insert sample data
            logging.info(f"Inserting {SAMPLE_VECTOR_COUNT} sample vectors...")
            for i in range(SAMPLE_VECTOR_COUNT):
                # Generate a simple vector like [0.1, 0.2, 0.3]
                vec = [round(random.random(), 4) for _ in range(VECTOR_DIMENSION)]
                
                # Store as string for VARCHAR column (e.g., "[0.1,0.2,0.3]")
                varchar_vec_str = f"[{','.join(map(str, vec))}]"
                
                # For TO_VECTOR, IRIS expects a comma-separated string like "0.1,0.2,0.3"
                to_vector_arg_str = ','.join(map(str, vec))

                cursor.execute(f"""
                    INSERT INTO {COMPARE_TABLE_NAME} (id, varchar_embedding, vector_embedding)
                    VALUES (?, ?, TO_VECTOR(?))
                """, (i + 1, varchar_vec_str, to_vector_arg_str))
            conn.commit()
            logging.info(f"Inserted {SAMPLE_VECTOR_COUNT} rows into {COMPARE_TABLE_NAME}.")
            return True
        except Exception as e:
            logging.error(f"Error setting up {COMPARE_TABLE_NAME}: {e}")
            conn.rollback()
            return False

def test_hnsw_index(conn):
    logging.info(f"\n--- 3. Testing HNSW Index Creation on VECTOR Column ---")
    hnsw_success = False
    index_name = f"idx_hnsw_{COMPARE_TABLE_NAME.split('.')[-1]}_vec" # Ensure unique index name
    with conn.cursor() as cursor:
        try:
            # Drop index if it exists from a previous failed run
            try:
                cursor.execute(f"DROP INDEX {index_name} ON {COMPARE_TABLE_NAME}")
                logging.info(f"Dropped existing index {index_name} if it existed.")
                conn.commit()
            except Exception:
                pass # Index might not exist, which is fine

            # Create HNSW index
            # Parameters for HNSW: %DIMENSION, %TYPE, M, efConstruction, efSearch (optional)
            # For this test, only %DIMENSION and %TYPE are critical. %TYPE should match the vector's defined type.
            # Correct HNSW Index creation syntax based on db_init_complete.sql
            # CREATE INDEX IF NOT EXISTS idx_hnsw_source_embedding ON RAG.SourceDocuments (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');
            # The %TYPE in WITH clause for CREATE INDEX ... AS HNSW is not standard. The type is inferred from the column.
            # We need to specify HNSW parameters like M, efConstruction, Distance.
            # For a quick test, default parameters might be sufficient if the AS HNSW syntax works.
            # Let's try a minimal HNSW index creation first.
            hnsw_sql = f"""
                CREATE INDEX {index_name} ON {COMPARE_TABLE_NAME} (vector_embedding) AS HNSW
            """
            # A more complete version with parameters:
            # hnsw_sql = f"""
            #     CREATE INDEX {index_name} ON {COMPARE_TABLE_NAME} (vector_embedding)
            #     AS HNSW(M=16, efConstruction=100, Distance='COSINE')
            # """
            # The error "INDEX expected, IDENTIFIER (HNSW) found" suggests "CREATE HNSW INDEX" is wrong.
            # "CREATE INDEX ... AS HNSW" is the way.
            logging.info(f"Attempting to create HNSW index: {hnsw_sql}")
            start_time = time.perf_counter()
            cursor.execute(hnsw_sql)
            # HNSW index creation can be asynchronous. For a quick test, we assume it's done or errors out.
            # For production, one might need to check %SYS.WorkMgr_WorkItem for completion.
            # Some DDL like index creation might be auto-committed or require explicit commit.
            conn.commit() # Ensure DDL is committed
            end_time = time.perf_counter()
            logging.info(f"HNSW index {index_name} created (or command sent) successfully on vector_embedding. Time: {end_time - start_time:.4f}s")
            hnsw_success = True
        except Exception as e:
            logging.error(f"Failed to create HNSW index on vector_embedding: {e}")
            conn.rollback()
            hnsw_success = False
    return hnsw_success

def run_performance_comparison(conn):
    logging.info(f"\n--- 4. Performance Comparison ---")
    results = {"varchar_to_vector": {"times": [], "avg_time": 0},
               "native_vector": {"times": [], "avg_time": 0}}
    
    # Generate a query vector
    query_vec_list = [round(random.random(), 4) for _ in range(VECTOR_DIMENSION)]
    query_vec_for_to_vector = ','.join(map(str, query_vec_list)) # "0.1,0.2,0.3"

    with conn.cursor() as cursor:
        # Ensure data is in the correct simple string format for varchar_embedding
        logging.info("Adjusting varchar_embedding format for simpler TO_VECTOR in query...")
        for i in range(SAMPLE_VECTOR_COUNT):
            vec = [round(random.random(), 4) for _ in range(VECTOR_DIMENSION)]
            varchar_vec_direct_str = ','.join(map(str, vec)) # "0.1,0.2,0.3"
            to_vector_arg_str = varchar_vec_direct_str # Same string for native vector insertion
            
            # Update varchar_embedding to be '0.1,0.2,0.3'
            cursor.execute(f"UPDATE {COMPARE_TABLE_NAME} SET varchar_embedding = ? WHERE id = ?", (varchar_vec_direct_str, i + 1))
            # Update vector_embedding with the same vector data
            cursor.execute(f"UPDATE {COMPARE_TABLE_NAME} SET vector_embedding = TO_VECTOR(?) WHERE id = ?", (to_vector_arg_str, i + 1))
        conn.commit()
        logging.info("Re-inserted/updated data with varchar_embedding as '0.1,0.2,0.3' string and matching native vectors.")

        # Test Query 1: Native VECTOR column vs. TO_VECTOR(?)
        logging.info(f"Running Native VECTOR query vs TO_VECTOR(?) {QUERY_REPETITIONS} times...")
        # query_vec_for_to_vector is '0.1,0.2,0.3'
        sql_native_vs_param = f"""
            SELECT TOP 3 id, VECTOR_COSINE(vector_embedding, TO_VECTOR(?)) AS similarity
            FROM {COMPARE_TABLE_NAME}
            ORDER BY similarity DESC
        """
        try:
            for i in range(QUERY_REPETITIONS):
                start_time = time.perf_counter()
                cursor.execute(sql_native_vs_param, (query_vec_for_to_vector,))
                res_native = cursor.fetchall()
                end_time = time.perf_counter()
                results["native_vector"]["times"].append(end_time - start_time)
                if i == 0: logging.info(f"  NATIVE VECTOR vs TO_VECTOR(?) query sample result: {res_native}")
            results["native_vector"]["avg_time"] = sum(results["native_vector"]["times"]) / QUERY_REPETITIONS
            logging.info(f"  NATIVE VECTOR vs TO_VECTOR(?) avg query time: {results['native_vector']['avg_time']:.6f}s")
        except Exception as e:
            logging.error(f"Error during Native VECTOR vs TO_VECTOR(?) query: {e}")
            results["native_vector"]["avg_time"] = -1 # Indicate error


        # Test Query 2: VARCHAR + TO_VECTOR() vs. TO_VECTOR(?)
        logging.info(f"Running VARCHAR + TO_VECTOR() query vs TO_VECTOR(?) {QUERY_REPETITIONS} times...")
        sql_varchar_vs_param = f"""
            SELECT TOP 3 id, VECTOR_COSINE(TO_VECTOR(varchar_embedding), TO_VECTOR(?)) AS similarity
            FROM {COMPARE_TABLE_NAME}
            ORDER BY similarity DESC
        """
        try:
            for i in range(QUERY_REPETITIONS):
                start_time = time.perf_counter()
                cursor.execute(sql_varchar_vs_param, (query_vec_for_to_vector,))
                res_varchar = cursor.fetchall()
                end_time = time.perf_counter()
                results["varchar_to_vector"]["times"].append(end_time - start_time)
                if i == 0: logging.info(f"  VARCHAR + TO_VECTOR() vs TO_VECTOR(?) query sample result: {res_varchar}")
            results["varchar_to_vector"]["avg_time"] = sum(results["varchar_to_vector"]["times"]) / QUERY_REPETITIONS
            logging.info(f"  VARCHAR + TO_VECTOR() vs TO_VECTOR(?) avg query time: {results['varchar_to_vector']['avg_time']:.6f}s")
        except Exception as e:
            logging.error(f"Error during VARCHAR + TO_VECTOR() vs TO_VECTOR(?) query: {e}")
            results["varchar_to_vector"]["avg_time"] = -1 # Indicate error

    return results


def main():
    logging.info("Starting Quick Vector Migration & Performance Test...")
    conn = None
    final_summary = {}

    try:
        conn = get_iris_connection()
        conn.autocommit = False # Control transactions

        # 1. Test ALTER TABLE
        alter_feasible = test_alter_table_feasibility(conn)
        final_summary["alter_table_feasible"] = alter_feasible
        logging.info(f"ALTER TABLE VARCHAR to VECTOR feasible: {alter_feasible}")

        # 2. Setup Comparison Table
        if not setup_comparison_table(conn):
            logging.error("Failed to set up comparison table. Aborting further tests.")
            return final_summary # Or raise exception

        # 3. Test HNSW Index Creation
        hnsw_creation_works = test_hnsw_index(conn)
        final_summary["hnsw_index_creation_works"] = hnsw_creation_works
        logging.info(f"HNSW index creation on native VECTOR column works: {hnsw_creation_works}")

        # 4. Performance Comparison
        perf_results = run_performance_comparison(conn)
        final_summary["performance_comparison"] = perf_results
        if perf_results["native_vector"]["avg_time"] > 0 and perf_results["varchar_to_vector"]["avg_time"] > 0:
            native_is_faster = perf_results["native_vector"]["avg_time"] < perf_results["varchar_to_vector"]["avg_time"]
            factor = perf_results["varchar_to_vector"]["avg_time"] / perf_results["native_vector"]["avg_time"] if native_is_faster else perf_results["native_vector"]["avg_time"] / perf_results["varchar_to_vector"]["avg_time"]
            logging.info(f"Native VECTOR performance vs VARCHAR+TO_VECTOR(): Native is {'FASTER' if native_is_faster else 'SLOWER/SAME'} by a factor of ~{factor:.2f}x")
            final_summary["native_vector_faster"] = native_is_faster
            final_summary["performance_factor"] = factor
        else:
            logging.warning("Could not reliably compare performance due to zero/error in timings.")


        # 5. Migration Strategy Assessment (based on findings)
        migration_strategy = "Unknown"
        if alter_feasible:
            migration_strategy = "In-place ALTER TABLE might be possible."
        else:
            migration_strategy = "Create new column, copy data (using TO_VECTOR), drop old column. Or, new table and data migration."
        final_summary["suggested_migration_strategy"] = migration_strategy
        logging.info(f"Suggested migration strategy: {migration_strategy}")


    except Exception as e:
        logging.critical(f"An unexpected error occurred in the main test process: {e}")
        if conn:
            conn.rollback()
        final_summary["error"] = str(e)
    finally:
        if conn:
            # Cleanup the comparison table
            with conn.cursor() as cursor:
                cleanup_table(cursor, COMPARE_TABLE_NAME)
            conn.commit()
            conn.close()
        logging.info("\n--- Quick Test Summary ---")
        logging.info(f"ALTER TABLE Feasible: {final_summary.get('alter_table_feasible')}")
        logging.info(f"HNSW Index on VECTOR Works: {final_summary.get('hnsw_index_creation_works')}")
        if 'performance_comparison' in final_summary:
            logging.info(f"Perf - VARCHAR avg time: {final_summary['performance_comparison']['varchar_to_vector']['avg_time']:.6f}s")
            logging.info(f"Perf - NATIVE avg time: {final_summary['performance_comparison']['native_vector']['avg_time']:.6f}s")
            if 'native_vector_faster' in final_summary:
                 logging.info(f"Native VECTOR Faster: {final_summary['native_vector_faster']} (Factor: {final_summary.get('performance_factor', 0):.2f}x)")
        logging.info(f"Suggested Migration Strategy: {final_summary.get('suggested_migration_strategy')}")
        if 'error' in final_summary:
            logging.error(f"Test ended with error: {final_summary['error']}")
        
        # Output summary as JSON for easier parsing if needed
        print("\n--- JSON SUMMARY ---")
        print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()