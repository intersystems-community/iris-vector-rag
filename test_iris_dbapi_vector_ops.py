import sys
import os
import traceback
import json
import logging

print(f"--- Python version ---")
print(sys.version)
print(f"--- sys.executable ---")
print(sys.executable)
print(f"--- sys.path ---")
for p_idx, p_val in enumerate(sys.path):
    print(f"  sys.path[{p_idx}]: {p_val}")

print(f"--- Attempting import intersystems_iris ---")
try:
    import intersystems_iris
    print("--- intersystems_iris imported successfully ---")
    print(f"intersystems_iris version: {intersystems_iris.__version__ if hasattr(intersystems_iris, '__version__') else 'N/A'}")
    print(f"intersystems_iris module location: {intersystems_iris.__file__ if hasattr(intersystems_iris, '__file__') else 'N/A'}")
except ImportError as e_imp:
    print(f"--- ImportError for intersystems_iris ---")
    print(e_imp)
    traceback.print_exc()
    sys.exit(1)
except ModuleNotFoundError as e_mod:
    print(f"--- ModuleNotFoundError for intersystems_iris ---")
    print(e_mod)
    traceback.print_exc()
    sys.exit(1)
except Exception as e_other:
    print(f"--- Other Exception during intersystems_iris import ---")
    print(e_other)
    traceback.print_exc()
    sys.exit(1)

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

# --- Connection Parameters ---
TABLE_NAME = "RAG.DBAPITestTable"
VECTOR_DIM = 3

def main():
    conn = None
    cursor = None
    
    try:
        print("Attempting to connect to IRIS using DB-API...")
        conn = get_iris_connection()
        
        if not conn:
            raise IRISConnectionError("Failed to get a DB-API connection from iris_connector.")
            
        print(f"DB-API Connection successful. Type: {type(conn)}")
        
        cursor = conn.cursor()
        print(f"DB-API cursor obtained. Type: {type(cursor)}")

        # 1. Create test table
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
            conn.commit()
        except Exception as e_drop:
            print(f"Note: Could not drop table (might not exist): {e_drop}")
            try: conn.rollback() 
            except Exception: pass # May fail if connection is already bad

        # Test VECTOR data type
        create_table_sql = f"CREATE TABLE {TABLE_NAME} (id INTEGER PRIMARY KEY, embedding VECTOR(DOUBLE, {VECTOR_DIM}))"
        print(f"Executing: {create_table_sql}")
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Table {TABLE_NAME} created successfully.")

        # 2. Insert data using TO_VECTOR with string interpolation
        print(f"\nAttempting to insert data using TO_VECTOR with string interpolation...")
        test_id = 1
        test_vector_list = [0.1, 0.2, 0.3]
        test_vector_str = f"[{','.join(map(str, test_vector_list))}]"
        
        # Use string interpolation for TO_VECTOR as parameters don't work
        insert_sql = f"INSERT INTO {TABLE_NAME} (id, embedding) VALUES ({test_id}, TO_VECTOR('{test_vector_str}', 'DOUBLE', {VECTOR_DIM}))"
        print(f"Executing: {insert_sql}")
        cursor.execute(insert_sql)
        conn.commit()
        print("Data inserted successfully using TO_VECTOR with string interpolation.")

        # 3. Query data using VECTOR_COSINE with string interpolation
        print(f"\nAttempting to query data using VECTOR_COSINE with string interpolation...")
        query_vector_list = [0.1, 0.2, 0.3]
        query_vector_str = f"[{','.join(map(str, query_vector_list))}]"
        
        # Use string interpolation for TOP and TO_VECTOR as parameters don't work
        top_k = 5
        select_sql = f"SELECT TOP {top_k} id, VECTOR_COSINE(embedding, TO_VECTOR('{query_vector_str}', 'DOUBLE', {VECTOR_DIM})) AS score FROM {TABLE_NAME} ORDER BY score DESC"
        print(f"Executing: {select_sql}")
        cursor.execute(select_sql)
        rows = cursor.fetchall()
        
        if rows:
            print("\nQuery Results:")
            for i, row in enumerate(rows):
                print(f"  Row {i+1}:")
                # Try different ways to access row data since different drivers return different types
                try:
                    print(f"    ID: {row[0]}")
                    print(f"    Score: {row[1]} (type: {type(row[1])})")
                except (IndexError, TypeError):
                    try:
                        print(f"    ID: {row.id}")
                        print(f"    Score: {row.score} (type: {type(row.score)})")
                    except AttributeError:
                        print(f"    Row data: {row} (type: {type(row)})")
            
            # Check if the first row has a score close to 1.0 (cosine similarity with itself)
            if len(rows) > 0:
                first_row = rows[0]
                score_is_correct = False
                
                # Try different ways to access the score
                score = None
                try:
                    score = first_row[1]
                except (IndexError, TypeError):
                    try:
                        score = first_row.score
                    except AttributeError:
                        print(f"Could not access score from row: {first_row}")
                
                if score is not None and isinstance(score, float):
                    if abs(score - 1.0) < 1e-6:
                        score_is_correct = True
                
                if score_is_correct:
                    print("SUCCESS: VECTOR_COSINE returned expected score of ~1.0 for identical vectors.")
                else:
                    print(f"WARNING: Score was {score} (type: {type(score)}), expected ~1.0 (float).")
        else:
            print("ERROR: No rows returned from select query.")

    except IRISConnectionError as e:
        print(f"DB-API Connection Error during test: {e}")
        traceback.print_exc()
        return 1
    except Exception as e:
        # For intersystems_iris, dbapi exceptions are often subclasses of dbapi.Error
        # Check if it's a DBAPI error to print SQLCODE etc.
        if hasattr(e, 'sqlcode') and hasattr(e, 'message'):
            print(f"DB-API Error occurred: SQLCODE: {e.sqlcode}, Message: {e.message}")
        else:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
        return 1
        
    finally:
        if cursor: 
            cursor.close()
            print("DB-API cursor closed.")
        if conn: 
            conn.close()
            print("DB-API connection closed.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\nDB-API vector operations test completed.")
    else:
        print("\nDB-API vector operations test FAILED.")
    sys.exit(exit_code)