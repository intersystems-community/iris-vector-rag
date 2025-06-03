import sys
import random
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add project root
from common.iris_connector import get_iris_connection

def execute_sql(cursor, sql, description, ignore_errors=None):
    if ignore_errors is None:
        ignore_errors = []
    print(f"Executing: {description}")
    print(f"SQL: {sql.strip()}")
    try:
        cursor.execute(sql)
        print(f"Successfully executed: {description}")
        return True
    except Exception as e:
        err_code_str = str(e)
        # Check if the error is one of the ignorable ones
        for ignorable_code in ignore_errors:
            if ignorable_code in err_code_str:
                print(f"Warning: Ignored error for '{description}' (already exists?): {e}")
                return True # Treat as success if error is ignorable
        
        print(f"Error executing {description}: {e}")
        print(f"Failed SQL: {sql.strip()}")
        return False

def main():
    conn = None
    try:
        conn = get_iris_connection()
        if conn is None:
            print("Error: Could not establish database connection.")
            return

        cursor = conn.cursor()

        # 1. Check existing schemas
        print("\\n--- Step 1: Checking existing schemas ---")
        sql_check_schemas = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME IN ('RAGTEST', 'RAG_HNSW')"
        print(f"Executing: {sql_check_schemas}")
        cursor.execute(sql_check_schemas)
        existing_schemas = [row[0] for row in cursor.fetchall()]
        print(f"Found schemas: {existing_schemas}")

        ragtest_exists = 'RAGTEST' in existing_schemas
        rag_hnsw_exists = 'RAG_HNSW' in existing_schemas

        # 2. Create RAGTEST schema (or confirm existence)
        print("\\n--- Step 2: Creating/Verifying RAGTEST schema ---")
        sql_create_schema = "CREATE SCHEMA RAGTEST"
        # SQLCODE -476: Schema already exists
        if not execute_sql(cursor, sql_create_schema, "Create RAGTEST schema", ignore_errors=["<-476>"]):
            conn.rollback()
            return

        # For RAG_HNSW, just check and report, no creation needed for this task
        if rag_hnsw_exists:
            print("Schema 'RAG_HNSW' was found.")
        else:
            print("Schema 'RAG_HNSW' was not found (as expected for this task).")

        # 3. Create tables with proper VECTOR column definitions
        # Drop table first to ensure a clean state for this test
        print("\\n--- Step 3a: Dropping RAGTEST.SourceDocuments if it exists (for clean test) ---")
        sql_drop_table = "DROP TABLE RAGTEST.SourceDocuments"
        # SQLCODE -30: Table does not exist (if dropping a non-existent table) - this is fine.
        # Or other errors if dependencies exist, but for a clean schema, -30 is common.
        execute_sql(cursor, sql_drop_table, "Drop RAGTEST.SourceDocuments table", ignore_errors=["<-30>"])
        # We commit the drop if it happened, or if it didn't exist, no harm.
        # If drop failed for other reasons, the create might fail, which is intended.
        conn.commit()


        print("\\n--- Step 3b: Creating RAGTEST.SourceDocuments table ---")
        sql_create_table = """
        CREATE TABLE RAGTEST.SourceDocuments (
            doc_id VARCHAR(255) PRIMARY KEY,
            title TEXT,
            content TEXT,
            embedding VECTOR(DOUBLE, 384),  -- Native VECTOR column
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        # SQLCODE -30: Table already exists (should be handled by DROP, but good to have)
        if not execute_sql(cursor, sql_create_table, "Create RAGTEST.SourceDocuments table", ignore_errors=["<-30>"]):
            conn.rollback()
            return

        # 4. Test HNSW index creation
        # Drop index first to ensure a clean state for this test
        print("\\n--- Step 4a: Dropping idx_hnsw_test if it exists (for clean test) ---")
        sql_drop_index = "DROP INDEX RAGTEST.idx_hnsw_test"
        # SQLCODE -360: Index does not exist
        execute_sql(cursor, sql_drop_index, "Drop HNSW index idx_hnsw_test", ignore_errors=["<-360>"])
        conn.commit()


        print("\\n--- Step 4b: Testing HNSW index creation ---")
        sql_create_hnsw_index = """
        CREATE INDEX idx_hnsw_test
        ON RAGTEST.SourceDocuments (embedding)
        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
        """
        # SQLCODE -361: Index already exists
        if not execute_sql(cursor, sql_create_hnsw_index, "Create HNSW index on RAGTEST.SourceDocuments(embedding)", ignore_errors=["<-361>"]):
            conn.rollback()
            return
        
        print("\\nSchema, table, and HNSW index creation/verification steps completed.")

        # 5. If successful, load a small sample of data
        print("\\n--- Step 5: Loading sample data into RAGTEST.SourceDocuments ---")
        num_sample_docs = 5
        sample_docs_data = []
        for i in range(num_sample_docs):
            doc_id = f"RAGTEST_DOC_{i+1:03}"
            title = f"Sample Document Title {i+1}"
            content = f"This is the sample content for document {i+1}. " + "Lorem ipsum dolor sit amet. " * 20
            # Generate a random 384-dimension vector of doubles
            embedding_vector = [random.uniform(-1.0, 1.0) for _ in range(384)]
            # Convert to comma-separated string for SQL literal
            embedding_str = ','.join(map(str, embedding_vector))
            
            sample_docs_data.append((doc_id, title, content, embedding_str))

        insert_sql_template = "INSERT INTO RAGTEST.SourceDocuments (doc_id, title, content, embedding) VALUES (?, ?, ?, TO_VECTOR(?))"
        
        inserted_count = 0
        for doc_data in sample_docs_data:
            print(f"Inserting doc_id: {doc_data[0]}")
            try:
                cursor.execute(insert_sql_template, (doc_data[0], doc_data[1], doc_data[2], doc_data[3]))
                inserted_count +=1
            except Exception as e:
                # SQLCODE -119: Unique constraint violation (if doc_id already exists)
                if "<-119>" in str(e):
                     print(f"Warning: Document {doc_data[0]} already exists. Skipping insertion.")
                else:
                    print(f"Error inserting document {doc_data[0]}: {e}")
                    conn.rollback()
                    return
        
        if inserted_count > 0:
            print(f"Successfully inserted {inserted_count} sample documents.")
        else:
            print("No new sample documents were inserted (they might have existed already).")

        # Test a simple query
        print("\\n--- Step 6: Testing a simple query with the HNSW index ---")
        # Create a random query vector
        query_vector_list = [random.uniform(-1.0, 1.0) for _ in range(384)]
        query_vector_str = ','.join(map(str, query_vector_list))

        # Note: IRIS typically uses $vector.Cosine or $vector.EuclideanDistance for comparisons
        # The HNSW index uses COSINE, so we should aim for a query that leverages that.
        # A direct VECTOR_COSINE in WHERE clause might not always use the HNSW index directly
        # for TOP N queries, but the HNSW index speeds up nearest neighbor searches.
        # For this test, we'll use a query that should benefit from the index.
        
        # This query is more for checking if data is queryable and index is usable
        # rather than a strict performance benchmark here.
        # Explicitly cast the query vector to the same type as the column: VECTOR(DOUBLE, 384)
        sql_test_query = f"""
        SELECT TOP 3 doc_id, title, VECTOR_COSINE(embedding, TO_VECTOR('{query_vector_str}', DOUBLE, 384)) AS similarity
        FROM RAGTEST.SourceDocuments
        ORDER BY similarity DESC
        """
        print(f"Executing test query with explicit vector typing...")
        start_time = time.time()
        if execute_sql(cursor, sql_test_query, "Test query on RAGTEST.SourceDocuments"):
            end_time = time.time()
            print(f"Test query executed successfully in {end_time - start_time:.4f} seconds.")
            results = cursor.fetchall()
            print("Query results:")
            for row in results:
                print(f"  Doc ID: {row[0]}, Title: {row[1]}, Similarity: {row[2]}")
        else:
            print("Test query failed.")


        conn.commit()
        print("\\nAll steps completed successfully.")

    except Exception as e:
        print(f"An unexpected error occurred during the main script execution: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()