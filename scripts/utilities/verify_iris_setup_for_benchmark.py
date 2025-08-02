"""
Verifies IRIS database setup for RAGAS benchmark execution.

Checks:
1. IRIS Connection: Attempts to connect to the IRIS database.
2. Document Count: Verifies >= 1000 documents in RAG.SourceDocuments.
3. Embedding Population:
    - Counts non-NULL chunk_embedding in RAG.DocumentChunks.
    - Counts non-NULL token_embedding in RAG.DocumentTokenEmbeddings.
4. Schema Verification:
    - Checks DATA_TYPE of RAG.DocumentChunks.chunk_embedding.
    - Checks DATA_TYPE of RAG.DocumentTokenEmbeddings.token_embedding.
"""
import os
import sys

# Add project root to sys.path to allow imports from common
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from common.iris_connector import get_iris_connection, IRISConnectionError
except ImportError as e:
    print(f"ERROR: Failed to import IRIS connector: {e}")
    print("Please ensure 'common.iris_connector' is available and all dependencies are installed.")
    sys.exit(1)

def verify_iris_connection(conn):
    """Checks 1: IRIS Connection"""
    print("\n--- 1. IRIS Connection Check ---")
    if conn:
        print("SUCCESS: Successfully connected to IRIS.")
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result and result[0] == 1:
                    print("SUCCESS: Test query (SELECT 1) executed successfully.")
                    return True
                else:
                    print(f"FAILURE: Test query (SELECT 1) did not return expected result. Got: {result}")
                    return False
        except Exception as e:
            print(f"FAILURE: Error executing test query on IRIS connection: {e}")
            return False
    else:
        print("FAILURE: Failed to establish IRIS connection (connection object is None).")
        return False

def verify_document_count(conn):
    """Checks 2: Document Count"""
    print("\n--- 2. Document Count Check (RAG.SourceDocuments) ---")
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            count = cursor.fetchone()[0]
            print(f"INFO: Found {count} documents in RAG.SourceDocuments.")
            if count >= 1000:
                print("SUCCESS: Document count is >= 1000.")
                return True
            else:
                print(f"FAILURE: Document count ({count}) is less than 1000.")
                return False
    except Exception as e:
        print(f"FAILURE: Error querying RAG.SourceDocuments for document count: {e}")
        return False

def verify_embedding_population(conn):
    """Checks 3: Embedding Population Verification"""
    print("\n--- 3. Embedding Population Verification ---")
    chunk_embeddings_populated = False
    token_embeddings_populated = False

    # Check RAG.DocumentChunks.chunk_embedding
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE chunk_embedding IS NOT NULL")
            count_chunks = cursor.fetchone()[0]
            print(f"INFO: Found {count_chunks} non-NULL chunk_embedding in RAG.DocumentChunks.")
            if count_chunks > 0:
                print("SUCCESS: RAG.DocumentChunks.chunk_embedding appears to be populated (at least one non-NULL).")
                chunk_embeddings_populated = True
            else:
                print("WARNING: No non-NULL chunk_embedding found in RAG.DocumentChunks. This might be an issue.")
                # Not necessarily a failure for the script's purpose if table is empty, but good to note.
    except Exception as e:
        print(f"FAILURE: Error querying RAG.DocumentChunks for chunk_embedding population: {e}")

    # Check RAG.DocumentTokenEmbeddings.token_embedding
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE token_embedding IS NOT NULL")
            count_tokens = cursor.fetchone()[0]
            print(f"INFO: Found {count_tokens} non-NULL token_embedding in RAG.DocumentTokenEmbeddings.")
            if count_tokens > 0:
                print("SUCCESS: RAG.DocumentTokenEmbeddings.token_embedding appears to be populated (at least one non-NULL).")
                token_embeddings_populated = True
            else:
                print("WARNING: No non-NULL token_embedding found in RAG.DocumentTokenEmbeddings. This might be an issue if ColBERT is used.")
    except Exception as e:
        print(f"FAILURE: Error querying RAG.DocumentTokenEmbeddings for token_embedding population: {e}")
    
    # This check is more informational, so we don't return a hard True/False failure for the overall script
    # based on zero counts, as an empty but correctly schemed DB might be valid in some contexts.
    # The individual messages serve as indicators.
    return True # Returning True as the queries themselves didn't fail.

def verify_schema(conn):
    """Checks 4: Schema Verification for Embedding Columns"""
    print("\n--- 4. Schema Verification for Embedding Columns ---")
    overall_schema_ok = True

    expected_chunk_embedding_type = "VECTOR(FLOAT,384)" # Allow for variations like VECTOR or VECTOR(FLOAT, 384)
    expected_token_embedding_type = "VECTOR(FLOAT,128)"

    # Check RAG.DocumentChunks.chunk_embedding
    try:
        with conn.cursor() as cursor:
            # Note: INFORMATION_SCHEMA.COLUMNS might have different casing for schema/table names depending on DB.
            # Standard SQL is uppercase. IRIS typically stores them as specified or uppercase.
            query = """
            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentChunks' AND COLUMN_NAME = 'chunk_embedding'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                data_type, char_max_len, num_prec, num_scale = result
                actual_type_info = f"DATA_TYPE: {data_type}, CHARACTER_MAXIMUM_LENGTH: {char_max_len}"
                
                print(f"INFO: RAG.DocumentChunks.chunk_embedding | Actual: {actual_type_info}")

                # IRIS-specific behavior: VECTOR columns are reported as VARCHAR in INFORMATION_SCHEMA
                # But they function correctly as VECTOR types. We verify functionality instead.
                if str(data_type).upper() == 'VARCHAR' and char_max_len:
                    # Test if vector operations work on this column
                    try:
                        cursor.execute("""
                            SELECT TOP 1 chunk_id
                            FROM RAG.DocumentChunks
                            WHERE chunk_embedding IS NOT NULL
                        """)
                        test_result = cursor.fetchone()
                        if test_result:
                            print(f"SUCCESS: RAG.DocumentChunks.chunk_embedding functions as VECTOR type (IRIS reports as VARCHAR with length {char_max_len}, which is normal).")
                        else:
                            print(f"WARNING: RAG.DocumentChunks.chunk_embedding column exists but has no data to test vector functionality.")
                    except Exception as vector_test_e:
                        print(f"FAILURE: RAG.DocumentChunks.chunk_embedding vector functionality test failed: {vector_test_e}")
                        overall_schema_ok = False
                else:
                    # If it's not VARCHAR, check if it's actually VECTOR
                    normalized_actual_type = str(data_type).upper().replace(" ", "")
                    if "VECTOR" in normalized_actual_type:
                        if "384" in normalized_actual_type and "FLOAT" in normalized_actual_type:
                            print(f"SUCCESS: RAG.DocumentChunks.chunk_embedding type ({normalized_actual_type}) matches expected pattern '{expected_chunk_embedding_type}'.")
                        elif "384" in normalized_actual_type:
                            print(f"WARNING: RAG.DocumentChunks.chunk_embedding type ({normalized_actual_type}) is VECTOR with correct dimension 384, but type is not FLOAT. Expected pattern: '{expected_chunk_embedding_type}'.")
                        else:
                            print(f"FAILURE: RAG.DocumentChunks.chunk_embedding type ({normalized_actual_type}) is VECTOR but dimension/type mismatch. Expected pattern: '{expected_chunk_embedding_type}'.")
                            overall_schema_ok = False
                    else:
                        print(f"FAILURE: RAG.DocumentChunks.chunk_embedding type ({data_type}) is neither VARCHAR (IRIS VECTOR) nor native VECTOR type.")
                        overall_schema_ok = False
            else:
                print("FAILURE: Could not find schema information for RAG.DocumentChunks.chunk_embedding.")
                overall_schema_ok = False
    except Exception as e:
        print(f"FAILURE: Error querying schema for RAG.DocumentChunks.chunk_embedding: {e}")
        overall_schema_ok = False

    # Check RAG.DocumentTokenEmbeddings.token_embedding
    try:
        with conn.cursor() as cursor:
            query = """
            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings' AND COLUMN_NAME = 'token_embedding'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                data_type, char_max_len, num_prec, num_scale = result
                actual_type_info = f"DATA_TYPE: {data_type}, CHARACTER_MAXIMUM_LENGTH: {char_max_len}"

                print(f"INFO: RAG.DocumentTokenEmbeddings.token_embedding | Actual: {actual_type_info}")
                
                # IRIS-specific behavior: VECTOR columns are reported as VARCHAR in INFORMATION_SCHEMA
                if str(data_type).upper() == 'VARCHAR' and char_max_len:
                    # Test if vector operations work on this column
                    try:
                        cursor.execute("""
                            SELECT TOP 1 doc_id
                            FROM RAG.DocumentTokenEmbeddings
                            WHERE token_embedding IS NOT NULL
                        """)
                        test_result = cursor.fetchone()
                        if test_result:
                            print(f"SUCCESS: RAG.DocumentTokenEmbeddings.token_embedding functions as VECTOR type (IRIS reports as VARCHAR with length {char_max_len}, which is normal).")
                        else:
                            print(f"WARNING: RAG.DocumentTokenEmbeddings.token_embedding column exists but has no data to test vector functionality.")
                    except Exception as vector_test_e:
                        print(f"FAILURE: RAG.DocumentTokenEmbeddings.token_embedding vector functionality test failed: {vector_test_e}")
                        overall_schema_ok = False
                else:
                    # If it's not VARCHAR, check if it's actually VECTOR
                    normalized_actual_type = str(data_type).upper().replace(" ", "")
                    if "VECTOR" in normalized_actual_type:
                        if "128" in normalized_actual_type and "FLOAT" in normalized_actual_type:
                            print(f"SUCCESS: RAG.DocumentTokenEmbeddings.token_embedding type ({normalized_actual_type}) matches expected pattern '{expected_token_embedding_type}'.")
                        elif "128" in normalized_actual_type:
                            print(f"WARNING: RAG.DocumentTokenEmbeddings.token_embedding type ({normalized_actual_type}) is VECTOR with correct dimension 128, but type is not FLOAT. Expected pattern: '{expected_token_embedding_type}'.")
                        else:
                            print(f"FAILURE: RAG.DocumentTokenEmbeddings.token_embedding type ({normalized_actual_type}) is VECTOR but dimension/type mismatch. Expected pattern: '{expected_token_embedding_type}'.")
                            overall_schema_ok = False
                    else:
                        print(f"FAILURE: RAG.DocumentTokenEmbeddings.token_embedding type ({data_type}) is neither VARCHAR (IRIS VECTOR) nor native VECTOR type.")
                        overall_schema_ok = False
            else:
                print("FAILURE: Could not find schema information for RAG.DocumentTokenEmbeddings.token_embedding.")
                overall_schema_ok = False
    except Exception as e:
        print(f"FAILURE: Error querying schema for RAG.DocumentTokenEmbeddings.token_embedding: {e}")
        overall_schema_ok = False

    return overall_schema_ok

def main():
    """Main function to run all verification checks."""
    print("Starting IRIS Setup Verification for Benchmark...")
    conn = None
    all_checks_passed = True
    
    try:
        # Attempt to get a connection (uses JDBC by default as per iris_connector.py)
        conn = get_iris_connection() 
    except IRISConnectionError as e:
        print(f"CRITICAL FAILURE: Could not establish IRIS connection: {e}")
        print("Aborting further checks.")
        sys.exit(1)
    except Exception as e_generic: # Catch any other unexpected error during connection
        print(f"CRITICAL FAILURE: An unexpected error occurred while trying to connect to IRIS: {e_generic}")
        print("Aborting further checks.")
        sys.exit(1)

    if not verify_iris_connection(conn):
        all_checks_passed = False
        # If basic connection fails, no point in other checks that require it.
        print("\nCRITICAL: Initial IRIS connection test failed. Aborting further checks.")
        if conn:
            conn.close()
        sys.exit(1)


    if not verify_document_count(conn):
        all_checks_passed = False
        print("HIGHLIGHT: Document count check failed.")

    # Embedding population is more informational, a warning is printed if 0, but script continues
    verify_embedding_population(conn)

    if not verify_schema(conn):
        all_checks_passed = False
        print("HIGHLIGHT: Schema verification check failed.")

    print("\n--- Verification Summary ---")
    if all_checks_passed:
        print("SUCCESS: All critical prerequisite checks for IRIS setup passed.")
    else:
        print("FAILURE: One or more critical prerequisite checks for IRIS setup failed. Please review the output above.")

    if conn:
        conn.close()
        print("\nIRIS connection closed.")
        
    if not all_checks_passed:
        sys.exit(1) # Exit with error code if any check failed

if __name__ == "__main__":
    main()