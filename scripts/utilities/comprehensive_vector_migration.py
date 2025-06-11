import sys
import os
import time
import logging
import json

# Add project root to sys.path to allow imports from common etc.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection

# --- Configuration ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500  # As specified in the task

# Expected dimensions
DOC_EMBEDDING_DIM = 384
CHUNK_EMBEDDING_DIM = 384
TOKEN_EMBEDDING_DIM = 128

# --- Helper Functions ---
def parse_embedding_varchar(embedding_str: str, expected_dim: int, record_identifier: str) -> list[float] | None:
    """
    Parses a comma-separated string of floats into a list of floats.
    Returns None if parsing fails or dimension is incorrect.
    """
    if not embedding_str:
        logger.warning(f"Empty embedding string for {record_identifier}")
        return None
    try:
        embedding_list = [float(x.strip()) for x in embedding_str.split(',')]
        if len(embedding_list) != expected_dim:
            logger.error(
                f"Dimension mismatch for {record_identifier}: Expected {expected_dim}, got {len(embedding_list)}. "
                f"Embedding preview: {embedding_str[:100]}..."
            )
            return None
        return embedding_list
    except ValueError as e:
        logger.error(f"ValueError parsing embedding for {record_identifier}: {e}. Embedding: {embedding_str[:100]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing embedding for {record_identifier}: {e}. Embedding: {embedding_str[:100]}...")
        return None

def format_vector_for_sql(vector_list: list[float]) -> str:
    """Formats a list of floats into a string like '[f1,f2,...]' for TO_VECTOR()."""
    return '[' + ','.join(map(str, vector_list)) + ']'

# --- Migration Functions ---
def migrate_source_documents(conn):
    logger.info("Starting RAG.SourceDocuments_V2 to RAG.SourceDocuments_V2 migration...")
    start_time = time.time()
    migrated_count = 0
    processed_batches = 0

    with conn.cursor() as cursor:
        # Get total count of documents to migrate for progress reporting
        cursor.execute("""
            SELECT COUNT(*)
            FROM RAG.SourceDocuments_V2 s
            LEFT JOIN RAG.SourceDocuments_V2 s2 ON s.doc_id = s2.doc_id
            WHERE s.embedding IS NOT NULL AND s.embedding <> '' AND s2.doc_id IS NULL
        """)
        total_to_migrate = cursor.fetchone()[0]
        logger.info(f"Total SourceDocuments to migrate: {total_to_migrate}")
        if total_to_migrate == 0:
            logger.info("No new SourceDocuments to migrate.")
            return

        # Iteratively fetch batches of documents that haven't been migrated yet
        # Using a cursor-like approach with TOP and WHERE clause for large tables
        last_processed_doc_id = "" # Assuming doc_id is string and can be ordered

        while True:
            select_query = f"""
                SELECT TOP {BATCH_SIZE} s.doc_id, s.title, s.text_content, s.abstract, s.authors, s.keywords, s.embedding
                FROM RAG.SourceDocuments_V2 s
                LEFT JOIN RAG.SourceDocuments_V2 s2 ON s.doc_id = s2.doc_id
                WHERE s.embedding IS NOT NULL AND s.embedding <> '' AND s2.doc_id IS NULL AND s.doc_id > ?
                ORDER BY s.doc_id
            """
            cursor.execute(select_query, (last_processed_doc_id,))
            batch_records = cursor.fetchall()

            if not batch_records:
                break # No more records to migrate

            insert_data = []
            current_batch_last_doc_id = ""
            for row in batch_records:
                doc_id, title, text_content, abstract, authors, keywords, embedding_varchar = row
                current_batch_last_doc_id = doc_id # Keep track of the last doc_id in this batch
                
                vector_list = parse_embedding_varchar(embedding_varchar, DOC_EMBEDDING_DIM, f"SourceDocument doc_id={doc_id}")
                if vector_list:
                    vector_sql_str = format_vector_for_sql(vector_list)
                    insert_data.append((doc_id, title, text_content, abstract, authors, keywords, embedding_varchar, vector_sql_str))
                else:
                    logger.warning(f"Skipping SourceDocument doc_id={doc_id} due to embedding parsing error.")
            
            if insert_data:
                insert_query = """
                    INSERT INTO RAG.SourceDocuments_V2 
                    (doc_id, title, text_content, abstract, authors, keywords, embedding, document_embedding_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, TO_VECTOR(?))
                """
                try:
                    cursor.executemany(insert_query, insert_data)
                    conn.commit()
                    migrated_count += len(insert_data)
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error inserting batch for SourceDocuments: {e}. Batch skipped. Last doc_id: {current_batch_last_doc_id}")
                    # Potentially log failed doc_ids or retry with smaller batches

            last_processed_doc_id = current_batch_last_doc_id
            processed_batches += 1
            logger.info(f"SourceDocuments: Migrated {migrated_count}/{total_to_migrate} records. Processed batch {processed_batches}.")
            if migrated_count >= total_to_migrate and total_to_migrate > 0 : # Ensure loop terminates if all are processed
                 logger.info(f"All {total_to_migrate} SourceDocuments migrated or processed.")
                 break


    end_time = time.time()
    logger.info(f"Finished RAG.SourceDocuments_V2 migration. Migrated {migrated_count} records in {end_time - start_time:.2f} seconds.")

def migrate_document_chunks(conn):
    logger.info("Starting RAG.DocumentChunks to RAG.DocumentChunks_V2 migration...")
    start_time = time.time()
    migrated_count = 0
    processed_batches = 0

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*)
            FROM RAG.DocumentChunks c
            LEFT JOIN RAG.DocumentChunks_V2 c2 ON c.chunk_id = c2.chunk_id
            WHERE c.embedding IS NOT NULL AND c.embedding <> '' AND c2.chunk_id IS NULL
        """)
        total_to_migrate = cursor.fetchone()[0]
        logger.info(f"Total DocumentChunks to migrate: {total_to_migrate}")
        if total_to_migrate == 0:
            logger.info("No new DocumentChunks to migrate.")
            return

        last_processed_chunk_id = "" 

        while True:
            select_query = f"""
                SELECT TOP {BATCH_SIZE} c.chunk_id, c.doc_id, c.chunk_text, c.chunk_index, c.embedding, c.chunk_type
                FROM RAG.DocumentChunks c
                LEFT JOIN RAG.DocumentChunks_V2 c2 ON c.chunk_id = c2.chunk_id
                WHERE c.embedding IS NOT NULL AND c.embedding <> '' AND c2.chunk_id IS NULL AND c.chunk_id > ?
                ORDER BY c.chunk_id
            """
            cursor.execute(select_query, (last_processed_chunk_id,))
            batch_records = cursor.fetchall()

            if not batch_records:
                break

            insert_data = []
            current_batch_last_chunk_id = ""
            for row in batch_records:
                chunk_id, doc_id, chunk_text, chunk_index, embedding_varchar, chunk_type = row
                current_batch_last_chunk_id = chunk_id
                
                vector_list = parse_embedding_varchar(embedding_varchar, CHUNK_EMBEDDING_DIM, f"DocumentChunk chunk_id={chunk_id}")
                if vector_list:
                    vector_sql_str = format_vector_for_sql(vector_list)
                    insert_data.append((chunk_id, doc_id, chunk_text, chunk_index, embedding_varchar, chunk_type, vector_sql_str))
                else:
                    logger.warning(f"Skipping DocumentChunk chunk_id={chunk_id} due to embedding parsing error.")

            if insert_data:
                insert_query = """
                    INSERT INTO RAG.DocumentChunks_V2
                    (chunk_id, doc_id, chunk_text, chunk_index, embedding, chunk_type, chunk_embedding_vector)
                    VALUES (?, ?, ?, ?, ?, ?, TO_VECTOR(?))
                """
                try:
                    cursor.executemany(insert_query, insert_data)
                    conn.commit()
                    migrated_count += len(insert_data)
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error inserting batch for DocumentChunks: {e}. Batch skipped. Last chunk_id: {current_batch_last_chunk_id}")

            last_processed_chunk_id = current_batch_last_chunk_id
            processed_batches += 1
            logger.info(f"DocumentChunks: Migrated {migrated_count}/{total_to_migrate} records. Processed batch {processed_batches}.")
            if migrated_count >= total_to_migrate and total_to_migrate > 0:
                 logger.info(f"All {total_to_migrate} DocumentChunks migrated or processed.")
                 break

    end_time = time.time()
    logger.info(f"Finished RAG.DocumentChunks migration. Migrated {migrated_count} records in {end_time - start_time:.2f} seconds.")

def migrate_document_token_embeddings(conn):
    logger.info("Starting RAG.DocumentTokenEmbeddings to RAG.DocumentTokenEmbeddings_V2 migration...")
    start_time = time.time()
    migrated_count = 0
    processed_batches = 0

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*)
            FROM RAG.DocumentTokenEmbeddings t
            LEFT JOIN RAG.DocumentTokenEmbeddings_V2 t2 ON t.doc_id = t2.doc_id AND t.token_sequence_index = t2.token_sequence_index
            WHERE t.token_embedding IS NOT NULL AND t.token_embedding <> '' AND t2.doc_id IS NULL
        """)
        total_to_migrate = cursor.fetchone()[0]
        logger.info(f"Total DocumentTokenEmbeddings to migrate: {total_to_migrate}")
        if total_to_migrate == 0:
            logger.info("No new DocumentTokenEmbeddings to migrate.")
            return

        # For composite keys (doc_id, token_sequence_index), cursor-like iteration is more complex.
        # We'll use (doc_id > last_doc_id) OR (doc_id = last_doc_id AND token_sequence_index > last_token_sequence_index)
        last_doc_id = ""  # Assuming doc_id is string
        last_token_sequence_index = -1 # Assuming token_sequence_index is integer

        while True:
            select_query = f"""
                SELECT TOP {BATCH_SIZE} t.doc_id, t.token_sequence_index, t.token_text, t.token_embedding, t.metadata_json
                FROM RAG.DocumentTokenEmbeddings t
                LEFT JOIN RAG.DocumentTokenEmbeddings_V2 t2 ON t.doc_id = t2.doc_id AND t.token_sequence_index = t2.token_sequence_index
                WHERE t.token_embedding IS NOT NULL AND t.token_embedding <> '' AND t2.doc_id IS NULL
                  AND (t.doc_id > ? OR (t.doc_id = ? AND t.token_sequence_index > ?))
                ORDER BY t.doc_id, t.token_sequence_index
            """
            cursor.execute(select_query, (last_doc_id, last_doc_id, last_token_sequence_index))
            batch_records = cursor.fetchall()

            if not batch_records:
                break

            insert_data = []
            current_batch_last_doc_id = ""
            current_batch_last_token_sequence_index = -1

            for row in batch_records:
                doc_id, token_seq_idx, token_text, token_emb_varchar, metadata_json = row
                current_batch_last_doc_id = doc_id
                current_batch_last_token_sequence_index = token_seq_idx
                
                record_id = f"Token doc_id={doc_id}, seq_idx={token_seq_idx}"
                vector_list = parse_embedding_varchar(token_emb_varchar, TOKEN_EMBEDDING_DIM, record_id)
                if vector_list:
                    vector_sql_str = format_vector_for_sql(vector_list)
                    insert_data.append((doc_id, token_seq_idx, token_text, token_emb_varchar, metadata_json, vector_sql_str))
                else:
                    logger.warning(f"Skipping {record_id} due to embedding parsing error.")
            
            if insert_data:
                insert_query = """
                    INSERT INTO RAG.DocumentTokenEmbeddings_V2
                    (doc_id, token_sequence_index, token_text, token_embedding, metadata_json, token_embedding_vector)
                    VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
                """
                try:
                    cursor.executemany(insert_query, insert_data)
                    conn.commit()
                    migrated_count += len(insert_data)
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error inserting batch for DocumentTokenEmbeddings: {e}. Batch skipped. Last processed: ({current_batch_last_doc_id}, {current_batch_last_token_sequence_index})")

            last_doc_id = current_batch_last_doc_id
            last_token_sequence_index = current_batch_last_token_sequence_index
            processed_batches += 1
            logger.info(f"DocumentTokenEmbeddings: Migrated {migrated_count}/{total_to_migrate} records. Processed batch {processed_batches}.")
            if migrated_count >= total_to_migrate and total_to_migrate > 0:
                 logger.info(f"All {total_to_migrate} DocumentTokenEmbeddings migrated or processed.")
                 break
                 
    end_time = time.time()
    logger.info(f"Finished RAG.DocumentTokenEmbeddings migration. Migrated {migrated_count} records in {end_time - start_time:.2f} seconds.")

def create_hnsw_indexes(conn):
    logger.info("Creating HNSW indexes on _V2 tables...")
    start_time = time.time()
    indexes_sql = [
        "CREATE INDEX idx_hnsw_docs_v2 ON RAG.SourceDocuments_V2 (document_embedding_vector) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
        "CREATE INDEX idx_hnsw_chunks_v2 ON RAG.DocumentChunks_V2 (chunk_embedding_vector) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
        "CREATE INDEX idx_hnsw_tokens_v2 ON RAG.DocumentTokenEmbeddings_V2 (token_embedding_vector) AS HNSW(M=16, efConstruction=200, Distance='COSINE')"
    ]

    with conn.cursor() as cursor:
        for i, sql_command in enumerate(indexes_sql):
            index_name = sql_command.split(" ")[2] # Extract index name
            logger.info(f"Attempting to create index: {index_name}...")
            try:
                # Check if index already exists
                # Note: INFORMATION_SCHEMA.INDEXES might not show HNSW index type details or might require specific queries for IRIS
                # For simplicity, we'll try to create and catch exception if it exists, or drop then create.
                # A more robust check would query system tables for the index.
                # Let's assume we can try creating it. If it fails because it exists, it's okay.
                
                # A simple check:
                check_sql = f"SELECT INDEX_NAME FROM INFORMATION_SCHEMA.INDEXES WHERE TABLE_SCHEMA = 'RAG' AND INDEX_NAME = '{index_name}'"
                cursor.execute(check_sql)
                if cursor.fetchone():
                    logger.info(f"Index {index_name} already exists. Skipping creation.")
                    continue

                cursor.execute(sql_command)
                conn.commit() # DDL might auto-commit or require it
                logger.info(f"Successfully created index: {index_name}")
            except Exception as e:
                conn.rollback() # Rollback if DDL was part of a transaction that failed
                logger.error(f"Error creating index {index_name}: {e}")
                logger.warning(f"Index creation for {index_name} might have failed or it might already exist with incompatible definition.")

    end_time = time.time()
    logger.info(f"Finished HNSW index creation attempts in {end_time - start_time:.2f} seconds.")

def validate_migration(conn):
    logger.info("Performing data integrity validation (record counts)...")
    validation_results = {}
    with conn.cursor() as cursor:
        # SourceDocuments
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE embedding IS NOT NULL AND embedding <> ''")
        source_docs_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE document_embedding_vector IS NOT NULL")
        target_docs_count = cursor.fetchone()[0]
        validation_results["SourceDocuments_V2"] = {"source_with_embedding": source_docs_count, "target_v2_with_vector": target_docs_count}
        logger.info(f"SourceDocuments: Source with embedding: {source_docs_count}, Target_V2 with vector: {target_docs_count}")

        # DocumentChunks
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE embedding IS NOT NULL AND embedding <> ''")
        source_chunks_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks_V2 WHERE chunk_embedding_vector IS NOT NULL")
        target_chunks_count = cursor.fetchone()[0]
        validation_results["DocumentChunks"] = {"source_with_embedding": source_chunks_count, "target_v2_with_vector": target_chunks_count}
        logger.info(f"DocumentChunks: Source with embedding: {source_chunks_count}, Target_V2 with vector: {target_chunks_count}")

        # DocumentTokenEmbeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE token_embedding IS NOT NULL AND token_embedding <> ''")
        source_tokens_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings_V2 WHERE token_embedding_vector IS NOT NULL")
        target_tokens_count = cursor.fetchone()[0]
        validation_results["DocumentTokenEmbeddings"] = {"source_with_embedding": source_tokens_count, "target_v2_with_vector": target_tokens_count}
        logger.info(f"DocumentTokenEmbeddings: Source with embedding: {source_tokens_count}, Target_V2 with vector: {target_tokens_count}")

    # Further checks could compare a sample of records if needed.
    # For now, count comparison is a good first step.
    # If source and target counts (for successfully parsed embeddings) match, it's a good sign.
    # Note: The script migrates only if target record doesn't exist. So target_v2_with_vector should ideally equal source_with_embedding
    # if all embeddings were parsable and all records were new. If script is re-run, target_v2_with_vector will be sum of all successful migrations.
    logger.info(f"Validation results: {json.dumps(validation_results, indent=2)}")


# --- Main Execution ---
def run_comprehensive_migration():
    overall_start_time = time.time()
    logger.info("=== Starting Comprehensive Vector Migration ===")
    
    conn = None
    try:
        conn = get_iris_connection()
        if conn is None:
            logger.error("Failed to get database connection. Aborting.")
            return

        # Phase 1: Data Migration
        logger.info("--- Phase 1: Data Migration ---")
        migrate_source_documents(conn)
        migrate_document_chunks(conn)
        migrate_document_token_embeddings(conn)
        logger.info("--- Phase 1: Data Migration Complete ---")

        # Phase 2: Create HNSW Indexes
        logger.info("--- Phase 2: Create HNSW Indexes ---")
        create_hnsw_indexes(conn)
        logger.info("--- Phase 2: Create HNSW Indexes Complete ---")
        
        # Validation
        logger.info("--- Validation Step ---")
        validate_migration(conn)
        logger.info("--- Validation Step Complete ---")

    except Exception as e:
        logger.critical(f"An critical error occurred during the migration process: {e}", exc_info=True)
        if conn: # Attempt to rollback if there was a global transaction context, though individual functions manage commits/rollbacks
            try: conn.rollback()
            except: pass 
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

    overall_end_time = time.time()
    logger.info(f"=== Comprehensive Vector Migration Finished in {overall_end_time - overall_start_time:.2f} seconds ===")

if __name__ == "__main__":
    run_comprehensive_migration()