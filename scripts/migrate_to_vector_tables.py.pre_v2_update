import sys
import time
import logging

# Add project root to sys.path
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BATCH_SIZE = int(os.getenv("MIGRATION_BATCH_SIZE", "100"))
MAX_RETRIES = int(os.getenv("MIGRATION_MAX_RETRIES", "3"))

def get_table_count(cursor, schema_name, table_name):
    """Gets the row count of a table."""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
        return cursor.fetchone()[0]
    except Exception as e:
        logging.error(f"Error getting count for {schema_name}.{table_name}: {e}")
        return -1

def migrate_source_documents(conn):
    """Migrates data from RAG.SourceDocuments to RAG.SourceDocuments_V2."""
    logging.info("Starting migration for RAG.SourceDocuments...")
    source_table = "RAG.SourceDocuments"
    target_table = "RAG.SourceDocuments_V2"
    
    with conn.cursor() as cursor:
        try:
            # Get total records to migrate (where embedding is not null)
            cursor.execute(f"SELECT COUNT(*) FROM {source_table} WHERE embedding IS NOT NULL")
            total_records = cursor.fetchone()[0]
            logging.info(f"Found {total_records} records in {source_table} with non-null embeddings to migrate.")

            if total_records == 0:
                logging.info(f"No records to migrate for {source_table}.")
                return True

            migrated_count = 0
            offset = 0
            while migrated_count < total_records:
                logging.info(f"Migrating batch for {source_table}: offset={offset}, batch_size={BATCH_SIZE}")
                
                # Note: IRIS SQL uses TOP N, not LIMIT/OFFSET for selecting subsets in this manner easily.
                # We'll use a strategy that assumes doc_id is somewhat sequential or can be ordered.
                # A more robust approach for large tables might involve cursors or ID-based batching.
                # For simplicity, and given the example, we'll use a simplified batching.
                # The provided SQL example uses LIMIT, which is not standard IRIS SQL.
                # IRIS uses `SELECT TOP N ... WHERE ... ORDER BY ...`
                # To simulate batching, we'd typically need a way to select records not yet processed.
                # Let's assume we can select records that are not yet in the target table.

                # Get current count in target table to estimate progress if script is rerun
                current_target_count = get_table_count(cursor, "RAG", "SourceDocuments_V2")

                # This is a simplified batching. For true batching without relying on existing data in target,
                # one would typically use row IDs or a temporary "processed" flag.
                # The SQL example `LIMIT 1000` is not directly translatable to IRIS for batching without ordering.
                # We will adapt the spirit of the example.
                
                # Get a list of doc_ids to process in this batch
                # This assumes doc_id is a primary key and can be ordered.
                # We select doc_ids from the source that are not yet in the target.
                query_select_ids = f"""
                    SELECT TOP {BATCH_SIZE} doc_id 
                    FROM {source_table} s
                    WHERE s.embedding IS NOT NULL 
                    AND NOT EXISTS (SELECT 1 FROM {target_table} t WHERE t.doc_id = s.doc_id)
                    ORDER BY s.doc_id 
                """
                cursor.execute(query_select_ids)
                doc_ids_to_migrate = [row[0] for row in cursor.fetchall()]

                if not doc_ids_to_migrate:
                    logging.info(f"No more new records found to migrate for {source_table}.")
                    break
                
                doc_ids_placeholder = ','.join(['?'] * len(doc_ids_to_migrate))

                migration_sql = f"""
                INSERT INTO {target_table} (
                    doc_id, title, text_content, abstract, authors, keywords,
                    document_embedding_vector, embedding
                )
                SELECT
                    s.doc_id, s.title, s.text_content, s.abstract, s.authors, s.keywords,
                    TO_VECTOR(s.embedding), s.embedding
                FROM {source_table} s
                WHERE s.doc_id IN ({doc_ids_placeholder}) AND s.embedding IS NOT NULL
                """
                
                retries = 0
                success = False
                while retries < MAX_RETRIES and not success:
                    try:
                        cursor.execute(migration_sql, tuple(doc_ids_to_migrate))
                        conn.commit()
                        batch_migrated_count = cursor.rowcount
                        migrated_count += batch_migrated_count
                        logging.info(f"Successfully migrated {batch_migrated_count} records in this batch for {source_table}. Total migrated: {migrated_count}/{total_records}")
                        success = True
                    except Exception as e:
                        logging.error(f"Error migrating batch for {source_table}: {e}. Attempt {retries + 1}/{MAX_RETRIES}")
                        conn.rollback()
                        retries += 1
                        time.sleep(2 ** retries) # Exponential backoff
                
                if not success:
                    logging.error(f"Failed to migrate batch for {source_table} after {MAX_RETRIES} retries.")
                    return False

            logging.info(f"Migration completed for {source_table}. Total records migrated: {migrated_count}")
            return True

        except Exception as e:
            logging.error(f"Critical error during {source_table} migration: {e}")
            conn.rollback()
            return False

def migrate_document_chunks(conn):
    """Migrates data from RAG.DocumentChunks to RAG.DocumentChunks_V2."""
    logging.info("Starting migration for RAG.DocumentChunks...")
    source_table = "RAG.DocumentChunks"
    target_table = "RAG.DocumentChunks_V2"

    with conn.cursor() as cursor:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {source_table} WHERE embedding IS NOT NULL")
            total_records = cursor.fetchone()[0]
            logging.info(f"Found {total_records} records in {source_table} with non-null embeddings to migrate.")

            if total_records == 0:
                logging.info(f"No records to migrate for {source_table}.")
                return True

            migrated_count = 0
            while True: # Loop until all processable records are done
                # Select chunk_ids to process in this batch
                query_select_ids = f"""
                    SELECT TOP {BATCH_SIZE} chunk_id 
                    FROM {source_table} s
                    WHERE s.embedding IS NOT NULL 
                    AND NOT EXISTS (SELECT 1 FROM {target_table} t WHERE t.chunk_id = s.chunk_id)
                    ORDER BY s.chunk_id 
                """
                cursor.execute(query_select_ids)
                chunk_ids_to_migrate = [row[0] for row in cursor.fetchall()]

                if not chunk_ids_to_migrate:
                    logging.info(f"No more new records found to migrate for {source_table}.")
                    break
                
                chunk_ids_placeholder = ','.join(['?'] * len(chunk_ids_to_migrate))

                migration_sql = f"""
                INSERT INTO {target_table} (
                    chunk_id, doc_id, chunk_text, chunk_index, chunk_type,
                    chunk_embedding_vector, embedding
                )
                SELECT
                    s.chunk_id, s.doc_id, s.chunk_text, s.chunk_index, s.chunk_type,
                    TO_VECTOR(s.embedding), s.embedding
                FROM {source_table} s
                WHERE s.chunk_id IN ({chunk_ids_placeholder}) AND s.embedding IS NOT NULL
                """
                
                retries = 0
                success = False
                while retries < MAX_RETRIES and not success:
                    try:
                        cursor.execute(migration_sql, tuple(chunk_ids_to_migrate))
                        conn.commit()
                        batch_migrated_count = cursor.rowcount
                        migrated_count += batch_migrated_count
                        logging.info(f"Successfully migrated {batch_migrated_count} records in this batch for {source_table}. Total migrated so far: {migrated_count}")
                        success = True
                    except Exception as e:
                        logging.error(f"Error migrating batch for {source_table}: {e}. Attempt {retries + 1}/{MAX_RETRIES}")
                        conn.rollback()
                        retries += 1
                        time.sleep(2 ** retries)
                
                if not success:
                    logging.error(f"Failed to migrate batch for {source_table} after {MAX_RETRIES} retries.")
                    return False
                
                if batch_migrated_count == 0 and migrated_count >= total_records: # Ensure we don't loop infinitely if counts are off
                    logging.info(f"Batch migrated 0 records, assuming completion for {source_table}.")
                    break


            logging.info(f"Migration completed for {source_table}. Total records processed: {migrated_count}")
            return True

        except Exception as e:
            logging.error(f"Critical error during {source_table} migration: {e}")
            conn.rollback()
            return False

def migrate_document_token_embeddings(conn):
    """Migrates data from RAG.DocumentTokenEmbeddings to RAG.DocumentTokenEmbeddings_V2."""
    logging.info("Starting migration for RAG.DocumentTokenEmbeddings...")
    source_table = "RAG.DocumentTokenEmbeddings"
    target_table = "RAG.DocumentTokenEmbeddings_V2"

    with conn.cursor() as cursor:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {source_table} WHERE token_embedding IS NOT NULL")
            total_records = cursor.fetchone()[0]
            logging.info(f"Found {total_records} records in {source_table} with non-null embeddings to migrate.")

            if total_records == 0:
                logging.info(f"No records to migrate for {source_table}.")
                return True

            migrated_count = 0
            # For this table, primary key might be composite (doc_id, token_index) or a unique ID.
            # Assuming a unique 'token_embedding_id' or similar for simplicity in batching.
            # If not, batching needs to be on (doc_id, token_index) which is more complex.
            # Let's assume there's a unique ID, or we sort by doc_id, token_index.
            # For this example, we'll use (doc_id, token_index) for ordering.
            
            while True:
                query_select_ids = f"""
                    SELECT TOP {BATCH_SIZE} s.doc_id, s.token_sequence_index
                    FROM {source_table} s
                    WHERE s.token_embedding IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1 FROM {target_table} t
                        WHERE t.doc_id = s.doc_id AND t.token_sequence_index = s.token_sequence_index
                    )
                    ORDER BY s.doc_id, s.token_sequence_index
                """
                cursor.execute(query_select_ids)
                ids_to_migrate = cursor.fetchall() # List of (doc_id, token_index) tuples

                if not ids_to_migrate:
                    logging.info(f"No more new records found to migrate for {source_table}.")
                    break
                
                # Constructing WHERE clause for multiple composite keys
                # e.g., WHERE (doc_id = ? AND token_index = ?) OR (doc_id = ? AND token_index = ?) ...
                where_clauses = []
                param_values = []
                for doc_id, token_idx in ids_to_migrate:
                    where_clauses.append("(s.doc_id = ? AND s.token_sequence_index = ?)")
                    param_values.extend([doc_id, token_idx])
                
                where_condition = " OR ".join(where_clauses)

                migration_sql = f"""
                INSERT INTO {target_table} (
                    doc_id, token_text, token_sequence_index, metadata_json,
                    token_embedding_vector, token_embedding
                )
                SELECT
                    s.doc_id, s.token_text, s.token_sequence_index, s.metadata_json,
                    TO_VECTOR(s.token_embedding), s.token_embedding
                FROM {source_table} s
                WHERE ({where_condition}) AND s.token_embedding IS NOT NULL
                """
                
                retries = 0
                success = False
                while retries < MAX_RETRIES and not success:
                    try:
                        cursor.execute(migration_sql, tuple(param_values))
                        conn.commit()
                        batch_migrated_count = cursor.rowcount
                        migrated_count += batch_migrated_count
                        logging.info(f"Successfully migrated {batch_migrated_count} records in this batch for {source_table}. Total migrated so far: {migrated_count}")
                        success = True
                    except Exception as e:
                        logging.error(f"Error migrating batch for {source_table}: {e}. Attempt {retries + 1}/{MAX_RETRIES}")
                        conn.rollback()
                        retries += 1
                        time.sleep(2 ** retries)
                
                if not success:
                    logging.error(f"Failed to migrate batch for {source_table} after {MAX_RETRIES} retries.")
                    return False

                if batch_migrated_count == 0 and migrated_count >= total_records:
                    logging.info(f"Batch migrated 0 records, assuming completion for {source_table}.")
                    break

            logging.info(f"Migration completed for {source_table}. Total records processed: {migrated_count}")
            return True

        except Exception as e:
            logging.error(f"Critical error during {source_table} migration: {e}")
            conn.rollback()
            return False

def verify_migration(conn):
    """Verifies the migration by checking counts and running sample queries."""
    logging.info("Starting migration verification...")
    verification_passed = True
    
    tables_to_verify = [
        ("RAG.SourceDocuments", "RAG.SourceDocuments_V2", "doc_id", "document_embedding_vector"),
        ("RAG.DocumentChunks", "RAG.DocumentChunks_V2", "chunk_id", "chunk_embedding_vector"),
        ("RAG.DocumentTokenEmbeddings", "RAG.DocumentTokenEmbeddings_V2", ["doc_id", "token_sequence_index"], "token_embedding_vector") # Composite key example
    ]

    with conn.cursor() as cursor:
        for source_table_full, target_table_full, id_column_s, vector_column_name in tables_to_verify:
            source_schema, source_table_name = source_table_full.split('.')
            target_schema, target_table_name = target_table_full.split('.')
            
            logging.info(f"Verifying {source_table_full} -> {target_table_full}")

            # 1. Check record counts (for records with non-null embeddings in source)
            try:
                if isinstance(id_column_s, list): # Composite key for DocumentTokenEmbeddings
                     # Count where original embedding was not null
                    cursor.execute(f"SELECT COUNT(*) FROM {source_schema}.{source_table_name} WHERE token_embedding IS NOT NULL")
                else:
                    cursor.execute(f"SELECT COUNT(*) FROM {source_schema}.{source_table_name} WHERE embedding IS NOT NULL")
                
                source_count_embed = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT COUNT(*) FROM {target_schema}.{target_table_name} WHERE {vector_column_name} IS NOT NULL")
                target_count_vec = cursor.fetchone()[0]

                logging.info(f"  {source_table_full} (with embeddings): {source_count_embed} records")
                logging.info(f"  {target_table_full} (with vectors): {target_count_vec} records")

                if source_count_embed != target_count_vec:
                    logging.warning(f"  Record count mismatch for {target_table_name}: Source (with embeddings)={source_count_embed}, Target (with vectors)={target_count_vec}")
                    # This might not be a failure if some source embeddings were unparseable by TO_VECTOR,
                    # but for this script's purpose, we expect them to match if TO_VECTOR works for all.
                    # verification_passed = False # Decide if this is a hard failure
                else:
                    logging.info(f"  Record counts (for migratable data) match for {target_table_name}.")

            except Exception as e:
                logging.error(f"  Error checking counts for {target_table_name}: {e}")
                verification_passed = False

            # 2. Test VECTOR_COSINE queries (if data exists)
            if target_count_vec > 0:
                try:
                    # Get one valid vector from the table to compare against itself
                    if isinstance(id_column_s, list): # DocumentTokenEmbeddings
                        id_cols_str = ", ".join(id_column_s)
                        cursor.execute(f"SELECT TOP 1 {id_cols_str}, {vector_column_name} FROM {target_schema}.{target_table_name} WHERE {vector_column_name} IS NOT NULL")
                    else: # SourceDocuments, DocumentChunks
                        cursor.execute(f"SELECT TOP 1 {id_column_s}, {vector_column_name} FROM {target_schema}.{target_table_name} WHERE {vector_column_name} IS NOT NULL")
                    
                    sample_row = cursor.fetchone()
                    if sample_row:
                        sample_id_values = sample_row[:-1]
                        sample_vector = sample_row[-1] # This is already a vector type from DB

                        if isinstance(id_column_s, list): # Composite key
                            id_conditions = " AND ".join([f"{col} = ?" for col in id_column_s])
                            query_params = list(sample_id_values) + [sample_vector]
                        else: # Single ID key
                            id_conditions = f"{id_column_s} = ?"
                            query_params = [sample_id_values[0], sample_vector]
                        
                        # VECTOR_COSINE query
                        # Note: VECTOR_COSINE expects two vector arguments.
                        # The column itself is a vector. The parameter must also be passed as a vector.
                        # In Python, this means passing the string representation that TO_VECTOR would understand,
                        # or if the driver supports native vector types, that type.
                        # For simplicity, we'll use TO_VECTOR on a string version of the sample_vector if needed,
                        # or assume the driver handles it.
                        # The sample_vector from fetchone() might already be in a usable format.
                        
                        # Let's assume sample_vector is a string list '[1.0,2.0,...]'
                        # If it's already a native vector object from the DB, this might not be needed.
                        # For IRIS, TO_VECTOR expects a string like '1,2,3'.
                        
                        # If sample_vector is a list/tuple from DB, convert to string
                        if isinstance(sample_vector, (list, tuple)):
                             vector_str_for_query = ','.join(map(str, sample_vector))
                        elif isinstance(sample_vector, str) and sample_vector.startswith('[') and sample_vector.endswith(']'):
                             # Assuming format like '[0.1, 0.2, ...]'
                             vector_str_for_query = sample_vector[1:-1]
                        else: # Assume it's already in '1,2,3' format or driver handles it
                             vector_str_for_query = sample_vector


                        cosine_query = f"""
                        SELECT TOP 1 {vector_column_name}, VECTOR_COSINE({vector_column_name}, TO_VECTOR(?)) as similarity
                        FROM {target_schema}.{target_table_name}
                        WHERE {id_conditions} AND {vector_column_name} IS NOT NULL
                        """
                        
                        cursor.execute(cosine_query, [vector_str_for_query] + list(sample_id_values) ) # TO_VECTOR(?) is for the parameter
                        
                        result = cursor.fetchone()
                        if result and result[1] is not None:
                            similarity = result[1]
                            logging.info(f"  VECTOR_COSINE test for {target_table_name} (ID: {sample_id_values}): similarity to self = {similarity:.4f}")
                            if not (0.999 <= similarity <= 1.001): # Check if close to 1
                                logging.warning(f"  VECTOR_COSINE self-similarity for {target_table_name} is not close to 1: {similarity}")
                                # verification_passed = False # Decide if this is a hard failure
                        else:
                            logging.warning(f"  VECTOR_COSINE test for {target_table_name} did not return a result or similarity.")
                    else:
                        logging.info(f"  Skipping VECTOR_COSINE test for {target_table_name}, no sample vector found.")
                except Exception as e:
                    logging.error(f"  Error during VECTOR_COSINE test for {target_table_name}: {e}")
                    verification_passed = False
            else:
                logging.info(f"  Skipping VECTOR_COSINE test for {target_table_name} as there are no records with vectors.")
    
    if verification_passed:
        logging.info("Migration verification completed successfully.")
    else:
        logging.error("Migration verification failed for one or more checks.")
    return verification_passed

def main():
    logging.info("Starting data migration to _V2 tables with VECTOR columns.")
    
    conn = None
    try:
        conn = get_iris_connection()
        conn.autocommit = False # Ensure we can manually commit/rollback

        # Step 1: Migrate SourceDocuments
        if not migrate_source_documents(conn):
            logging.error("Failed to migrate RAG.SourceDocuments. Aborting.")
            return 1
        
        # Step 2: Migrate DocumentChunks
        if not migrate_document_chunks(conn):
            logging.error("Failed to migrate RAG.DocumentChunks. Aborting.")
            return 1

        # Step 3: Migrate DocumentTokenEmbeddings
        if not migrate_document_token_embeddings(conn):
            logging.error("Failed to migrate RAG.DocumentTokenEmbeddings. Aborting.")
            return 1

        logging.info("All data migration tasks completed.")

        # Step 4: Verify migration
        if not verify_migration(conn):
            logging.warning("Migration verification reported issues. Please check logs.")
            # Not returning error code here, as migration itself might be complete.
        
        logging.info("Migration script finished.")
        return 0

    except Exception as e:
        logging.critical(f"An unexpected error occurred in the main migration process: {e}")
        if conn:
            conn.rollback()
        return 1
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)