import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration - PLEASE REVIEW AND UPDATE THESE VALUES ---
SCHEMA_NAME = "RAG"
TABLE_NAME = "SourceDocuments"
OLD_VARCHAR_COLUMN_NAME = "document_embedding_vector"
TEMP_VECTOR_COLUMN_NAME = "embedding_vector_new" # Target: VECTOR(FLOAT, 128)
INTERMEDIATE_TEMP_DOUBLE_COLUMN_NAME = "embedding_vector_temp_double" # Intermediate: VECTOR(FLOAT, 128)
FINAL_VECTOR_COLUMN_NAME = "embedding_vector" # This will be the new name for the vector column

# !! IMPORTANT !! If an HNSW index exists on the OLD_VARCHAR_COLUMN_NAME, specify its exact name here.
# If empty, the script will skip attempting to drop an old HNSW index.
OLD_HNSW_INDEX_NAME = ""

# Suggested name for the new HNSW index on the native VECTOR column.
NEW_HNSW_INDEX_NAME = "idx_hnsw_sourcedocuments_embedding_128d"

# !! IMPORTANT !! Review and update HNSW index parameters
# Parameters for the AS HNSW() clause. Dimension is inferred from the column type.
# M, efConstruction, Distance are based on common/db_init_complete.sql.
# Ensure these are appropriate for a 128-dimension vector.
HNSW_INDEX_PARAMS = "M=16, efConstruction=200, Distance='COSINE'"
# Example for E5-large might use different M or efConstruction.

# Batch size for the UPDATE operation if we decide to implement batching.
# For now, the UPDATE is a single operation.
# BATCH_SIZE_UPDATE = int(os.getenv("MIGRATION_UPDATE_BATCH_SIZE", "10000"))
# MAX_RETRIES = int(os.getenv("MIGRATION_MAX_RETRIES", "3"))

def execute_sql(cursor, sql, params=None, DDL=False):
    """Executes a given SQL statement."""
    logging.info(f"Executing SQL: {sql}" + (f" with params {params}" if params else ""))
    try:
        cursor.execute(sql, params if params else ())
        if not DDL: # DDL statements like ALTER, CREATE, DROP don't have rowcount in the same way
             logging.info(f"SQL executed successfully. Rows affected: {cursor.rowcount if cursor.rowcount is not None else 'N/A (DDL)'}")
        else:
            logging.info(f"DDL SQL executed successfully.")
        return True
    except Exception as e:
        logging.error(f"Error executing SQL: {sql}\n{e}")
        raise

def get_table_count(cursor, schema, table, where_clause=""):
    """Gets the row count of a table, optionally with a WHERE clause."""
    query = f"SELECT COUNT(*) FROM {schema}.{table} {where_clause}"
    try:
        cursor.execute(query)
        return cursor.fetchone()[0]
    except Exception as e:
        logging.error(f"Error getting count for {schema}.{table} with '{where_clause}': {e}")
        return -1

def main_migration():
    logging.info(f"Starting migration for {SCHEMA_NAME}.{TABLE_NAME} to native VECTOR column.")
    conn = None
    
    try:
        conn = get_iris_connection()
        conn.autocommit = False # Manual commit/rollback control
        
        with conn.cursor() as cursor:
            logging.info("--- Step 1: Add new temporary VECTOR column ---")
            # Check if column already exists to make script more idempotent
            cursor.execute(f"""
                SELECT 1 
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{SCHEMA_NAME}' 
                AND TABLE_NAME = '{TABLE_NAME}' 
                AND COLUMN_NAME = '{TEMP_VECTOR_COLUMN_NAME}'
            """)
            if cursor.fetchone():
                logging.info(f"Column {TEMP_VECTOR_COLUMN_NAME} already exists in {SCHEMA_NAME}.{TABLE_NAME}. Skipping add.")
            else:
                sql_add_target_float_column = f"ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} ADD COLUMN {TEMP_VECTOR_COLUMN_NAME} VECTOR(FLOAT, 128)"
                execute_sql(cursor, sql_add_target_float_column, DDL=True)
            conn.commit()
            
            # The data in OLD_VARCHAR_COLUMN_NAME is already in comma-separated format.
            # We just need to wrap it in brackets for TO_VECTOR to parse it correctly.

            logging.info(f"--- Step 2: Populate new VECTOR column '{TEMP_VECTOR_COLUMN_NAME}' from '{OLD_VARCHAR_COLUMN_NAME}' ---")
            
            # Check if INTERMEDIATE_TEMP_DOUBLE_COLUMN_NAME exists and drop it if it does, as it's from a failed strategy.
            cursor.execute(f"""
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{SCHEMA_NAME}' AND TABLE_NAME = '{TABLE_NAME}' AND COLUMN_NAME = '{INTERMEDIATE_TEMP_DOUBLE_COLUMN_NAME}'
            """)
            if cursor.fetchone():
                logging.info(f"Dropping now unused intermediate column {INTERMEDIATE_TEMP_DOUBLE_COLUMN_NAME}.")
                sql_drop_intermediate_column = f"ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} DROP COLUMN {INTERMEDIATE_TEMP_DOUBLE_COLUMN_NAME}"
                execute_sql(cursor, sql_drop_intermediate_column, DDL=True)
                conn.commit()

            sql_populate_column = f"""
            UPDATE {SCHEMA_NAME}.{TABLE_NAME}
            SET {TEMP_VECTOR_COLUMN_NAME} = TO_VECTOR('[' || {OLD_VARCHAR_COLUMN_NAME} || ']')
            WHERE {OLD_VARCHAR_COLUMN_NAME} IS NOT NULL
            AND {OLD_VARCHAR_COLUMN_NAME} <> ''
            AND {TEMP_VECTOR_COLUMN_NAME} IS NULL
            """
            logging.info(f"Attempting to populate {TEMP_VECTOR_COLUMN_NAME} using TO_VECTOR with bracket wrapping.")
            execute_sql(cursor, sql_populate_column)
            conn.commit()

            logging.info("--- Step 3: Verify data integrity (counts) ---")
            count_source_populated = get_table_count(cursor, SCHEMA_NAME, TABLE_NAME, 
                                                     f"WHERE {OLD_VARCHAR_COLUMN_NAME} IS NOT NULL AND {OLD_VARCHAR_COLUMN_NAME} <> ''")
            count_target_populated = get_table_count(cursor, SCHEMA_NAME, TABLE_NAME, 
                                                     f"WHERE {TEMP_VECTOR_COLUMN_NAME} IS NOT NULL")
            
            logging.info(f"Rows in source with non-empty '{OLD_VARCHAR_COLUMN_NAME}': {count_source_populated}")
            logging.info(f"Rows in target with non-null '{TEMP_VECTOR_COLUMN_NAME}': {count_target_populated}")

            if count_source_populated == count_target_populated:
                logging.info("Data integrity check (counts) passed.")
            else:
                logging.warning(f"Data integrity check (counts) FAILED or indicates partial migration. Source: {count_source_populated}, Target: {count_target_populated}. "
                                "This could be due to issues in the 2-stage population (VARCHAR -> VECTOR(FLOAT) -> VECTOR(FLOAT)). "
                                "Check logs for errors in Step 2a or 2b.")
                # Decide if this is a hard stop. For now, it's a warning.

            # Before dropping/renaming, consider dropping the intermediate temporary column if it's no longer needed
            # For now, let's keep it until after the main migration steps for potential debugging.
            # It can be dropped later manually or in a cleanup step.

            logging.info(f"--- Step 4: Drop old HNSW index on VARCHAR column (if specified) ---")
            if OLD_HNSW_INDEX_NAME and OLD_HNSW_INDEX_NAME.strip():
                logging.info(f"Attempting to drop specified old HNSW index: '{OLD_HNSW_INDEX_NAME}'")
                # Note: We are not checking for existence here due to issues with %dictionary.IndexDefinition queries.
                # The DROP INDEX command will fail if the index doesn't exist, which is acceptable.
                # Or, for a more graceful skip, a specific check would be needed if %dictionary queries worked.
                try:
                    sql_drop_old_index = f"DROP INDEX {OLD_HNSW_INDEX_NAME} ON {SCHEMA_NAME}.{TABLE_NAME}"
                    execute_sql(cursor, sql_drop_old_index, DDL=True)
                    conn.commit()
                except Exception as e:
                    logging.warning(f"Could not drop index '{OLD_HNSW_INDEX_NAME}'. It might not exist or another issue occurred: {e}")
                    conn.rollback() # Rollback this specific attempt
            else:
                logging.info(f"No OLD_HNSW_INDEX_NAME specified. Skipping drop of old HNSW index.")
            
            # Ensure commit if we skipped or if drop was successful and committed by execute_sql
            # If drop failed and rolled back, we still want to proceed with other steps.
            # The execute_sql commits on success for DDL. If it raised, it's caught.
            # If OLD_HNSW_INDEX_NAME was empty, no transaction started here.
            # A general commit here might be redundant or interfere if execute_sql handles it.
            # Let's ensure conn.commit() is called if a transaction was effectively made.
            # The current structure of execute_sql doesn't commit itself, the main loop does.
            # So, if OLD_HNSW_INDEX_NAME was set and drop was attempted and succeeded within execute_sql (no exception),
            # we need a commit here.
            if OLD_HNSW_INDEX_NAME and OLD_HNSW_INDEX_NAME.strip(): # Re-check if an attempt was made
                 # If execute_sql for DROP didn't raise, it means it was accepted by DB (though might warn if not found)
                 # We need to ensure the transaction is committed if the DDL was sent.
                 # The `execute_sql` itself does not commit.
                 pass # The commit is handled after each logical step in the main flow.

            # The main script commits after each major step. If drop was attempted, it's part of this step.
            # The commit for step 4 will happen after this block.
            # The try-except around execute_sql for DROP INDEX handles its specific failure.
            conn.commit() # Commit changes for Step 4 (or lack thereof if skipped)


            logging.info(f"--- Step 5: Drop old VARCHAR column '{OLD_VARCHAR_COLUMN_NAME}' ---")
            # Check if column exists before dropping
            cursor.execute(f"""
                SELECT 1 
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{SCHEMA_NAME}' 
                AND TABLE_NAME = '{TABLE_NAME}' 
                AND COLUMN_NAME = '{OLD_VARCHAR_COLUMN_NAME}'
            """)
            if cursor.fetchone():
                sql_drop_old_column = f"ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} DROP COLUMN {OLD_VARCHAR_COLUMN_NAME}"
                execute_sql(cursor, sql_drop_old_column, DDL=True)
            else:
                logging.info(f"Column {OLD_VARCHAR_COLUMN_NAME} not found in {SCHEMA_NAME}.{TABLE_NAME}. Skipping drop.")
            conn.commit()

            logging.info(f"--- Step 6: Rename new VECTOR column '{TEMP_VECTOR_COLUMN_NAME}' to '{FINAL_VECTOR_COLUMN_NAME}' ---")
            # Check if temp column exists and final column does not (or is the same)
            cursor.execute(f"""
                SELECT 1 
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{SCHEMA_NAME}' 
                AND TABLE_NAME = '{TABLE_NAME}' 
                AND COLUMN_NAME = '{TEMP_VECTOR_COLUMN_NAME}'
            """)
            temp_col_exists = cursor.fetchone()

            cursor.execute(f"""
                SELECT 1 
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{SCHEMA_NAME}' 
                AND TABLE_NAME = '{TABLE_NAME}' 
                AND COLUMN_NAME = '{FINAL_VECTOR_COLUMN_NAME}'
            """)
            final_col_exists = cursor.fetchone()

            if temp_col_exists and not final_col_exists:
                # IRIS syntax for renaming a column:
                sql_rename_column = f"ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} ALTER ({TEMP_VECTOR_COLUMN_NAME} NAME {FINAL_VECTOR_COLUMN_NAME})"
                execute_sql(cursor, sql_rename_column, DDL=True)
            elif temp_col_exists and final_col_exists and TEMP_VECTOR_COLUMN_NAME == FINAL_VECTOR_COLUMN_NAME:
                 logging.info(f"Column is already named '{FINAL_VECTOR_COLUMN_NAME}'. Skipping rename.")
            elif not temp_col_exists:
                 logging.warning(f"Temporary column {TEMP_VECTOR_COLUMN_NAME} not found. Cannot rename. Check previous steps.")
            elif final_col_exists and TEMP_VECTOR_COLUMN_NAME != FINAL_VECTOR_COLUMN_NAME:
                 logging.warning(f"Final column {FINAL_VECTOR_COLUMN_NAME} already exists and is different from temp column. Skipping rename to avoid conflict.")
            conn.commit()

            logging.info(f"--- Step 7: Create new HNSW index '{NEW_HNSW_INDEX_NAME}' on native VECTOR column '{FINAL_VECTOR_COLUMN_NAME}' ---")
            # Check if index already exists
            cursor.execute(f"""
                SELECT IndexName from %dictionary.IndexDefinition
                WHERE TableName = '{SCHEMA_NAME}.{TABLE_NAME}' AND IndexName = '{NEW_HNSW_INDEX_NAME}'
            """)
            if cursor.fetchone():
                logging.info(f"Index {NEW_HNSW_INDEX_NAME} already exists on {SCHEMA_NAME}.{TABLE_NAME}. Skipping creation.")
            else:
                # Using CREATE INDEX ... AS HNSW syntax, similar to db_init_complete.sql
                sql_create_new_index = f"""
                CREATE INDEX {NEW_HNSW_INDEX_NAME}
                ON {SCHEMA_NAME}.{TABLE_NAME}({FINAL_VECTOR_COLUMN_NAME})
                AS HNSW({HNSW_INDEX_PARAMS})
                """
                execute_sql(cursor, sql_create_new_index, DDL=True)
            conn.commit()

            logging.info("--- Step 8: Testing performance and functionality (Manual Step Reminder) ---")
            logging.info("Migration script has completed the schema changes and data movement.")
            logging.info(f"Please now manually test your RAG pipelines and query performance with the new native VECTOR column '{FINAL_VECTOR_COLUMN_NAME}'.")
            logging.info(f"Ensure HNSW index '{NEW_HNSW_INDEX_NAME}' is active and providing good performance (sub-100ms queries).")
            logging.info("Remember to update your application code to use the new column name if it changed, and remove any TO_VECTOR() calls on this column in queries.")

            logging.info(f"Migration for {SCHEMA_NAME}.{TABLE_NAME} to native VECTOR column completed successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the migration: {e}")
        if conn:
            try:
                conn.rollback()
                logging.info("Database transaction rolled back.")
            except Exception as rb_e:
                logging.error(f"Error during rollback: {rb_e}")
        return 1 # Indicate failure
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")
    
    return 0 # Indicate success

if __name__ == "__main__":
    logging.info("Starting RAG.SourceDocuments VECTOR migration script.")
    logging.warning("IMPORTANT: Review and update placeholder configurations at the top of this script (index names, HNSW params) before running.")
    logging.warning("IMPORTANT: Ensure you have a database backup before proceeding.")
    # Add a small delay with a prompt for final confirmation if run directly,
    # or expect it to be run in a controlled environment.
    # For now, direct execution.
    
    # Example:
    # confirm = input("Have you reviewed configurations and backed up your DB? (yes/no): ")
    # if confirm.lower() != 'yes':
    #     logging.info("Migration aborted by user.")
    #     sys.exit(1)

    exit_code = main_migration()
    if exit_code == 0:
        logging.info("Migration script finished successfully.")
    else:
        logging.error("Migration script encountered errors.")
    sys.exit(exit_code)