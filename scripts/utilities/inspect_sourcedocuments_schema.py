import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCHEMA_NAME = "RAG"
TABLE_NAME = "SourceDocuments"
VARCHAR_COLUMN_NAME = "document_embedding_vector" # The old VARCHAR column

def inspect_schema():
    logging.info(f"Inspecting schema for {SCHEMA_NAME}.{TABLE_NAME}...")
    conn = None
    
    try:
        conn = get_iris_connection()
        with conn.cursor() as cursor:
            # 1. Get current row count for %ELEMENTS
            logging.info(f"\n--- Querying row count for {SCHEMA_NAME}.{TABLE_NAME} ---")
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{TABLE_NAME}")
                row_count = cursor.fetchone()[0]
                logging.info(f"Total rows in {SCHEMA_NAME}.{TABLE_NAME}: {row_count}")
                print(f"SUGGESTED %ELEMENTS: {row_count}")
            except Exception as e:
                logging.error(f"Error getting row count: {e}")
                print(f"SUGGESTED %ELEMENTS: (Error fetching count)")

            # 2. List indexes on the table and try to identify one on the VARCHAR column
            logging.info(f"\n--- Querying indexes for {SCHEMA_NAME}.{TABLE_NAME} ---")
            # %dictionary.IndexDefinition stores index metadata.
            # Selecting all columns and then inspecting description to be more robust.
            sql_get_indexes = f"""
            SELECT *
            FROM %dictionary.IndexDefinition
            WHERE TableName = '{SCHEMA_NAME}.{TABLE_NAME}'
            """
            try:
                cursor.execute(sql_get_indexes)
                indexes = cursor.fetchall()
                
                if not indexes:
                    logging.info("No indexes found for this table in %dictionary.IndexDefinition.")
                    print(f"OLD_HNSW_INDEX_NAME: (No indexes found for table)")
                else:
                    # Get column names from cursor.description
                    column_names = [desc[0] for desc in cursor.description]
                    logging.info(f"Retrieved columns from %dictionary.IndexDefinition: {column_names}")
                    logging.info("Found indexes (raw data):")
                    
                    # Try to find common names for index name, type, and properties/data
                    # These are guesses; the raw printout will be most reliable.
                    name_col_idx = column_names.index('Name') if 'Name' in column_names else -1
                    type_col_idx = column_names.index('Type') if 'Type' in column_names else \
                                   (column_names.index('IndexType') if 'IndexType' in column_names else -1)
                    props_col_idx = column_names.index('Properties') if 'Properties' in column_names else \
                                    (column_names.index('Data') if 'Data' in column_names else -1)

                    found_varchar_hnsw_index = None
                    for i, index_row_tuple in enumerate(indexes):
                        logging.info(f"  Index #{i+1}:")
                        row_dict = {}
                        for col_idx, col_name in enumerate(column_names):
                            logging.info(f"    {col_name}: {index_row_tuple[col_idx]}")
                            row_dict[col_name] = index_row_tuple[col_idx]
                        
                        # Attempt to parse with guessed column names
                        index_name_val = row_dict.get('Name', str(row_dict)) # Default to full row if 'Name' not found
                        index_type_val = row_dict.get('Type') or row_dict.get('IndexType')
                        index_props_val = row_dict.get('Properties') or row_dict.get('Data')

                        if index_props_val and VARCHAR_COLUMN_NAME in str(index_props_val):
                            logging.info(f"    -> Potential match for an index on '{VARCHAR_COLUMN_NAME}' (Name: {index_name_val}).")
                            if index_type_val and ("hnsw" in str(index_type_val).lower() or "vector" in str(index_type_val).lower()):
                                found_varchar_hnsw_index = index_name_val
                                logging.info(f"    -> This appears to be an HNSW-like index on the VARCHAR column: {found_varchar_hnsw_index}")
                            elif not found_varchar_hnsw_index:
                                logging.info(f"    -> This is a non-HNSW index on the VARCHAR column: {index_name_val}")
                    
                    if found_varchar_hnsw_index:
                        print(f"OLD_HNSW_INDEX_NAME: {found_varchar_hnsw_index} (Found HNSW-like index on {VARCHAR_COLUMN_NAME})")
                    else:
                        print(f"OLD_HNSW_INDEX_NAME: (No HNSW index automatically identified on '{VARCHAR_COLUMN_NAME}'. Review raw data above.)")
                        logging.info(f"No index explicitly identified as HNSW on '{VARCHAR_COLUMN_NAME}'. "
                                     "Please review the full raw data for each index above. If an old HNSW index exists on this column, "
                                     "identify it manually for the migration script. If not, the drop step for it can be skipped.")
            except Exception as e:
                logging.error(f"Error querying or processing indexes: {e}")
                print(f"OLD_HNSW_INDEX_NAME: (Error fetching or processing indexes)")

            logging.info("\n--- Suggested NEW_HNSW_INDEX_NAME ---")
            # Suggest a name based on conventions seen or a default
            suggested_new_name = f"idx_hnsw_{TABLE_NAME.lower()}_embedding_128d" # Or similar to existing patterns
            logging.info(f"Consider a name like: {suggested_new_name} or idx_hnsw_{TABLE_NAME.lower()}_vector")
            print(f"SUGGESTED NEW_HNSW_INDEX_NAME: {suggested_new_name}")

            logging.info("\nInspection complete. Use the printed 'SUGGESTED' values to update your migration script.")

    except Exception as e:
        logging.critical(f"A critical error occurred during schema inspection: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    inspect_schema()