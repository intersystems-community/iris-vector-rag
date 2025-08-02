import sys
import logging
sys.path.append('.')
from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_table_columns(schema_name, table_name):
    logger.info(f"Connecting to IRIS to get columns for {schema_name}.{table_name}...")
    iris = get_iris_connection()
    if not iris:
        logger.error("Failed to connect to IRIS.")
        return []
        
    cursor = iris.cursor()
    columns = []
    try:
        # Standard SQL way to get columns
        # Note: JDBC metadata methods like getColumns() are often more robust
        # but this is a direct query approach.
        # For IRIS, INFORMATION_SCHEMA.COLUMNS is standard.
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        cursor.execute(query, (schema_name.upper(), table_name.upper()))
        
        logger.info(f"Columns for {schema_name}.{table_name}:")
        for row in cursor.fetchall():
            col_name, data_type, char_max_len = row
            logger.info(f"  - {col_name} (Type: {data_type}, MaxLen: {char_max_len})")
            columns.append(col_name)
        
        if not columns:
            logger.warning(f"No columns found for {schema_name}.{table_name}. Table might not exist or schema name is incorrect.")
            
    except Exception as e:
        logger.error(f"Error getting table columns for {schema_name}.{table_name}: {e}")
    finally:
        if 'iris' in locals() and iris:
            cursor.close()
            iris.close()
    return columns

if __name__ == "__main__":
    # Example usage:
    # python get_table_schema.py RAG SourceDocuments
    if len(sys.argv) == 3:
        schema = sys.argv[1]
        table = sys.argv[2]
        get_table_columns(schema, table)
    else:
        logger.info("Defaulting to RAG.SourceDocuments")
        get_table_columns("RAG", "SourceDocuments")