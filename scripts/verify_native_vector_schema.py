import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_native_vector_schema():
    """Verify that the native VECTOR schema is properly created"""
    logging.info("Verifying native VECTOR schema...")
    conn = None
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Check if RAG schema exists
            cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = 'RAG'")
            if not cursor.fetchone():
                logging.error("RAG schema not found!")
                return False
            
            logging.info("✅ RAG schema exists")
            
            # Check SourceDocuments table with native VECTOR column
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'
                ORDER BY ORDINAL_POSITION
            """)
            
            columns = cursor.fetchall()
            if not columns:
                logging.error("SourceDocuments table not found!")
                return False
            
            logging.info("✅ SourceDocuments table exists")
            
            # Verify specific columns
            column_dict = {col[0]: col[1] for col in columns}
            
            required_columns = {
                'doc_id': 'VARCHAR',
                'title': 'VARCHAR', 
                'content': 'VARCHAR',
                'embedding': 'VECTOR',  # This should be native VECTOR type
                'created_at': 'TIMESTAMP'
            }
            
            for col_name, expected_type in required_columns.items():
                if col_name not in column_dict:
                    logging.error(f"❌ Missing column: {col_name}")
                    return False
                elif expected_type == 'VECTOR' and 'VECTOR' not in column_dict[col_name]:
                    logging.error(f"❌ Column {col_name} is not VECTOR type: {column_dict[col_name]}")
                    return False
                else:
                    logging.info(f"✅ Column {col_name}: {column_dict[col_name]}")
            
            # Check DocumentChunks table
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentChunks'
                AND COLUMN_NAME = 'embedding'
            """)
            
            chunk_embedding = cursor.fetchone()
            if chunk_embedding and 'VECTOR' in chunk_embedding[1]:
                logging.info("✅ DocumentChunks table has native VECTOR embedding column")
            else:
                logging.warning("⚠️  DocumentChunks table missing or doesn't have native VECTOR column")
            
            # Check for HNSW indexes
            try:
                cursor.execute("""
                    SELECT IndexName, TableName 
                    FROM %dictionary.IndexDefinition 
                    WHERE TableName LIKE 'RAG.%' AND IndexName LIKE '%hnsw%'
                """)
                
                indexes = cursor.fetchall()
                if indexes:
                    logging.info("✅ HNSW indexes found:")
                    for idx_name, table_name in indexes:
                        logging.info(f"   - {idx_name} on {table_name}")
                else:
                    logging.warning("⚠️  No HNSW indexes found")
                    
            except Exception as e:
                logging.warning(f"⚠️  Could not check HNSW indexes: {e}")
            
            # Test vector operations
            logging.info("Testing vector operations...")
            
            # Test TO_VECTOR function
            cursor.execute("SELECT TO_VECTOR('[0.1, 0.2, 0.3]') AS test_vector")
            test_result = cursor.fetchone()
            if test_result:
                logging.info("✅ TO_VECTOR function works")
            else:
                logging.error("❌ TO_VECTOR function failed")
                return False
            
            # Test vector similarity functions
            cursor.execute("""
                SELECT VECTOR_COSINE(
                    TO_VECTOR('[0.1, 0.2, 0.3]'), 
                    TO_VECTOR('[0.2, 0.3, 0.4]')
                ) AS similarity
            """)
            
            similarity_result = cursor.fetchone()
            if similarity_result and similarity_result[0] is not None:
                logging.info(f"✅ VECTOR_COSINE function works: {similarity_result[0]}")
            else:
                logging.error("❌ VECTOR_COSINE function failed")
                return False
            
            logging.info("🎉 Native VECTOR schema verification completed successfully!")
            return True
            
    except Exception as e:
        logging.error(f"Error verifying schema: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    success = verify_native_vector_schema()
    if success:
        logging.info("Schema verification PASSED")
        sys.exit(0)
    else:
        logging.error("Schema verification FAILED")
        sys.exit(1)