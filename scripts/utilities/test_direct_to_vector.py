import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_direct_to_vector():
    """Test if TO_VECTOR works directly on the existing data"""
    logging.info("Testing TO_VECTOR directly on existing vector data...")
    conn = None
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Test TO_VECTOR directly on the existing data
            test_sql = """
            SELECT TOP 1 
                document_embedding_vector,
                TO_VECTOR('[' || document_embedding_vector || ']') AS converted_vector
            FROM RAG.SourceDocuments 
            WHERE document_embedding_vector IS NOT NULL
            """
            
            logging.info("Testing TO_VECTOR with bracket wrapping...")
            cursor.execute(test_sql)
            result = cursor.fetchone()
            
            if result:
                original = result[0]
                converted = result[1]
                logging.info(f"Original (first 100 chars): {original[:100]}...")
                logging.info(f"Converted: {converted}")
                logging.info("✅ TO_VECTOR with brackets works!")
                return 0
            else:
                logging.error("No data returned")
                return 1
                
    except Exception as e:
        logging.error(f"Error testing TO_VECTOR: {e}")
        
        # Try without brackets
        try:
            logging.info("Trying TO_VECTOR without brackets...")
            test_sql2 = """
            SELECT TOP 1 
                TO_VECTOR(document_embedding_vector) AS converted_vector
            FROM RAG.SourceDocuments 
            WHERE document_embedding_vector IS NOT NULL
            """
            cursor.execute(test_sql2)
            result = cursor.fetchone()
            
            if result:
                converted = result[0]
                logging.info(f"Converted without brackets: {converted}")
                logging.info("✅ TO_VECTOR without brackets works!")
                return 0
            else:
                logging.error("No data returned")
                return 1
                
        except Exception as e2:
            logging.error(f"TO_VECTOR without brackets also failed: {e2}")
            return 1
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    exit_code = test_direct_to_vector()
    if exit_code == 0:
        logging.info("Direct TO_VECTOR test completed successfully.")
    else:
        logging.error("Direct TO_VECTOR test failed.")
    sys.exit(exit_code)