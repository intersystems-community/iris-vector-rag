import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_vector_udf():
    """Test the RAG.GetVectorAsStringFromVarchar function"""
    logging.info("Testing RAG.GetVectorAsStringFromVarchar function...")
    conn = None
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # First, get a sample vector value from the table
            sample_sql = """
            SELECT TOP 1 document_embedding_vector 
            FROM RAG.SourceDocuments 
            WHERE document_embedding_vector IS NOT NULL
            """
            
            logging.info("Getting sample vector value...")
            cursor.execute(sample_sql)
            result = cursor.fetchone()
            
            if not result:
                logging.error("No vector data found in RAG.SourceDocuments")
                return 1
            
            sample_vector = result[0]
            logging.info(f"Sample vector value: {sample_vector}")
            
            # Test the UDF
            test_sql = "SELECT RAG.GetVectorAsStringFromVarchar(?) AS ConvertedVector"
            
            logging.info("Testing UDF with sample vector...")
            cursor.execute(test_sql, (sample_vector,))
            udf_result = cursor.fetchone()
            
            if udf_result:
                converted_vector = udf_result[0]
                logging.info(f"UDF result: {converted_vector}")
                
                if converted_vector and converted_vector.strip():
                    # Test if TO_VECTOR can parse the result
                    to_vector_sql = "SELECT TO_VECTOR(?) AS ParsedVector"
                    try:
                        cursor.execute(to_vector_sql, (converted_vector,))
                        parsed_result = cursor.fetchone()
                        logging.info(f"TO_VECTOR parsing successful: {parsed_result[0] if parsed_result else 'None'}")
                        logging.info("✅ UDF test successful!")
                        return 0
                    except Exception as e:
                        logging.error(f"TO_VECTOR parsing failed: {e}")
                        return 1
                else:
                    logging.error("UDF returned empty or null result")
                    return 1
            else:
                logging.error("UDF returned no result")
                return 1
                
    except Exception as e:
        logging.error(f"Error testing UDF: {e}")
        return 1
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    exit_code = test_vector_udf()
    if exit_code == 0:
        logging.info("UDF test completed successfully.")
    else:
        logging.error("UDF test failed.")
    sys.exit(exit_code)