import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_vector_data():
    """Debug the vector data to understand what's happening"""
    logging.info("Debugging vector data in RAG.SourceDocuments...")
    conn = None
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Get multiple samples to see if there are different formats
            sample_sql = """
            SELECT TOP 5 
                doc_id,
                document_embedding_vector,
                LENGTH(document_embedding_vector) AS vector_length,
                SUBSTRING(document_embedding_vector, 1, 100) AS first_100_chars
            FROM RAG.SourceDocuments 
            WHERE document_embedding_vector IS NOT NULL
            ORDER BY doc_id
            """
            
            logging.info("Getting multiple sample vector values...")
            cursor.execute(sample_sql)
            results = cursor.fetchall()
            
            for i, result in enumerate(results):
                doc_id, vector_data, vector_length, first_100 = result
                logging.info(f"Sample {i+1}: doc_id={doc_id}, length={vector_length}")
                logging.info(f"  First 100 chars: {first_100}")
                
                # Test TO_VECTOR with brackets on each sample
                try:
                    test_sql = "SELECT TO_VECTOR('[' || ? || ']') AS converted"
                    cursor.execute(test_sql, (vector_data,))
                    converted_result = cursor.fetchone()
                    logging.info(f"  TO_VECTOR result: {converted_result[0] if converted_result else 'None'}")
                except Exception as e:
                    logging.error(f"  TO_VECTOR failed: {e}")
                
                # Test direct TO_VECTOR without brackets
                try:
                    test_sql2 = "SELECT TO_VECTOR(?) AS converted_direct"
                    cursor.execute(test_sql2, (vector_data,))
                    converted_result2 = cursor.fetchone()
                    logging.info(f"  TO_VECTOR direct result: {converted_result2[0] if converted_result2 else 'None'}")
                except Exception as e:
                    logging.error(f"  TO_VECTOR direct failed: {e}")
                
                logging.info("---")
            
            # Check if there are any rows where the UPDATE would work
            logging.info("Testing UPDATE on a single row...")
            try:
                update_test_sql = """
                UPDATE RAG.SourceDocuments
                SET embedding_vector_new = TO_VECTOR('[' || document_embedding_vector || ']')
                WHERE doc_id = (SELECT TOP 1 doc_id FROM RAG.SourceDocuments WHERE document_embedding_vector IS NOT NULL)
                AND embedding_vector_new IS NULL
                """
                cursor.execute(update_test_sql)
                logging.info(f"Single row update successful. Rows affected: {cursor.rowcount}")
                conn.rollback()  # Don't commit the test
            except Exception as e:
                logging.error(f"Single row update failed: {e}")
                conn.rollback()
                
    except Exception as e:
        logging.error(f"Error debugging vector data: {e}")
        return 1
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    debug_vector_data()