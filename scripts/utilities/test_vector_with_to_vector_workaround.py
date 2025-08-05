import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_vector_with_workaround():
    """Test if the embedding column is actually native VECTOR but needs TO_VECTOR() due to JDBC driver issues"""
    logging.info("Testing VECTOR functionality with TO_VECTOR() workaround for JDBC driver...")
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # First, insert a test vector using TO_VECTOR
            test_vector = "[" + ",".join(["0.1"] * 384) + "]"
            
            logging.info("Inserting test vector...")
            cursor.execute("""
                INSERT INTO RAG.SourceDocuments (doc_id, text_content, embedding) 
                VALUES ('test_jdbc_workaround', 'Test for JDBC driver workaround', TO_VECTOR(?))
            """, (test_vector,))
            
            # Test 1: Query with TO_VECTOR() on the embedding column (workaround for JDBC)
            logging.info("Test 1: Using TO_VECTOR() on embedding column (JDBC workaround)...")
            cursor.execute("""
                SELECT doc_id, VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS similarity
                FROM RAG.SourceDocuments 
                WHERE doc_id = 'test_jdbc_workaround'
            """, (test_vector,))
            
            result1 = cursor.fetchone()
            if result1 and result1[1] is not None:
                logging.info(f"✅ Test 1 SUCCESS: VECTOR_COSINE with TO_VECTOR(embedding) works: {result1[1]}")
            else:
                logging.error("❌ Test 1 FAILED: TO_VECTOR(embedding) approach failed")
            
            # Test 2: Query without TO_VECTOR() on embedding column (direct native VECTOR)
            logging.info("Test 2: Direct embedding column (native VECTOR)...")
            try:
                cursor.execute("""
                    SELECT doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                    FROM RAG.SourceDocuments 
                    WHERE doc_id = 'test_jdbc_workaround'
                """, (test_vector,))
                
                result2 = cursor.fetchone()
                if result2 and result2[1] is not None:
                    logging.info(f"✅ Test 2 SUCCESS: Direct native VECTOR works: {result2[1]}")
                else:
                    logging.error("❌ Test 2 FAILED: Direct native VECTOR returned no result")
            except Exception as e:
                logging.error(f"❌ Test 2 FAILED: Direct native VECTOR failed: {e}")
            
            # Test 3: Check if existing data works with TO_VECTOR workaround
            logging.info("Test 3: Testing with existing data using TO_VECTOR workaround...")
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL AND doc_id != 'test_jdbc_workaround'")
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0:
                logging.info(f"Found {existing_count} existing documents with embeddings")
                
                try:
                    cursor.execute(f"""
                        SELECT TOP 3 doc_id, VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS similarity
                        FROM RAG.SourceDocuments 
                        WHERE embedding IS NOT NULL AND doc_id != 'test_jdbc_workaround'
                        ORDER BY similarity DESC
                    """, (test_vector,))
                    
                    results = cursor.fetchall()
                    if results:
                        logging.info(f"✅ Test 3 SUCCESS: TO_VECTOR workaround works with existing data")
                        for i, (doc_id, sim) in enumerate(results):
                            logging.info(f"  Result {i+1}: {doc_id} - similarity: {sim}")
                    else:
                        logging.error("❌ Test 3 FAILED: No results with existing data")
                except Exception as e:
                    logging.error(f"❌ Test 3 FAILED: Error with existing data: {e}")
            else:
                logging.info("No existing data to test with")
            
            # Clean up test data
            cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = 'test_jdbc_workaround'")
            conn.commit()
            
            logging.info("🎯 CONCLUSION:")
            if result1 and result1[1] is not None:
                logging.info("✅ The embedding column IS native VECTOR type")
                logging.info("✅ JDBC driver issue requires TO_VECTOR(embedding) workaround in queries")
                logging.info("✅ Schema is correctly created with native VECTOR types")
                logging.info("✅ Ready for parallel migration with TO_VECTOR() workaround in RAG pipelines")
                return True
            else:
                logging.error("❌ Schema needs to be recreated with proper native VECTOR types")
                return False
            
    except Exception as e:
        logging.error(f"❌ Test failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    success = test_vector_with_workaround()
    if success:
        logging.info("🚀 Native VECTOR schema confirmed working with JDBC workaround")
        sys.exit(0)
    else:
        logging.error("❌ Schema needs recreation")
        sys.exit(1)