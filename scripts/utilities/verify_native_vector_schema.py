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
            
            # Check SourceDocuments table exists
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'
            """)
            
            if not cursor.fetchone()[0]:
                logging.error("SourceDocuments table not found!")
                return False
            
            logging.info("✅ SourceDocuments table exists")
            
            # IMPORTANT: JDBC driver cannot properly show VECTOR types in INFORMATION_SCHEMA
            # They appear as VARCHAR even when they are actually native VECTOR columns
            # So we need to test VECTOR functionality directly instead of relying on schema inspection
            
            logging.info("⚠️  Note: JDBC driver shows VECTOR columns as VARCHAR in INFORMATION_SCHEMA")
            logging.info("Testing native VECTOR functionality directly...")
            
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
            
            # Test native VECTOR operations directly
            logging.info("Testing native VECTOR operations...")
            
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
            
            # Test inserting into native VECTOR column
            logging.info("Testing native VECTOR column insertion...")
            test_vector_384 = "[" + ",".join(["0.1"] * 384) + "]"
            
            try:
                # Try to insert a test document with native VECTOR
                cursor.execute("""
                    INSERT INTO RAG.SourceDocuments (doc_id, text_content, embedding)
                    VALUES ('test_vector_insert', 'Test document for vector verification', TO_VECTOR(?))
                """, (test_vector_384,))
                
                # Try to query it back
                cursor.execute("""
                    SELECT doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                    FROM RAG.SourceDocuments
                    WHERE doc_id = 'test_vector_insert'
                """, (test_vector_384,))
                
                result = cursor.fetchone()
                if result and result[1] is not None:
                    logging.info(f"✅ Native VECTOR column insert/query works: similarity = {result[1]}")
                    
                    # Clean up test data
                    cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = 'test_vector_insert'")
                else:
                    logging.error("❌ Native VECTOR column query failed")
                    return False
                    
            except Exception as e:
                logging.error(f"❌ Native VECTOR column test failed: {e}")
                return False
            
            # Test HNSW index functionality (if data exists)
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            vector_count = cursor.fetchone()[0]
            
            if vector_count > 0:
                logging.info(f"Found {vector_count} documents with embeddings")
                
                # Test HNSW performance
                import time
                start_time = time.time()
                cursor.execute(f"""
                    SELECT TOP 5 doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                """, (test_vector_384,))
                
                results = cursor.fetchall()
                end_time = time.time()
                query_time_ms = (end_time - start_time) * 1000
                
                if results:
                    logging.info(f"✅ HNSW vector search works: {len(results)} results in {query_time_ms:.1f}ms")
                    if query_time_ms < 100:
                        logging.info("🚀 Excellent performance: <100ms")
                    elif query_time_ms < 500:
                        logging.info("✅ Good performance: <500ms")
                    else:
                        logging.warning(f"⚠️  Slow performance: {query_time_ms:.1f}ms")
                else:
                    logging.warning("⚠️  HNSW search returned no results")
            else:
                logging.info("No existing vector data found - HNSW test skipped")
            
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