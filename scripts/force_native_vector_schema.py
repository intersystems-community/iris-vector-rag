import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def force_native_vector_schema():
    """Force complete recreation of schema with native VECTOR types"""
    logging.info("🔥 Force recreating schema with native VECTOR types...")
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Step 1: Drop all existing tables completely
            logging.info("--- Step 1: Dropping all existing RAG tables ---")
            
            tables_to_drop = [
                "RAG.DocumentChunks",
                "RAG.SourceDocuments", 
                "RAG.Entities",
                "RAG.Relationships",
                "RAG.Communities"
            ]
            
            for table in tables_to_drop:
                try:
                    cursor.execute(f"DROP TABLE {table} CASCADE")
                    logging.info(f"✅ Dropped {table}")
                except Exception as e:
                    logging.info(f"⚠️  {table} not found or already dropped: {e}")
            
            # Step 2: Create SourceDocuments with native VECTOR
            logging.info("--- Step 2: Creating SourceDocuments with native VECTOR ---")
            
            create_source_docs = """
            CREATE TABLE RAG.SourceDocuments (
                doc_id VARCHAR(255) PRIMARY KEY,
                text_content TEXT,
                embedding VECTOR(DOUBLE, 384),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            cursor.execute(create_source_docs)
            logging.info("✅ Created SourceDocuments with native VECTOR(DOUBLE, 384)")
            
            # Step 3: Create DocumentChunks with native VECTOR
            logging.info("--- Step 3: Creating DocumentChunks with native VECTOR ---")
            
            create_chunks = """
            CREATE TABLE RAG.DocumentChunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                doc_id VARCHAR(255),
                chunk_text TEXT,
                chunk_embedding VECTOR(DOUBLE, 384),
                chunk_index INTEGER,
                chunk_type VARCHAR(100),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
            )
            """
            
            cursor.execute(create_chunks)
            logging.info("✅ Created DocumentChunks with native VECTOR(DOUBLE, 384)")
            
            # Step 4: Create indexes
            logging.info("--- Step 4: Creating indexes ---")
            
            indexes = [
                "CREATE INDEX idx_source_docs_id ON RAG.SourceDocuments (doc_id)",
                "CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks (doc_id)",
                "CREATE INDEX idx_chunks_type ON RAG.DocumentChunks (chunk_type)"
            ]
            
            for idx_sql in indexes:
                try:
                    cursor.execute(idx_sql)
                    logging.info(f"✅ Created index")
                except Exception as e:
                    logging.warning(f"⚠️  Index creation issue: {e}")
            
            # Step 5: Create HNSW indexes
            logging.info("--- Step 5: Creating HNSW indexes ---")
            
            hnsw_indexes = [
                "CREATE INDEX idx_hnsw_source_embedding ON RAG.SourceDocuments (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
                "CREATE INDEX idx_hnsw_chunk_embedding ON RAG.DocumentChunks (chunk_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')"
            ]
            
            for hnsw_sql in hnsw_indexes:
                try:
                    cursor.execute(hnsw_sql)
                    logging.info(f"✅ Created HNSW index")
                except Exception as e:
                    logging.warning(f"⚠️  HNSW index creation issue: {e}")
            
            # Step 6: Test native VECTOR functionality
            logging.info("--- Step 6: Testing native VECTOR functionality ---")
            
            # Test insert with native VECTOR
            test_vector = "[" + ",".join(["0.1"] * 384) + "]"
            
            cursor.execute("""
                INSERT INTO RAG.SourceDocuments (doc_id, text_content, embedding) 
                VALUES ('test_native_vector', 'Test document with native VECTOR', TO_VECTOR(?))
            """, (test_vector,))
            
            # Test query with native VECTOR
            cursor.execute("""
                SELECT doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                FROM RAG.SourceDocuments 
                WHERE doc_id = 'test_native_vector'
            """, (test_vector,))
            
            result = cursor.fetchone()
            if result and result[1] is not None:
                logging.info(f"✅ Native VECTOR test successful: similarity = {result[1]}")
                
                # Clean up test data
                cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = 'test_native_vector'")
            else:
                logging.error("❌ Native VECTOR test failed")
                return False
            
            conn.commit()
            
            logging.info("🎉 Native VECTOR schema created successfully!")
            logging.info("✅ Ready for data ingestion with native VECTOR types")
            
            return True
            
    except Exception as e:
        logging.error(f"❌ Force schema recreation failed: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    success = force_native_vector_schema()
    if success:
        logging.info("🚀 Native VECTOR schema force recreation successful")
        sys.exit(0)
    else:
        logging.error("❌ Native VECTOR schema force recreation failed")
        sys.exit(1)