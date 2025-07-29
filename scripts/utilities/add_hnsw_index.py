import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_hnsw_index():
    """
    Adds an HNSW index to the RAG.SourceDocuments table on the embedding column.
    """
    conn = None
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()

        # Check if the HNSW index already exists
        # This query might vary based on IRIS version and metadata tables
        # For simplicity, we'll just try to create it and handle errors
        logger.info("Attempting to create HNSW index on RAG.SourceDocuments.embedding...")
        
        # Assuming 'VECTOR_HNSW_INDEX' is the correct syntax for creating an HNSW index
        # and that IRIS can index TEXT columns containing vector strings.
        # The dimension (768) should match the embedding model used (e.g., all-MiniLM-L6-v2)
        create_index_sql = """
        CREATE INDEX idx_source_docs_embedding_hnsw
        ON RAG.SourceDocuments (embedding VECTOR_HNSW_INDEX (768))
        """
        
        cursor.execute(create_index_sql)
        conn.commit()
        logger.info("✅ HNSW index 'idx_source_docs_embedding_hnsw' created successfully (or already existed).")

    except Exception as e:
        logger.error(f"❌ Failed to create HNSW index: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    add_hnsw_index()