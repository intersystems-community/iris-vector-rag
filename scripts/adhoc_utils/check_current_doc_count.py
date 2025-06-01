import sys
import logging
sys.path.append('.')

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_current_documents():
    """Check how many documents are currently in the database"""
    logger.info("Connecting to IRIS to check document counts...")
    iris = get_iris_connection()
    if not iris:
        logger.error("Failed to connect to IRIS.")
        return 0, 0
        
    cursor = iris.cursor()
    
    doc_count = 0
    unique_count = 0
    entity_count = 0
    rel_count = 0
    
    try:
        # Check SourceDocuments
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count_result = cursor.fetchone()
        if doc_count_result:
            doc_count = doc_count_result[0]
        
        # Check unique Document IDs
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.SourceDocuments WHERE doc_id IS NOT NULL AND doc_id <> ''")
        unique_count_result = cursor.fetchone()
        if unique_count_result:
            unique_count = unique_count_result[0]
        
        # Check GraphRAG data (handle if tables don't exist)
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            entity_count_result = cursor.fetchone()
            if entity_count_result:
                entity_count = entity_count_result[0]
        except Exception:
            logger.warning("RAG.Entities table not found or error querying.")
            entity_count = 0
            
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
            rel_count_result = cursor.fetchone()
            if rel_count_result:
                rel_count = rel_count_result[0]
        except Exception:
            logger.warning("RAG.Relationships table not found or error querying.")
            rel_count = 0
            
        logger.info(f"Current database state:")
        logger.info(f"  Total rows in RAG.SourceDocuments: {doc_count:,}")
        logger.info(f"  Unique non-empty Document IDs in RAG.SourceDocuments: {unique_count:,}")
        logger.info(f"  GraphRAG entities: {entity_count:,}")
        logger.info(f"  GraphRAG relationships: {rel_count:,}")
        
        return doc_count, unique_count
        
    except Exception as e:
        logger.error(f"Error checking document counts: {e}")
        return 0,0
    finally:
        if 'iris' in locals() and iris:
            cursor.close()
            iris.close()

if __name__ == "__main__":
    check_current_documents()