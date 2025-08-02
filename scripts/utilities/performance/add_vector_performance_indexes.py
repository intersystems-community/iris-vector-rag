#!/usr/bin/env python3
"""
Add performance indexes for vector operations to speed up RAG queries.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # Go up two levels to project root

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_performance_indexes():
    """Add indexes to improve vector search performance"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        logger.info("🚀 Adding performance indexes for vector operations...")
        
        # 1. Add index on embedding column for faster vector operations
        logger.info("📊 Adding index on embedding column...")
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_length 
                ON RAG.SourceDocuments (LENGTH(embedding))
            """)
            logger.info("✅ Added embedding length index")
        except Exception as e:
            logger.warning(f"Embedding length index: {e}")
        
        # 2. Add index on doc_id for faster lookups
        logger.info("📊 Adding index on doc_id...")
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_id 
                ON RAG.SourceDocuments (doc_id)
            """)
            logger.info("✅ Added doc_id index")
        except Exception as e:
            logger.warning(f"Doc_id index: {e}")
            
        # 3. Add composite index for common query patterns
        logger.info("📊 Adding composite index...")
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_not_null 
                ON RAG.SourceDocuments (doc_id) 
                WHERE embedding IS NOT NULL
            """)
            logger.info("✅ Added composite index")
        except Exception as e:
            logger.warning(f"Composite index: {e}")
            
        # 4. Update table statistics for query optimizer
        logger.info("📊 Updating table statistics...")
        try:
            cursor.execute("ANALYZE TABLE RAG.SourceDocuments")
            logger.info("✅ Updated table statistics")
        except Exception as e:
            logger.warning(f"Statistics update: {e}")
            
        logger.info("🎉 Performance optimization completed!")
        
        # Test query performance
        logger.info("🧪 Testing query performance...")
        import time
        start_time = time.time()
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL 
              AND LENGTH(embedding) > 1000
        """)
        
        result = cursor.fetchone()
        query_time = time.time() - start_time
        
        logger.info(f"✅ Query completed in {query_time:.3f}s")
        logger.info(f"📊 Found {result[0]:,} documents with real embeddings")
        
    except Exception as e:
        logger.error(f"❌ Error adding indexes: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    add_performance_indexes()