#!/usr/bin/env python3
"""
Add simple performance indexes for IRIS.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # Go up two levels to project root

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_simple_indexes():
    """Add simple indexes for IRIS"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        logger.info("🚀 Adding simple performance indexes...")
        
        # 1. Try to add index on doc_id
        try:
            cursor.execute("CREATE INDEX idx_doc_id ON RAG.SourceDocuments (doc_id)")
            logger.info("✅ Added doc_id index")
        except Exception as e:
            if "already exists" in str(e) or "duplicate" in str(e).lower():
                logger.info("✅ Doc_id index already exists")
            else:
                logger.warning(f"Doc_id index failed: {e}")
        
        # 2. Check current performance
        logger.info("🧪 Testing current query performance...")
        import time
        
        # Test simple count query
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        result = cursor.fetchone()
        query_time = time.time() - start_time
        
        logger.info(f"✅ Count query: {query_time:.3f}s ({result[0]:,} docs)")
        
        # Test filtered count query
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL AND LENGTH(embedding) > 1000")
        result = cursor.fetchone()
        query_time = time.time() - start_time
        
        logger.info(f"✅ Filtered count: {query_time:.3f}s ({result[0]:,} real embeddings)")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    add_simple_indexes()