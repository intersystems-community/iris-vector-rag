#!/usr/bin/env python3
"""
Add indexes to speed up graph ingestion process.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_graph_ingestion_indexes():
    """Add indexes to speed up graph ingestion"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        logger.info("🚀 Adding indexes to speed up graph ingestion...")
        
        # 1. Index on SourceDocuments for faster batch processing
        logger.info("📊 Adding index on SourceDocuments.doc_id...")
        try:
            cursor.execute("CREATE INDEX idx_source_docs_id ON RAG.SourceDocuments (doc_id)")
            logger.info("✅ Added SourceDocuments doc_id index")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("✅ SourceDocuments doc_id index already exists")
            else:
                logger.warning(f"SourceDocuments doc_id index: {e}")
        
        # 2. Index on SourceDocuments text_content for faster filtering
        logger.info("📊 Adding index on SourceDocuments for text filtering...")
        try:
            cursor.execute("CREATE INDEX idx_source_docs_text_not_null ON RAG.SourceDocuments (doc_id) WHERE text_content IS NOT NULL")
            logger.info("✅ Added SourceDocuments text filtering index")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("✅ SourceDocuments text filtering index already exists")
            else:
                logger.warning(f"SourceDocuments text filtering index: {e}")
        
        # 3. Index on Entities for faster duplicate checking
        logger.info("📊 Adding index on Entities.entity_id...")
        try:
            cursor.execute("CREATE INDEX idx_entities_id ON RAG.Entities (entity_id)")
            logger.info("✅ Added Entities entity_id index")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("✅ Entities entity_id index already exists")
            else:
                logger.warning(f"Entities entity_id index: {e}")
        
        # 4. Index on Entities source_doc_id for faster lookups
        logger.info("📊 Adding index on Entities.source_doc_id...")
        try:
            cursor.execute("CREATE INDEX idx_entities_source_doc ON RAG.Entities (source_doc_id)")
            logger.info("✅ Added Entities source_doc_id index")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("✅ Entities source_doc_id index already exists")
            else:
                logger.warning(f"Entities source_doc_id index: {e}")
        
        # 5. Index on Relationships for faster duplicate checking
        logger.info("📊 Adding index on Relationships.relationship_id...")
        try:
            cursor.execute("CREATE INDEX idx_relationships_id ON RAG.Relationships (relationship_id)")
            logger.info("✅ Added Relationships relationship_id index")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("✅ Relationships relationship_id index already exists")
            else:
                logger.warning(f"Relationships relationship_id index: {e}")
        
        # 6. Composite index on Relationships for foreign key lookups
        logger.info("📊 Adding composite index on Relationships...")
        try:
            cursor.execute("CREATE INDEX idx_relationships_entities ON RAG.Relationships (source_entity_id, target_entity_id)")
            logger.info("✅ Added Relationships composite index")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("✅ Relationships composite index already exists")
            else:
                logger.warning(f"Relationships composite index: {e}")
        
        # 7. Update table statistics for better query planning
        logger.info("📊 Updating table statistics...")
        tables_to_analyze = ['SourceDocuments', 'Entities', 'Relationships']
        for table in tables_to_analyze:
            try:
                # IRIS uses different syntax for updating statistics
                cursor.execute(f"SELECT COUNT(*) FROM RAG.{table}")
                count = cursor.fetchone()[0]
                logger.info(f"✅ RAG.{table}: {count:,} rows")
            except Exception as e:
                logger.warning(f"Statistics for {table}: {e}")
        
        logger.info("🎉 Graph ingestion indexes completed!")
        logger.info("⚡ Ingestion should now be significantly faster!")
        
    except Exception as e:
        logger.error(f"❌ Error adding indexes: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    add_graph_ingestion_indexes()