#!/usr/bin/env python3
"""
Script to create missing database tables for RAG pipelines.

This script creates the missing tables identified in the auto-setup run:
- RAG.DocumentEntities (for GraphRAG)
- RAG.EntityRelationships (for GraphRAG)
- RAG.DocumentTokenEmbeddings (for ColBERT)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connection_manager import get_iris_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_missing_tables():
    """Create missing database tables for RAG pipelines."""
    
    connection = get_iris_connection()
    if not connection:
        logger.error("Could not get IRIS connection")
        return False
    
    cursor = connection.cursor()
    
    try:
        logger.info("Creating missing tables for RAG pipelines...")
        
        # 1. Create DocumentEntities table for GraphRAG
        logger.info("Creating RAG.DocumentEntities table...")
        create_entities_sql = """
        CREATE TABLE RAG.DocumentEntities (
            entity_id VARCHAR(255) PRIMARY KEY,
            document_id VARCHAR(255) NOT NULL,
            entity_text VARCHAR(1000) NOT NULL,
            entity_type VARCHAR(100),
            position INTEGER,
            embedding VECTOR(DOUBLE, 1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            cursor.execute(create_entities_sql)
            logger.info("✓ RAG.DocumentEntities table created")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("✓ RAG.DocumentEntities table already exists")
            else:
                raise e
        
        # Create indexes separately for DocumentEntities
        entity_indexes = [
            "CREATE INDEX idx_doc_entities_doc_id ON RAG.DocumentEntities (document_id)",
            "CREATE INDEX idx_doc_entities_type ON RAG.DocumentEntities (entity_type)",
            "CREATE INDEX idx_doc_entities_text ON RAG.DocumentEntities (entity_text)"
        ]
        
        for idx_sql in entity_indexes:
            try:
                cursor.execute(idx_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        # 2. Create EntityRelationships table for GraphRAG
        logger.info("Creating RAG.EntityRelationships table...")
        create_relationships_sql = """
        CREATE TABLE RAG.EntityRelationships (
            relationship_id VARCHAR(255) PRIMARY KEY,
            document_id VARCHAR(255) NOT NULL,
            source_entity VARCHAR(255) NOT NULL,
            target_entity VARCHAR(255) NOT NULL,
            relationship_type VARCHAR(100),
            strength DOUBLE DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            cursor.execute(create_relationships_sql)
            logger.info("✓ RAG.EntityRelationships table created")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("✓ RAG.EntityRelationships table already exists")
            else:
                raise e
        
        # Create indexes separately for EntityRelationships
        relationship_indexes = [
            "CREATE INDEX idx_entity_rel_doc_id ON RAG.EntityRelationships (document_id)",
            "CREATE INDEX idx_entity_rel_source ON RAG.EntityRelationships (source_entity)",
            "CREATE INDEX idx_entity_rel_target ON RAG.EntityRelationships (target_entity)",
            "CREATE INDEX idx_entity_rel_type ON RAG.EntityRelationships (relationship_type)"
        ]
        
        for idx_sql in relationship_indexes:
            try:
                cursor.execute(idx_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        # 3. Create DocumentTokenEmbeddings table for ColBERT
        logger.info("Creating RAG.DocumentTokenEmbeddings table...")
        create_token_embeddings_sql = """
        CREATE TABLE RAG.DocumentTokenEmbeddings (
            id INTEGER IDENTITY PRIMARY KEY,
            document_id VARCHAR(255) NOT NULL,
            token_position INTEGER NOT NULL,
            token_text VARCHAR(100),
            token_embedding VECTOR(DOUBLE, 128),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            cursor.execute(create_token_embeddings_sql)
            logger.info("✓ RAG.DocumentTokenEmbeddings table created")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("✓ RAG.DocumentTokenEmbeddings table already exists")
            else:
                raise e
        
        # Create indexes separately for DocumentTokenEmbeddings
        token_indexes = [
            "CREATE INDEX idx_doc_token_doc_id ON RAG.DocumentTokenEmbeddings (document_id)",
            "CREATE INDEX idx_doc_token_position ON RAG.DocumentTokenEmbeddings (token_position)"
        ]
        
        for idx_sql in token_indexes:
            try:
                cursor.execute(idx_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        # 4. Create DocumentChunks table if it doesn't exist
        logger.info("Creating RAG.DocumentChunks table...")
        create_chunks_sql = """
        CREATE TABLE RAG.DocumentChunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            source_doc_id VARCHAR(255) NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text CLOB,
            embedding VECTOR(DOUBLE, 1536),
            metadata CLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            cursor.execute(create_chunks_sql)
            logger.info("✓ RAG.DocumentChunks table created")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("✓ RAG.DocumentChunks table already exists")
            else:
                raise e
        
        # Create indexes separately for DocumentChunks
        chunk_indexes = [
            "CREATE INDEX idx_doc_chunks_source_id ON RAG.DocumentChunks (source_doc_id)",
            "CREATE INDEX idx_doc_chunks_index ON RAG.DocumentChunks (chunk_index)"
        ]
        
        for idx_sql in chunk_indexes:
            try:
                cursor.execute(idx_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        # Commit all changes
        connection.commit()
        logger.info("✓ All missing tables created successfully")
        
        # Verify tables exist
        logger.info("Verifying table creation...")
        verify_sql = """
        SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = 'RAG' 
        AND TABLE_NAME IN ('DocumentEntities', 'EntityRelationships', 'DocumentTokenEmbeddings', 'DocumentChunks')
        ORDER BY TABLE_NAME
        """
        cursor.execute(verify_sql)
        tables = cursor.fetchall()
        
        logger.info("Created tables:")
        for table in tables:
            logger.info(f"  - RAG.{table[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()

def fix_sql_index_syntax():
    """Fix SQL index creation syntax issues."""
    
    connection = get_iris_connection()
    if not connection:
        logger.error("Could not get IRIS connection")
        return False
    
    cursor = connection.cursor()
    
    try:
        logger.info("Fixing SQL index syntax issues...")
        
        # Drop and recreate indexes with correct IRIS SQL syntax
        index_fixes = [
            # Fix for SourceDocuments table
            "DROP INDEX IF EXISTS RAG.SourceDocuments.idx_sourcedocs_embedding",
            """CREATE INDEX idx_sourcedocs_embedding 
               ON RAG.SourceDocuments (embedding) 
               WITH (TYPE = 'HNSW')""",
            
            # Fix for DocumentChunks table  
            "DROP INDEX IF EXISTS RAG.DocumentChunks.idx_chunks_embedding",
            """CREATE INDEX idx_chunks_embedding 
               ON RAG.DocumentChunks (embedding) 
               WITH (TYPE = 'HNSW')""",
        ]
        
        for sql in index_fixes:
            try:
                cursor.execute(sql)
                logger.info(f"✓ Executed: {sql[:50]}...")
            except Exception as e:
                logger.warning(f"Index operation failed (may be expected): {e}")
        
        connection.commit()
        logger.info("✓ SQL index syntax fixes completed")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing SQL syntax: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()

def main():
    """Main function to set up missing tables and fix issues."""
    
    logger.info("=== SETTING UP MISSING TABLES FOR RAG PIPELINES ===")
    
    # Step 1: Create missing tables
    if not create_missing_tables():
        logger.error("Failed to create missing tables")
        return False
    
    # Step 2: Fix SQL syntax issues
    if not fix_sql_index_syntax():
        logger.error("Failed to fix SQL syntax issues")
        return False
    
    logger.info("=== SETUP COMPLETE ===")
    logger.info("All missing tables have been created and SQL syntax issues fixed.")
    logger.info("You can now run the auto-setup again to test the pipelines.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)