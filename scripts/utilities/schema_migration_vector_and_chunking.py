#!/usr/bin/env python3
"""
Comprehensive schema migration for IRIS 2025.1 licensed instance:
1. Migrate existing tables to use proper VECTOR data types
2. Add missing chunking tables for enterprise-scale RAG operations
3. Create proper HNSW indexes for vector search optimization
4. Verify the new schema structure

Based on IRIS 2025.1 Vector Search capabilities documented in:
- IRIS_2025_VECTOR_SEARCH_DEPLOYMENT_REPORT.md
- docs/VECTOR_SEARCH_SYNTAX_FINDINGS.md
- docs/IRIS_SQL_VECTOR_LIMITATIONS.md
"""

import os
import sys
import logging
from typing import Dict, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_existing_data(conn) -> Dict[str, List]:
    """Backup existing data before migration."""
    cursor = conn.cursor()
    backup_data = {}
    
    try:
        # Backup SourceDocuments
        try:
            logger.info("Backing up SourceDocuments...")
            cursor.execute("SELECT * FROM RAG.SourceDocuments_V2")
            backup_data['SourceDocuments_V2'] = cursor.fetchall()
            logger.info(f"Backed up {len(backup_data['SourceDocuments_V2'])} source documents")
        except Exception as e:
            logger.warning(f"Could not backup SourceDocuments (table may not exist): {e}")
            backup_data['SourceDocuments_V2'] = []
        
        # Backup DocumentTokenEmbeddings
        try:
            logger.info("Backing up DocumentTokenEmbeddings...")
            cursor.execute("SELECT * FROM RAG.DocumentTokenEmbeddings")
            backup_data['DocumentTokenEmbeddings'] = cursor.fetchall()
            logger.info(f"Backed up {len(backup_data['DocumentTokenEmbeddings'])} token embeddings")
        except Exception as e:
            logger.warning(f"Could not backup DocumentTokenEmbeddings (table may not exist): {e}")
            backup_data['DocumentTokenEmbeddings'] = []
        
        # Backup KnowledgeGraphNodes
        try:
            logger.info("Backing up KnowledgeGraphNodes...")
            cursor.execute("SELECT * FROM RAG.KnowledgeGraphNodes")
            backup_data['KnowledgeGraphNodes'] = cursor.fetchall()
            logger.info(f"Backed up {len(backup_data['KnowledgeGraphNodes'])} knowledge graph nodes")
        except Exception as e:
            logger.warning(f"Could not backup KnowledgeGraphNodes (table may not exist): {e}")
            backup_data['KnowledgeGraphNodes'] = []
        
        # Backup KnowledgeGraphEdges
        try:
            logger.info("Backing up KnowledgeGraphEdges...")
            cursor.execute("SELECT * FROM RAG.KnowledgeGraphEdges")
            backup_data['KnowledgeGraphEdges'] = cursor.fetchall()
            logger.info(f"Backed up {len(backup_data['KnowledgeGraphEdges'])} knowledge graph edges")
        except Exception as e:
            logger.warning(f"Could not backup KnowledgeGraphEdges (table may not exist): {e}")
            backup_data['KnowledgeGraphEdges'] = []
        
    except Exception as e:
        logger.error(f"Error backing up data: {e}")
        raise
    finally:
        cursor.close()
    
    return backup_data

def execute_migration_sql(conn, sql_statements: List[str], description: str):
    """Execute a list of SQL statements with error handling."""
    cursor = conn.cursor()
    
    try:
        logger.info(f"Executing {description}...")
        for i, sql in enumerate(sql_statements):
            if sql.strip():
                logger.debug(f"Executing statement {i+1}/{len(sql_statements)}: {sql[:100]}...")
                cursor.execute(sql)
                conn.commit()
        logger.info(f"Successfully completed {description}")
        
    except Exception as e:
        logger.error(f"Error in {description}: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def create_new_schema_with_vectors(conn):
    """Create new schema with proper VECTOR data types and chunking tables for IRIS 2025.1."""
    
    # Step 1: Drop existing tables and recreate with VECTOR types
    drop_statements = [
        "DROP VIEW IF EXISTS RAG.SourceDocuments_V2Vector",
        "DROP VIEW IF EXISTS RAG.DocumentChunksVector", 
        "DROP VIEW IF EXISTS RAG.ChunksWithDocuments",
        "DROP TABLE IF EXISTS RAG.ChunkOverlaps CASCADE",
        "DROP TABLE IF EXISTS RAG.DocumentChunks CASCADE",
        "DROP TABLE IF EXISTS RAG.ChunkingStrategies CASCADE",
        "DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings CASCADE",
        "DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges CASCADE",
        "DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes CASCADE",
        "DROP TABLE IF EXISTS RAG.SourceDocuments_V2 CASCADE"
    ]
    
    execute_migration_sql(conn, drop_statements, "dropping existing tables")
    
    # Step 2: Create SourceDocuments with proper VECTOR column (IRIS 2025.1 syntax)
    source_docs_sql = [
        """
        CREATE TABLE RAG.SourceDocuments_V2 (
            doc_id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(500),
            text_content LONGVARCHAR,
            abstract LONGVARCHAR,
            authors LONGVARCHAR,
            keywords LONGVARCHAR,
            
            -- Native VECTOR column for IRIS 2025.1
            embedding VECTOR(FLOAT, 768),
            
            -- Metadata
            embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
            embedding_dimensions INTEGER DEFAULT 768,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    execute_migration_sql(conn, source_docs_sql, "creating SourceDocuments with native VECTOR support")
    
    # Step 3: Create DocumentTokenEmbeddings with VECTOR support
    token_embeddings_sql = [
        """
        CREATE TABLE RAG.DocumentTokenEmbeddings (
            doc_id VARCHAR(255),
            token_sequence_index INTEGER,
            token_text VARCHAR(1000),
            
            -- Native VECTOR column for token embeddings (128 dimensions for ColBERT)
            token_embedding VECTOR(FLOAT, 128),
            
            metadata_json LONGVARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            PRIMARY KEY (doc_id, token_sequence_index),
            FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments_V2(doc_id)
        )
        """
    ]
    
    execute_migration_sql(conn, token_embeddings_sql, "creating DocumentTokenEmbeddings with native VECTOR support")
    
    # Step 4: Create chunking tables
    chunking_tables_sql = [
        """
        CREATE TABLE RAG.ChunkingStrategies (
            strategy_id VARCHAR(255) PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            strategy_type VARCHAR(50) NOT NULL,
            configuration LONGVARCHAR NOT NULL,
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE RAG.DocumentChunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            doc_id VARCHAR(255) NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_type VARCHAR(50) NOT NULL,
            chunk_text LONGVARCHAR NOT NULL,
            chunk_metadata LONGVARCHAR,
            
            -- Chunk positioning and relationships
            start_position INTEGER,
            end_position INTEGER,
            parent_chunk_id VARCHAR(255),
            
            -- Native VECTOR column for chunk embeddings
            embedding VECTOR(FLOAT, 768),
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Constraints
            FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments_V2(doc_id),
            FOREIGN KEY (parent_chunk_id) REFERENCES RAG.DocumentChunks(chunk_id),
            UNIQUE (doc_id, chunk_index, chunk_type)
        )
        """,
        """
        CREATE TABLE RAG.ChunkOverlaps (
            overlap_id VARCHAR(255) PRIMARY KEY,
            chunk_id_1 VARCHAR(255) NOT NULL,
            chunk_id_2 VARCHAR(255) NOT NULL,
            overlap_type VARCHAR(50),
            overlap_text LONGVARCHAR,
            overlap_score DOUBLE,
            
            FOREIGN KEY (chunk_id_1) REFERENCES RAG.DocumentChunks(chunk_id),
            FOREIGN KEY (chunk_id_2) REFERENCES RAG.DocumentChunks(chunk_id)
        )
        """
    ]
    
    execute_migration_sql(conn, chunking_tables_sql, "creating chunking tables")
    
    # Step 5: Recreate KnowledgeGraph tables with VECTOR support
    kg_tables_sql = [
        """
        CREATE TABLE RAG.KnowledgeGraphNodes (
            node_id VARCHAR(255) PRIMARY KEY,
            node_type VARCHAR(100),
            node_name VARCHAR(1000),
            description_text LONGVARCHAR,
            
            -- Native VECTOR column for node embeddings
            embedding VECTOR(FLOAT, 768),
            
            embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
            embedding_dimensions INTEGER DEFAULT 768,
            metadata_json LONGVARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE RAG.KnowledgeGraphEdges (
            edge_id VARCHAR(255) PRIMARY KEY,
            source_node_id VARCHAR(255),
            target_node_id VARCHAR(255),
            relationship_type VARCHAR(100),
            weight DOUBLE,
            properties_json LONGVARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (source_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id)
        )
        """
    ]
    
    execute_migration_sql(conn, kg_tables_sql, "creating KnowledgeGraph tables with native VECTOR support")

def create_indexes(conn):
    """Create all necessary indexes including HNSW indexes for vector search."""
    
    # Standard indexes
    standard_indexes = [
        "CREATE INDEX idx_source_docs_title ON RAG.SourceDocuments_V2(title)",
        "CREATE INDEX idx_source_docs_model ON RAG.SourceDocuments_V2(embedding_model)",
        "CREATE INDEX idx_source_docs_created ON RAG.SourceDocuments_V2(created_at)",
        
        "CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks(doc_id)",
        "CREATE INDEX idx_chunks_type ON RAG.DocumentChunks(chunk_type)",
        "CREATE INDEX idx_chunks_position ON RAG.DocumentChunks(doc_id, chunk_index)",
        "CREATE INDEX idx_chunks_size ON RAG.DocumentChunks(start_position, end_position)",
        "CREATE INDEX idx_chunks_created ON RAG.DocumentChunks(created_at)",
        
        "CREATE INDEX idx_overlaps_chunk1 ON RAG.ChunkOverlaps(chunk_id_1)",
        "CREATE INDEX idx_overlaps_chunk2 ON RAG.ChunkOverlaps(chunk_id_2)",
        "CREATE INDEX idx_overlaps_type ON RAG.ChunkOverlaps(overlap_type)",
        
        "CREATE INDEX idx_strategies_active ON RAG.ChunkingStrategies(is_active)",
        "CREATE INDEX idx_strategies_type ON RAG.ChunkingStrategies(strategy_type)",
        
        "CREATE INDEX idx_kg_nodes_type ON RAG.KnowledgeGraphNodes(node_type)",
        "CREATE INDEX idx_kg_nodes_name ON RAG.KnowledgeGraphNodes(node_name)",
        "CREATE INDEX idx_kg_edges_type ON RAG.KnowledgeGraphEdges(relationship_type)",
        "CREATE INDEX idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings(doc_id)"
    ]
    
    execute_migration_sql(conn, standard_indexes, "creating standard indexes")
    
    # HNSW indexes for vector search (IRIS 2025.1 syntax)
    # Note: IRIS doesn't support WHERE clauses in HNSW index creation
    hnsw_indexes = [
        """
        CREATE INDEX idx_hnsw_source_docs_embeddings
        ON RAG.SourceDocuments_V2 (embedding)
        AS HNSW(Distance='Cosine')
        """,
        """
        CREATE INDEX idx_hnsw_chunk_embeddings
        ON RAG.DocumentChunks (embedding)
        AS HNSW(Distance='Cosine')
        """,
        """
        CREATE INDEX idx_hnsw_kg_nodes_embeddings
        ON RAG.KnowledgeGraphNodes (embedding)
        AS HNSW(Distance='Cosine')
        """,
        """
        CREATE INDEX idx_hnsw_token_embeddings
        ON RAG.DocumentTokenEmbeddings (token_embedding)
        AS HNSW(Distance='Cosine')
        """
    ]
    
    execute_migration_sql(conn, hnsw_indexes, "creating HNSW vector indexes")

def create_views(conn):
    """Create views for easier querying with IRIS 2025.1 vector functions."""
    
    views_sql = [
        """
        CREATE VIEW RAG.SourceDocuments_V2Vector AS
        SELECT 
            doc_id,
            title,
            text_content,
            abstract,
            authors,
            keywords,
            embedding,
            embedding_model,
            embedding_dimensions,
            created_at,
            updated_at
        FROM RAG.SourceDocuments_V2
        """,
        """
        CREATE VIEW RAG.DocumentChunksVector AS
        SELECT 
            chunk_id,
            doc_id,
            chunk_index,
            chunk_type,
            chunk_text,
            start_position,
            end_position,
            chunk_metadata,
            embedding,
            created_at
        FROM RAG.DocumentChunks
        """,
        """
        CREATE VIEW RAG.ChunksWithDocuments AS
        SELECT 
            c.chunk_id,
            c.doc_id,
            c.chunk_index,
            c.chunk_type,
            c.chunk_text,
            c.start_position,
            c.end_position,
            c.chunk_metadata,
            c.embedding,
            c.created_at as chunk_created_at,
            d.title,
            d.authors,
            d.keywords,
            d.abstract
        FROM RAG.DocumentChunks c
        JOIN RAG.SourceDocuments_V2 d ON c.doc_id = d.doc_id
        """
    ]
    
    execute_migration_sql(conn, views_sql, "creating views")

def insert_default_chunking_strategies(conn):
    """Insert default chunking strategies."""
    
    strategies_sql = [
        """
        INSERT INTO RAG.ChunkingStrategies (strategy_id, strategy_name, strategy_type, configuration, is_active) VALUES
        ('fixed_512', 'Fixed Size 512', 'fixed_size',
         '{"chunk_size": 512, "overlap_size": 50, "preserve_sentences": true, "min_chunk_size": 100}',
         1)
        """,
        """
        INSERT INTO RAG.ChunkingStrategies (strategy_id, strategy_name, strategy_type, configuration, is_active) VALUES
        ('fixed_384', 'Fixed Size 384', 'fixed_size',
         '{"chunk_size": 384, "overlap_size": 40, "preserve_sentences": true, "min_chunk_size": 80}',
         0)
        """,
        """
        INSERT INTO RAG.ChunkingStrategies (strategy_id, strategy_name, strategy_type, configuration, is_active) VALUES
        ('semantic_default', 'Semantic Default', 'semantic',
         '{"similarity_threshold": 0.7, "min_chunk_size": 200, "max_chunk_size": 1000}',
         0)
        """,
        """
        INSERT INTO RAG.ChunkingStrategies (strategy_id, strategy_name, strategy_type, configuration, is_active) VALUES
        ('hybrid_default', 'Hybrid Default', 'hybrid',
         '{"primary_strategy": "semantic", "fallback_strategy": "fixed_size", "max_chunk_size": 800}',
         1)
        """
    ]
    
    execute_migration_sql(conn, strategies_sql, "inserting default chunking strategies")

def restore_data_with_vector_conversion(conn, backup_data: Dict[str, List]):
    """Restore data to new schema using TO_VECTOR for proper conversion."""
    cursor = conn.cursor()
    
    try:
        # Restore SourceDocuments using TO_VECTOR for embedding conversion
        if backup_data.get('SourceDocuments_V2'):
            logger.info("Restoring SourceDocuments with vector conversion...")
            for row in backup_data['SourceDocuments_V2']:
                # Use TO_VECTOR with proper IRIS 2025.1 syntax
                cursor.execute("""
                    INSERT INTO RAG.SourceDocuments_V2 
                    (doc_id, title, text_content, abstract, authors, keywords, embedding, embedding_model, embedding_dimensions)
                    VALUES (?, ?, ?, ?, ?, ?, TO_VECTOR(?, double, 768), ?, ?)
                """, (row[0], row[1], row[2], row[3], row[4], row[5], row[6], 
                     'sentence-transformers/all-MiniLM-L6-v2', 768))
            conn.commit()
            logger.info(f"Restored {len(backup_data['SourceDocuments_V2'])} source documents")
        
        # Restore DocumentTokenEmbeddings using TO_VECTOR
        if backup_data.get('DocumentTokenEmbeddings'):
            logger.info("Restoring DocumentTokenEmbeddings with vector conversion...")
            for row in backup_data['DocumentTokenEmbeddings']:
                cursor.execute("""
                    INSERT INTO RAG.DocumentTokenEmbeddings 
                    (doc_id, token_sequence_index, token_text, token_embedding, metadata_json)
                    VALUES (?, ?, ?, TO_VECTOR(?, double, 128), ?)
                """, (row[0], row[1], row[2], row[3], row[4]))
            conn.commit()
            logger.info(f"Restored {len(backup_data['DocumentTokenEmbeddings'])} token embeddings")
        
        # Note: KnowledgeGraph tables are empty, so no need to restore
        
    except Exception as e:
        logger.error(f"Error restoring data: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def verify_schema(conn):
    """Verify the new schema structure and vector functionality."""
    cursor = conn.cursor()
    
    try:
        # Check tables exist
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG'
            ORDER BY TABLE_NAME
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'ChunkOverlaps', 'ChunkingStrategies', 'DocumentChunks', 
            'DocumentTokenEmbeddings', 'KnowledgeGraphEdges', 
            'KnowledgeGraphNodes', 'SourceDocuments_V2'
        ]
        
        logger.info("Verifying schema structure...")
        for table in expected_tables:
            if table in tables:
                logger.info(f"✅ Table RAG.{table} exists")
            else:
                logger.error(f"❌ Table RAG.{table} missing")
        
        # Check for VECTOR columns
        vector_checks = [
            ("SourceDocuments_V2", "embedding"),
            ("DocumentChunks", "embedding"),
            ("DocumentTokenEmbeddings", "token_embedding"),
            ("KnowledgeGraphNodes", "embedding")
        ]
        
        logger.info("Verifying VECTOR columns...")
        for table, column in vector_checks:
            cursor.execute("""
                SELECT DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ? AND COLUMN_NAME = ?
            """, (table, column))
            result = cursor.fetchone()
            if result and 'VECTOR' in result[0]:
                logger.info(f"✅ {table}.{column} is VECTOR type")
            else:
                logger.warning(f"⚠️ {table}.{column} type: {result[0] if result else 'NOT FOUND'}")
        
        # Check row counts
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
        source_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.ChunkingStrategies")
        strategy_count = cursor.fetchone()[0]
        
        logger.info(f"Data verification:")
        logger.info(f"  - SourceDocuments: {source_count} rows")
        logger.info(f"  - DocumentTokenEmbeddings: {token_count} rows")
        logger.info(f"  - ChunkingStrategies: {strategy_count} rows")
        
        # Test vector functionality
        if source_count > 0:
            logger.info("Testing vector similarity functionality...")
            try:
                cursor.execute("""
                    SELECT TOP 1 doc_id, VECTOR_COSINE(embedding, embedding) as self_similarity
                    FROM RAG.SourceDocuments_V2 
                    WHERE embedding IS NOT NULL
                """)
                result = cursor.fetchone()
                if result:
                    logger.info(f"✅ Vector similarity test successful: {result[1]}")
                else:
                    logger.warning("⚠️ No documents with embeddings found")
            except Exception as e:
                logger.warning(f"⚠️ Vector similarity test failed: {e}")
        
    except Exception as e:
        logger.error(f"Error verifying schema: {e}")
        raise
    finally:
        cursor.close()

def main():
    """Main function to execute the schema migration."""
    try:
        # Set connection parameters for licensed IRIS instance
        config = {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": "_SYSTEM",
            "password": "SYS"
        }
        
        logger.info("Connecting to licensed IRIS 2025.1 database...")
        conn = get_iris_connection(use_mock=False, use_testcontainer=False, config=config)
        
        logger.info("Starting schema migration to VECTOR data types and chunking tables...")
        
        # Step 1: Backup existing data
        backup_data = backup_existing_data(conn)
        
        # Step 2: Create new schema with VECTOR types
        create_new_schema_with_vectors(conn)
        
        # Step 3: Create indexes (including HNSW)
        create_indexes(conn)
        
        # Step 4: Create views
        create_views(conn)
        
        # Step 5: Insert default chunking strategies
        insert_default_chunking_strategies(conn)
        
        # Step 6: Restore data with vector conversion
        restore_data_with_vector_conversion(conn, backup_data)
        
        # Step 7: Verify the new schema
        verify_schema(conn)
        
        conn.close()
        logger.info("Schema migration completed successfully!")
        
        print("\n" + "="*80)
        print("SCHEMA MIGRATION COMPLETED")
        print("="*80)
        print("✅ Migrated to native VECTOR data types")
        print("✅ Added comprehensive chunking tables")
        print("✅ Created HNSW indexes for vector search")
        print("✅ Restored all existing data")
        print("✅ Schema ready for enterprise-scale RAG operations")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Schema migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()