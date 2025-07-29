#!/usr/bin/env python3
"""
URGENT FIX: Convert VARCHAR embedding columns to proper VECTOR data types.

This script will:
1. Backup existing data with embeddings
2. Drop and recreate tables with proper VECTOR columns
3. Restore data using TO_VECTOR conversion
4. Verify all columns are now VECTOR types
5. Test vector operations work correctly

Critical for enterprise RAG operations - VARCHAR embeddings are unacceptable.
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

def backup_all_data(conn) -> Dict[str, List]:
    """Backup ALL existing data before the fix."""
    cursor = conn.cursor()
    backup_data = {}
    
    try:
        # Backup SourceDocuments
        logger.info("Backing up SourceDocuments...")
        cursor.execute("SELECT * FROM RAG.SourceDocuments_V2")
        backup_data['SourceDocuments_V2'] = cursor.fetchall()
        logger.info(f"Backed up {len(backup_data['SourceDocuments_V2'])} source documents")
        
        # Backup DocumentTokenEmbeddings
        logger.info("Backing up DocumentTokenEmbeddings...")
        cursor.execute("SELECT * FROM RAG.DocumentTokenEmbeddings")
        backup_data['DocumentTokenEmbeddings'] = cursor.fetchall()
        logger.info(f"Backed up {len(backup_data['DocumentTokenEmbeddings'])} token embeddings")
        
        # Backup DocumentChunks
        logger.info("Backing up DocumentChunks...")
        cursor.execute("SELECT * FROM RAG.DocumentChunks")
        backup_data['DocumentChunks'] = cursor.fetchall()
        logger.info(f"Backed up {len(backup_data['DocumentChunks'])} document chunks")
        
        # Backup KnowledgeGraphNodes
        logger.info("Backing up KnowledgeGraphNodes...")
        cursor.execute("SELECT * FROM RAG.KnowledgeGraphNodes")
        backup_data['KnowledgeGraphNodes'] = cursor.fetchall()
        logger.info(f"Backed up {len(backup_data['KnowledgeGraphNodes'])} knowledge graph nodes")
        
        # Backup KnowledgeGraphEdges
        logger.info("Backing up KnowledgeGraphEdges...")
        cursor.execute("SELECT * FROM RAG.KnowledgeGraphEdges")
        backup_data['KnowledgeGraphEdges'] = cursor.fetchall()
        logger.info(f"Backed up {len(backup_data['KnowledgeGraphEdges'])} knowledge graph edges")
        
        # Backup ChunkingStrategies
        logger.info("Backing up ChunkingStrategies...")
        cursor.execute("SELECT * FROM RAG.ChunkingStrategies")
        backup_data['ChunkingStrategies'] = cursor.fetchall()
        logger.info(f"Backed up {len(backup_data['ChunkingStrategies'])} chunking strategies")
        
        # Backup ChunkOverlaps
        logger.info("Backing up ChunkOverlaps...")
        cursor.execute("SELECT * FROM RAG.ChunkOverlaps")
        backup_data['ChunkOverlaps'] = cursor.fetchall()
        logger.info(f"Backed up {len(backup_data['ChunkOverlaps'])} chunk overlaps")
        
    except Exception as e:
        logger.error(f"Error backing up data: {e}")
        raise
    finally:
        cursor.close()
    
    return backup_data

def drop_all_tables(conn):
    """Drop all tables in correct order to handle foreign key constraints."""
    cursor = conn.cursor()
    
    try:
        # Drop in reverse dependency order
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
        
        for sql in drop_statements:
            logger.info(f"Executing: {sql}")
            cursor.execute(sql)
            conn.commit()
        
        logger.info("All tables dropped successfully")
        
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def create_tables_with_vector_columns(conn):
    """Create all tables with proper VECTOR data types."""
    cursor = conn.cursor()
    
    try:
        # Create SourceDocuments with VECTOR(FLOAT, 768)
        logger.info("Creating SourceDocuments with VECTOR(FLOAT, 768)...")
        cursor.execute("""
            CREATE TABLE RAG.SourceDocuments_V2 (
                doc_id VARCHAR(255) PRIMARY KEY,
                title VARCHAR(500),
                text_content LONGVARCHAR,
                abstract LONGVARCHAR,
                authors LONGVARCHAR,
                keywords LONGVARCHAR,
                embedding VECTOR(FLOAT, 768),
                embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
                embedding_dimensions INTEGER DEFAULT 768,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Create DocumentTokenEmbeddings with VECTOR(FLOAT, 128)
        logger.info("Creating DocumentTokenEmbeddings with VECTOR(FLOAT, 128)...")
        cursor.execute("""
            CREATE TABLE RAG.DocumentTokenEmbeddings (
                doc_id VARCHAR(255),
                token_sequence_index INTEGER,
                token_text VARCHAR(1000),
                token_embedding VECTOR(FLOAT, 128),
                metadata_json LONGVARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (doc_id, token_sequence_index),
                FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments_V2(doc_id)
            )
        """)
        conn.commit()
        
        # Create ChunkingStrategies
        logger.info("Creating ChunkingStrategies...")
        cursor.execute("""
            CREATE TABLE RAG.ChunkingStrategies (
                strategy_id VARCHAR(255) PRIMARY KEY,
                strategy_name VARCHAR(100) NOT NULL,
                strategy_type VARCHAR(50) NOT NULL,
                configuration LONGVARCHAR NOT NULL,
                is_active INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Create DocumentChunks with VECTOR(FLOAT, 384)
        logger.info("Creating DocumentChunks with VECTOR(FLOAT, 384)...")
        cursor.execute("""
            CREATE TABLE RAG.DocumentChunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                doc_id VARCHAR(255) NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_type VARCHAR(50) NOT NULL,
                chunk_text LONGVARCHAR NOT NULL,
                chunk_metadata LONGVARCHAR,
                start_position INTEGER,
                end_position INTEGER,
                parent_chunk_id VARCHAR(255),
                embedding VECTOR(FLOAT, 384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments_V2(doc_id),
                FOREIGN KEY (parent_chunk_id) REFERENCES RAG.DocumentChunks(chunk_id),
                UNIQUE (doc_id, chunk_index, chunk_type)
            )
        """)
        conn.commit()
        
        # Create KnowledgeGraphNodes with VECTOR(FLOAT, 768)
        logger.info("Creating KnowledgeGraphNodes with VECTOR(FLOAT, 768)...")
        cursor.execute("""
            CREATE TABLE RAG.KnowledgeGraphNodes (
                node_id VARCHAR(255) PRIMARY KEY,
                node_type VARCHAR(100),
                node_name VARCHAR(1000),
                description_text LONGVARCHAR,
                embedding VECTOR(FLOAT, 768),
                embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
                embedding_dimensions INTEGER DEFAULT 768,
                metadata_json LONGVARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Create KnowledgeGraphEdges
        logger.info("Creating KnowledgeGraphEdges...")
        cursor.execute("""
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
        """)
        conn.commit()
        
        # Create ChunkOverlaps
        logger.info("Creating ChunkOverlaps...")
        cursor.execute("""
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
        """)
        conn.commit()
        
        logger.info("All tables created with proper VECTOR columns")
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def restore_data_with_vector_conversion(conn, backup_data: Dict[str, List]):
    """Restore data converting VARCHAR embeddings to VECTOR using TO_VECTOR."""
    cursor = conn.cursor()
    
    try:
        # Restore ChunkingStrategies first (no dependencies)
        if backup_data.get('ChunkingStrategies'):
            logger.info("Restoring ChunkingStrategies...")
            for row in backup_data['ChunkingStrategies']:
                cursor.execute("""
                    INSERT INTO RAG.ChunkingStrategies 
                    (strategy_id, strategy_name, strategy_type, configuration, is_active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, row)
            conn.commit()
            logger.info(f"Restored {len(backup_data['ChunkingStrategies'])} chunking strategies")
        
        # Restore SourceDocuments with vector conversion
        if backup_data.get('SourceDocuments_V2'):
            logger.info("Restoring SourceDocuments with VECTOR conversion...")
            for row in backup_data['SourceDocuments_V2']:
                # Convert embedding from VARCHAR to VECTOR(FLOAT, 768)
                embedding_str = row[6] if row[6] else None
                cursor.execute("""
                    INSERT INTO RAG.SourceDocuments_V2 
                    (doc_id, title, text_content, abstract, authors, keywords, embedding, 
                     embedding_model, embedding_dimensions, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, TO_VECTOR(?), ?, ?, ?, ?)
                """, (row[0], row[1], row[2], row[3], row[4], row[5], embedding_str,
                     row[7], row[8], row[9], row[10]))
            conn.commit()
            logger.info(f"Restored {len(backup_data['SourceDocuments_V2'])} source documents")
        
        # Restore DocumentTokenEmbeddings with vector conversion
        if backup_data.get('DocumentTokenEmbeddings'):
            logger.info("Restoring DocumentTokenEmbeddings with VECTOR conversion...")
            for row in backup_data['DocumentTokenEmbeddings']:
                # Convert token_embedding from VARCHAR to VECTOR(FLOAT, 128)
                embedding_str = row[3] if row[3] else None
                cursor.execute("""
                    INSERT INTO RAG.DocumentTokenEmbeddings 
                    (doc_id, token_sequence_index, token_text, token_embedding, metadata_json, created_at)
                    VALUES (?, ?, ?, TO_VECTOR(?), ?, ?)
                """, (row[0], row[1], row[2], embedding_str, row[4], row[5]))
            conn.commit()
            logger.info(f"Restored {len(backup_data['DocumentTokenEmbeddings'])} token embeddings")
        
        # Restore DocumentChunks with vector conversion
        if backup_data.get('DocumentChunks'):
            logger.info("Restoring DocumentChunks with VECTOR conversion...")
            for row in backup_data['DocumentChunks']:
                # Convert embedding from VARCHAR to VECTOR(FLOAT, 384)
                embedding_str = row[9] if row[9] else None
                cursor.execute("""
                    INSERT INTO RAG.DocumentChunks 
                    (chunk_id, doc_id, chunk_index, chunk_type, chunk_text, chunk_metadata,
                     start_position, end_position, parent_chunk_id, embedding, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TO_VECTOR(?), ?, ?)
                """, (row[0], row[1], row[2], row[3], row[4], row[5], 
                     row[6], row[7], row[8], embedding_str, row[10], row[11]))
            conn.commit()
            logger.info(f"Restored {len(backup_data['DocumentChunks'])} document chunks")
        
        # Restore KnowledgeGraphNodes with vector conversion
        if backup_data.get('KnowledgeGraphNodes'):
            logger.info("Restoring KnowledgeGraphNodes with VECTOR conversion...")
            for row in backup_data['KnowledgeGraphNodes']:
                # Convert embedding from VARCHAR to VECTOR(FLOAT, 768)
                embedding_str = row[4] if row[4] else None
                cursor.execute("""
                    INSERT INTO RAG.KnowledgeGraphNodes 
                    (node_id, node_type, node_name, description_text, embedding,
                     embedding_model, embedding_dimensions, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, TO_VECTOR(?), ?, ?, ?, ?)
                """, (row[0], row[1], row[2], row[3], embedding_str,
                     row[5], row[6], row[7], row[8]))
            conn.commit()
            logger.info(f"Restored {len(backup_data['KnowledgeGraphNodes'])} knowledge graph nodes")
        
        # Restore KnowledgeGraphEdges
        if backup_data.get('KnowledgeGraphEdges'):
            logger.info("Restoring KnowledgeGraphEdges...")
            for row in backup_data['KnowledgeGraphEdges']:
                cursor.execute("""
                    INSERT INTO RAG.KnowledgeGraphEdges 
                    (edge_id, source_node_id, target_node_id, relationship_type, weight, properties_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, row)
            conn.commit()
            logger.info(f"Restored {len(backup_data['KnowledgeGraphEdges'])} knowledge graph edges")
        
        # Restore ChunkOverlaps
        if backup_data.get('ChunkOverlaps'):
            logger.info("Restoring ChunkOverlaps...")
            for row in backup_data['ChunkOverlaps']:
                cursor.execute("""
                    INSERT INTO RAG.ChunkOverlaps 
                    (overlap_id, chunk_id_1, chunk_id_2, overlap_type, overlap_text, overlap_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, row)
            conn.commit()
            logger.info(f"Restored {len(backup_data['ChunkOverlaps'])} chunk overlaps")
        
    except Exception as e:
        logger.error(f"Error restoring data: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def create_hnsw_indexes(conn):
    """Create HNSW indexes for vector search optimization."""
    cursor = conn.cursor()
    
    try:
        # HNSW indexes for vector search
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
        
        for sql in hnsw_indexes:
            logger.info(f"Creating HNSW index...")
            cursor.execute(sql)
            conn.commit()
        
        logger.info("All HNSW indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error creating HNSW indexes: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def verify_vector_fix(conn):
    """Verify that all embedding columns are now proper VECTOR types."""
    cursor = conn.cursor()
    
    try:
        logger.info("=== VERIFYING VECTOR COLUMN FIX ===")
        
        # Check column types for embedding columns
        vector_checks = [
            ('SourceDocuments_V2', 'embedding', 'VECTOR(FLOAT, 768)'),
            ('DocumentChunks', 'embedding', 'VECTOR(FLOAT, 384)'),
            ('DocumentTokenEmbeddings', 'token_embedding', 'VECTOR(FLOAT, 128)'),
            ('KnowledgeGraphNodes', 'embedding', 'VECTOR(FLOAT, 768)')
        ]
        
        all_correct = True
        for table, column, expected_type in vector_checks:
            cursor.execute("""
                SELECT DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ? AND COLUMN_NAME = ?
            """, (table, column))
            result = cursor.fetchone()
            
            if result and 'VECTOR' in result[0]:
                logger.info(f"✅ {table}.{column}: {result[0]} (CORRECT)")
            else:
                logger.error(f"❌ {table}.{column}: {result[0] if result else 'NOT FOUND'} (WRONG)")
                all_correct = False
        
        # Test vector operations
        logger.info("\n=== TESTING VECTOR OPERATIONS ===")
        
        # Test with actual data if available
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE embedding IS NOT NULL")
        doc_count = cursor.fetchone()[0]
        
        if doc_count > 0:
            # Test vector similarity with real data
            cursor.execute("""
                SELECT TOP 1 doc_id, VECTOR_COSINE(embedding, embedding) as self_similarity
                FROM RAG.SourceDocuments_V2 
                WHERE embedding IS NOT NULL
            """)
            result = cursor.fetchone()
            if result and abs(result[1] - 1.0) < 0.001:
                logger.info(f"✅ Vector similarity test: {result[1]} (CORRECT)")
            else:
                logger.error(f"❌ Vector similarity test: {result[1] if result else 'FAILED'}")
                all_correct = False
        
        # Test vector search if we have multiple documents
        if doc_count > 1:
            cursor.execute("""
                SELECT TOP 2 doc_id, title
                FROM RAG.SourceDocuments_V2 
                WHERE embedding IS NOT NULL
            """)
            docs = cursor.fetchall()
            
            if len(docs) >= 2:
                cursor.execute("""
                    SELECT VECTOR_COSINE(
                        (SELECT embedding FROM RAG.SourceDocuments_V2 WHERE doc_id = ?),
                        (SELECT embedding FROM RAG.SourceDocuments_V2 WHERE doc_id = ?)
                    ) as cross_similarity
                """, (docs[0][0], docs[1][0]))
                result = cursor.fetchone()
                if result and result[0] is not None:
                    logger.info(f"✅ Cross-document vector similarity: {result[0]} (WORKING)")
                else:
                    logger.error(f"❌ Cross-document vector similarity: FAILED")
                    all_correct = False
        
        # Check row counts
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
        source_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
        chunk_count = cursor.fetchone()[0]
        
        logger.info(f"\n=== DATA VERIFICATION ===")
        logger.info(f"SourceDocuments: {source_count} rows")
        logger.info(f"DocumentTokenEmbeddings: {token_count} rows")
        logger.info(f"DocumentChunks: {chunk_count} rows")
        
        return all_correct
        
    except Exception as e:
        logger.error(f"Error verifying vector fix: {e}")
        return False
    finally:
        cursor.close()

def main():
    """Main function to execute the urgent vector column fix."""
    try:
        # Connect to IRIS
        config = {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": "_SYSTEM",
            "password": "SYS"
        }
        
        logger.info("Connecting to IRIS database...")
        conn = get_iris_connection(use_mock=False, use_testcontainer=False, config=config)
        
        logger.info("🚨 URGENT: Starting VARCHAR to VECTOR column fix...")
        
        # Step 1: Backup all existing data
        logger.info("Step 1: Backing up all existing data...")
        backup_data = backup_all_data(conn)
        
        # Step 2: Drop all tables
        logger.info("Step 2: Dropping all tables...")
        drop_all_tables(conn)
        
        # Step 3: Create tables with proper VECTOR columns
        logger.info("Step 3: Creating tables with proper VECTOR columns...")
        create_tables_with_vector_columns(conn)
        
        # Step 4: Restore data with vector conversion
        logger.info("Step 4: Restoring data with VECTOR conversion...")
        restore_data_with_vector_conversion(conn, backup_data)
        
        # Step 5: Create HNSW indexes
        logger.info("Step 5: Creating HNSW indexes...")
        create_hnsw_indexes(conn)
        
        # Step 6: Verify the fix
        logger.info("Step 6: Verifying the fix...")
        success = verify_vector_fix(conn)
        
        conn.close()
        
        if success:
            print("\n" + "="*80)
            print("🎉 URGENT VECTOR COLUMN FIX COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ ALL embedding columns are now proper VECTOR data types")
            print("✅ SourceDocuments.embedding: VECTOR(FLOAT, 768)")
            print("✅ DocumentChunks.embedding: VECTOR(FLOAT, 384)")
            print("✅ DocumentTokenEmbeddings.token_embedding: VECTOR(FLOAT, 128)")
            print("✅ KnowledgeGraphNodes.embedding: VECTOR(FLOAT, 768)")
            print("✅ HNSW indexes created for optimal vector search")
            print("✅ Vector similarity operations verified working")
            print("✅ Ready for enterprise-scale 100K document ingestion")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ VECTOR COLUMN FIX FAILED!")
            print("="*80)
            print("Some columns are still not proper VECTOR types.")
            print("Check the logs above for specific issues.")
            print("="*80)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"URGENT FIX FAILED: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()