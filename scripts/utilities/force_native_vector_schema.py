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
                # Drop order can matter with Foreign Keys if not using CASCADE,
                # but CASCADE should handle it. Listing dependent ones first for clarity.
                "RAG.DocumentTokenEmbeddings",
                "RAG.DocumentChunks",
                "RAG.DocumentEntities",      # New
                "RAG.EntityRelationships",   # Renamed from RAG.Relationships
                "RAG.KnowledgeGraphEdges",   # New
                "RAG.Entities",
                "RAG.KnowledgeGraphNodes",   # New
                "RAG.SourceDocuments",
                "RAG.Communities"            # Existing, origin/use unclear but kept for now
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
                title VARCHAR(1000) NULL, -- Added title column
                text_content TEXT,
                embedding VECTOR(FLOAT, 384),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            cursor.execute(create_source_docs)
            logging.info("✅ Created SourceDocuments with native VECTOR(FLOAT, 384)")
            
            # Step 3: Create DocumentChunks with native VECTOR
            logging.info("--- Step 3: Creating DocumentChunks with native VECTOR ---")
            
            create_chunks = """
            CREATE TABLE RAG.DocumentChunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                doc_id VARCHAR(255),
                chunk_text TEXT,
                chunk_embedding VECTOR(FLOAT, 384),
                chunk_index INTEGER,
                chunk_type VARCHAR(100),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
            )
            """
            
            cursor.execute(create_chunks)
            logging.info("✅ Created DocumentChunks with native VECTOR(FLOAT, 384)")

            # Step 3b: Create DocumentTokenEmbeddings for ColBERT
            logging.info("--- Step 3b: Creating DocumentTokenEmbeddings with native VECTOR ---")
            create_token_embeddings = """
            CREATE TABLE RAG.DocumentTokenEmbeddings (
                id VARCHAR(512) PRIMARY KEY, -- Composite key like doc_id + token_index
                doc_id VARCHAR(255),
                token_index INTEGER,          -- Index of the token within the document
                token_text VARCHAR(1000),     -- Optional: the actual token string
                token_embedding VECTOR(FLOAT, 768), -- Reverted: Sticking with 768-dim based on observed model output
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
            )
            """
            cursor.execute(create_token_embeddings)
            logging.info("✅ Created DocumentTokenEmbeddings with native VECTOR(FLOAT, 768)")

            # Step 3c: Create Knowledge Graph Tables
            logging.info("--- Step 3c: Creating Knowledge Graph tables ---")
            
            # RAG.Entities
            create_entities_table = """
            CREATE TABLE RAG.Entities (
                entity_id VARCHAR(255) PRIMARY KEY,
                entity_name VARCHAR(500) NOT NULL,
                entity_type VARCHAR(100),
                description TEXT,
                source_doc_id VARCHAR(255),
                embedding VECTOR(FLOAT, 384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_entities_table)
            logging.info("✅ Created RAG.Entities table")

            # RAG.EntityRelationships
            create_entity_relationships_table = """
            CREATE TABLE RAG.EntityRelationships (
                relationship_id VARCHAR(255) PRIMARY KEY,
                source_entity_id VARCHAR(255),
                target_entity_id VARCHAR(255),
                relationship_type VARCHAR(100),
                description TEXT,
                strength DOUBLE DEFAULT 1.0,
                source_doc_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES RAG.Entities(entity_id),
                FOREIGN KEY (target_entity_id) REFERENCES RAG.Entities(entity_id)
            )
            """
            cursor.execute(create_entity_relationships_table)
            logging.info("✅ Created RAG.EntityRelationships table")

            # RAG.DocumentEntities
            create_document_entities_table = """
            CREATE TABLE RAG.DocumentEntities (
                document_id VARCHAR(255) NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                PRIMARY KEY (document_id, entity_id),
                FOREIGN KEY (entity_id) REFERENCES RAG.Entities(entity_id),
                FOREIGN KEY (document_id) REFERENCES RAG.SourceDocuments(doc_id)
            )
            """
            cursor.execute(create_document_entities_table)
            logging.info("✅ Created RAG.DocumentEntities table")

            # RAG.KnowledgeGraphNodes
            create_kg_nodes_table = """
            CREATE TABLE RAG.KnowledgeGraphNodes (
                node_id VARCHAR(255) PRIMARY KEY,
                node_type VARCHAR(100),
                content TEXT,
                embedding VECTOR(FLOAT, 384),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_kg_nodes_table)
            logging.info("✅ Created RAG.KnowledgeGraphNodes table")

            # RAG.KnowledgeGraphEdges
            create_kg_edges_table = """
            CREATE TABLE RAG.KnowledgeGraphEdges (
                edge_id VARCHAR(255) PRIMARY KEY,
                source_node_id VARCHAR(255),
                target_node_id VARCHAR(255),
                edge_type VARCHAR(100),
                weight DOUBLE DEFAULT 1.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id),
                FOREIGN KEY (target_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id)
            )
            """
            cursor.execute(create_kg_edges_table)
            logging.info("✅ Created RAG.KnowledgeGraphEdges table")
            
            # Step 4: Create indexes
            logging.info("--- Step 4: Creating indexes ---")
            
            indexes = [
                "CREATE INDEX idx_source_docs_id ON RAG.SourceDocuments (doc_id)",
                "CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks (doc_id)",
                "CREATE INDEX idx_chunks_type ON RAG.DocumentChunks (chunk_type)",
                "CREATE INDEX idx_token_embeddings_doc_id ON RAG.DocumentTokenEmbeddings (doc_id)",
                "CREATE INDEX idx_token_embeddings_doc_token_idx ON RAG.DocumentTokenEmbeddings (doc_id, token_index)",
                # Indexes for Graph Tables
                "CREATE INDEX idx_entities_name ON RAG.Entities (entity_name)",
                "CREATE INDEX idx_entities_type ON RAG.Entities (entity_type)",
                "CREATE INDEX idx_entityrelationships_source ON RAG.EntityRelationships (source_entity_id)",
                "CREATE INDEX idx_entityrelationships_target ON RAG.EntityRelationships (target_entity_id)",
                "CREATE INDEX idx_entityrelationships_type ON RAG.EntityRelationships (relationship_type)",
                "CREATE INDEX idx_kgnodes_type ON RAG.KnowledgeGraphNodes (node_type)",
                "CREATE INDEX idx_kgedges_source ON RAG.KnowledgeGraphEdges (source_node_id)",
                "CREATE INDEX idx_kgedges_target ON RAG.KnowledgeGraphEdges (target_node_id)"
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
                "CREATE INDEX idx_hnsw_chunk_embedding ON RAG.DocumentChunks (chunk_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
                "CREATE INDEX idx_hnsw_token_embedding ON RAG.DocumentTokenEmbeddings (token_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
                # HNSW Indexes for Graph Tables
                "CREATE INDEX idx_hnsw_entities_embedding ON RAG.Entities (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
                "CREATE INDEX idx_hnsw_kgnodes_embedding ON RAG.KnowledgeGraphNodes (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')"
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
                SELECT doc_id, VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS similarity
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

            # Verify table emptiness for graph tables
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
                entity_count = cursor.fetchone()[0]
                logging.info(f"VERIFICATION: RAG.Entities count after creation: {entity_count}")
                cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
                rel_count = cursor.fetchone()[0]
                logging.info(f"VERIFICATION: RAG.EntityRelationships count after creation: {rel_count}")
                if entity_count != 0 or rel_count != 0:
                    logging.error("❌ VERIFICATION FAILED: Graph tables are not empty after schema recreation!")
            except Exception as ve:
                logging.error(f"❌ VERIFICATION FAILED: Could not query graph table counts: {ve}")
            
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