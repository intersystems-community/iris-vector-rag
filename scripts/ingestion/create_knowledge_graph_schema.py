#!/usr/bin/env python3
"""
Create knowledge graph schema and populate it for GraphRAG.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # Corrected path to project root

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_knowledge_graph_schema():
    """Create knowledge graph tables for GraphRAG"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        logger.info("🚀 Creating knowledge graph schema for GraphRAG...")

        # Create RAG schema if it doesn't exist (though it should by now)
        try:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS RAG")
            logger.info("✅ Schema RAG ensured")
        except Exception as e_schema:
            logger.warning(f"⚠️ Could not explicitly create/ensure RAG schema (may already exist or not supported): {e_schema}")

        # 1. Create Entities table
        logger.info("📊 Creating RAG.Entities table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.Entities (
                entity_id VARCHAR(255) PRIMARY KEY,
                entity_name VARCHAR(500) NOT NULL,
                entity_type VARCHAR(100),
                description TEXT,
                source_doc_id VARCHAR(255),
                embedding VECTOR(DOUBLE, 384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Created RAG.Entities table")
        
        # 2. Create EntityRelationships table
        logger.info("📊 Creating RAG.EntityRelationships table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.EntityRelationships (
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
        """)
        logger.info("✅ Created RAG.EntityRelationships table")

        # 3. Create DocumentEntities table (as requested by user)
        # This table links documents to entities.
        logger.info("📊 Creating RAG.DocumentEntities table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.DocumentEntities (
                document_id VARCHAR(255) NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                PRIMARY KEY (document_id, entity_id),
                FOREIGN KEY (entity_id) REFERENCES RAG.Entities(entity_id),
                FOREIGN KEY (document_id) REFERENCES RAG.SourceDocuments(doc_id)
            )
        """)
        logger.info("✅ Created RAG.DocumentEntities table")
        
        # 4. Create KnowledgeGraphNodes table (for NodeRAG compatibility, under RAG schema)
        logger.info("📊 Creating RAG.KnowledgeGraphNodes table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.KnowledgeGraphNodes (
                node_id VARCHAR(255) PRIMARY KEY,
                node_type VARCHAR(100),
                content TEXT,
                embedding VECTOR(DOUBLE, 384),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Created RAG.KnowledgeGraphNodes table")
        
        # 5. Create KnowledgeGraphEdges table (for NodeRAG compatibility, under RAG schema)
        logger.info("📊 Creating RAG.KnowledgeGraphEdges table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.KnowledgeGraphEdges (
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
        """)
        logger.info("✅ Created RAG.KnowledgeGraphEdges table")
        
        # 6. Populate with sample data from SourceDocuments
        # Assuming SourceDocuments is in RAG schema.
        logger.info("📊 Populating sample entities from documents (assuming RAG.SourceDocuments)...")
        
        sample_entity_count = 0
        sample_relationship_count = 0
        sample_doc_entity_count = 0

        # Try RAG.SourceDocuments
        source_doc_table_options = ["RAG.SourceDocuments"]
        docs = []
        source_table_used = ""

        for table_option in source_doc_table_options:
            try:
                cursor.execute(f"SELECT TOP 10 doc_id, text_content FROM {table_option} WHERE text_content IS NOT NULL")
                docs = cursor.fetchall()
                if docs:
                    source_table_used = table_option
                    logger.info(f"Found sample documents in {source_table_used}")
                    break
            except Exception:
                logger.warning(f"Could not query {table_option}, trying next option.")
        
        if not docs:
            logger.warning("⚠️ No SourceDocuments found in RAG.SourceDocuments. Sample data population will be limited.")
        else:
            for doc_id, raw_content in docs:
                content_str = ""
                if hasattr(raw_content, 'read'):  # Check if it's a Java-style InputStream
                    try:
                        byte_list = []
                        while True:
                            byte_val = raw_content.read()
                            if byte_val == -1:
                                break
                            byte_list.append(byte_val)
                        if byte_list:
                            content_bytes = bytes(byte_list)
                            content_str = content_bytes.decode('utf-8', errors='replace')
                        else:
                            content_str = ""
                    except Exception as e_read:
                        logger.warning(f"Could not read content stream for doc_id {doc_id}: {e_read}", exc_info=True)
                        continue
                elif isinstance(raw_content, str):
                    content_str = raw_content
                elif isinstance(raw_content, bytes):
                    try:
                        content_str = raw_content.decode('utf-8', errors='replace')
                    except Exception as e_decode:
                        logger.warning(f"Could not decode bytes content for doc_id {doc_id}: {e_decode}", exc_info=True)
                        continue
                elif raw_content is None:
                    content_str = ""
                else:
                    logger.warning(f"Unexpected content type for doc_id {doc_id}: {type(raw_content)}. Value: '{str(raw_content)[:100]}'. Skipping.")
                    continue

                if content_str and len(content_str) > 50:
                    words = content_str.split()[:20]
                    
                    doc_entities_created_for_this_doc = []
                    for i, word in enumerate(words):
                        if len(word) > 3 and word.isalpha():
                            entity_id = f"entity_{doc_id}_{i}"
                            # Generate a dummy embedding string for now
                            dummy_embedding_str = "[0.1" + ",0.0" * 383 + "]"
                            try:
                                cursor.execute("""
                                    INSERT INTO RAG.Entities (entity_id, entity_name, entity_type, source_doc_id, embedding)
                                    VALUES (?, ?, ?, ?, TO_VECTOR(?))
                                """, (entity_id, word, "TERM", doc_id, dummy_embedding_str))
                                sample_entity_count += 1
                                doc_entities_created_for_this_doc.append(entity_id)

                                try:
                                    cursor.execute("""
                                        INSERT INTO RAG.DocumentEntities (document_id, entity_id)
                                        VALUES (?, ?)
                                    """, (doc_id, entity_id))
                                    sample_doc_entity_count +=1
                                except Exception:
                                    pass
                            except Exception:
                                pass
            
            logger.info(f"✅ Created {sample_entity_count} sample entities.")
            logger.info(f"✅ Created {sample_doc_entity_count} sample document-entity mappings.")

            logger.info("📊 Creating sample relationships...")
            cursor.execute("SELECT TOP 5 entity_id FROM RAG.Entities ORDER BY created_at DESC")
            created_entities = cursor.fetchall()
            
            if len(created_entities) > 1:
                for i in range(len(created_entities) - 1):
                    source_entity_id = created_entities[i][0]
                    target_entity_id = created_entities[i + 1][0]
                    rel_id = f"rel_sample_{i}"
                    
                    try:
                        cursor.execute("""
                            INSERT INTO RAG.EntityRelationships (relationship_id, source_entity_id, target_entity_id, relationship_type)
                            VALUES (?, ?, ?, ?)
                        """, (rel_id, source_entity_id, target_entity_id, "RELATED_TO_SAMPLE"))
                        sample_relationship_count += 1
                    except Exception:
                        pass
            logger.info(f"✅ Created {sample_relationship_count} sample relationships.")

        # 7. Verify the schema
        logger.info("🧪 Verifying RAG schema (for KG tables)...")
        
        tables_to_check = ['Entities', 'EntityRelationships', 'DocumentEntities', 'KnowledgeGraphNodes', 'KnowledgeGraphEdges']
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM RAG.{table}")
                count = cursor.fetchone()[0]
                logger.info(f"✅ RAG.{table}: {count:,} rows")
            except Exception as e:
                logger.error(f"❌ RAG.{table}: {e}")
        
        logger.info("🎉 RAG schema (for KG tables) setup completed!")
        
    except Exception as e:
        logger.error(f"❌ Error creating knowledge graph schema: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    create_knowledge_graph_schema()