#!/usr/bin/env python3
"""
Create knowledge graph schema and populate it for GraphRAG.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_knowledge_graph_schema():
    """Create knowledge graph tables for GraphRAG"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        logger.info("🚀 Creating knowledge graph schema for GraphRAG...")
        
        # 1. Create Entities table
        logger.info("📊 Creating Entities table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.Entities (
                entity_id VARCHAR(255) PRIMARY KEY,
                entity_name VARCHAR(500) NOT NULL,
                entity_type VARCHAR(100),
                description TEXT,
                source_doc_id VARCHAR(255),
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Created Entities table")
        
        # 2. Create Relationships table
        logger.info("📊 Creating Relationships table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.Relationships (
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
        logger.info("✅ Created Relationships table")
        
        # 3. Create KnowledgeGraphNodes table (for NodeRAG compatibility)
        logger.info("📊 Creating KnowledgeGraphNodes table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RAG.KnowledgeGraphNodes (
                node_id VARCHAR(255) PRIMARY KEY,
                node_type VARCHAR(100),
                content TEXT,
                embedding TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Created KnowledgeGraphNodes table")
        
        # 4. Create KnowledgeGraphEdges table (for NodeRAG compatibility)
        logger.info("📊 Creating KnowledgeGraphEdges table...")
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
        logger.info("✅ Created KnowledgeGraphEdges table")
        
        # 5. Populate with sample data from SourceDocuments
        logger.info("📊 Populating sample entities from documents...")
        
        # Get some sample documents
        cursor.execute("SELECT TOP 10 doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL")
        docs = cursor.fetchall()
        
        entity_count = 0
        for doc_id, content in docs:
            if content and len(content) > 50:
                # Extract simple entities (this is a basic implementation)
                # In a real system, you'd use NER (Named Entity Recognition)
                words = content.split()[:20]  # First 20 words
                
                for i, word in enumerate(words):
                    if len(word) > 3 and word.isalpha():  # Simple entity detection
                        entity_id = f"entity_{doc_id}_{i}"
                        try:
                            cursor.execute("""
                                INSERT INTO RAG.Entities (entity_id, entity_name, entity_type, source_doc_id)
                                VALUES (?, ?, ?, ?)
                            """, (entity_id, word, "TERM", doc_id))
                            entity_count += 1
                        except:
                            pass  # Skip duplicates
        
        logger.info(f"✅ Created {entity_count} sample entities")
        
        # 6. Create some sample relationships
        logger.info("📊 Creating sample relationships...")
        cursor.execute("SELECT TOP 5 entity_id, source_doc_id FROM RAG.Entities")
        entities = cursor.fetchall()
        
        relationship_count = 0
        for i in range(len(entities) - 1):
            source_entity = entities[i][0]
            target_entity = entities[i + 1][0]
            rel_id = f"rel_{i}"
            
            try:
                cursor.execute("""
                    INSERT INTO RAG.Relationships (relationship_id, source_entity_id, target_entity_id, relationship_type)
                    VALUES (?, ?, ?, ?)
                """, (rel_id, source_entity, target_entity, "RELATED_TO"))
                relationship_count += 1
            except:
                pass
        
        logger.info(f"✅ Created {relationship_count} sample relationships")
        
        # 7. Verify the schema
        logger.info("🧪 Verifying knowledge graph schema...")
        
        tables_to_check = ['Entities', 'Relationships', 'KnowledgeGraphNodes', 'KnowledgeGraphEdges']
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM RAG.{table}")
                count = cursor.fetchone()[0]
                logger.info(f"✅ RAG.{table}: {count:,} rows")
            except Exception as e:
                logger.error(f"❌ RAG.{table}: {e}")
        
        logger.info("🎉 Knowledge graph schema creation completed!")
        
    except Exception as e:
        logger.error(f"❌ Error creating knowledge graph schema: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    create_knowledge_graph_schema()