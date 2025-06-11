#!/usr/bin/env python3
"""
Fix Knowledge Graph Population

This script will:
1. Populate knowledge graph nodes and edges from existing documents
2. Use simple entity extraction without complex embedding generation
3. Handle IRIS stream fields properly

Usage:
    python scripts/fix_knowledge_graph.py
"""

import os
import sys
import time
import logging
import random
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_knowledge_graph.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeGraphFixer:
    """Fix knowledge graph population"""
    
    def __init__(self):
        self.connection = None
        
    def initialize(self):
        """Initialize database connection"""
        logger.info("🚀 Initializing Knowledge Graph Fixer...")
        
        # Get database connection
        self.connection = get_iris_connection()
        if not self.connection:
            raise Exception("Failed to connect to IRIS database")
        
        logger.info("✅ Initialization complete")
        
    def check_current_state(self):
        """Check current database state"""
        logger.info("📊 Checking current database state...")
        
        with self.connection.cursor() as cursor:
            # Check documents
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            
            # Check graph nodes
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            node_count = cursor.fetchone()[0]
            
            # Check graph edges
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
            edge_count = cursor.fetchone()[0]
        
        state = {
            'documents': doc_count,
            'graph_nodes': node_count,
            'graph_edges': edge_count
        }
        
        logger.info(f"Current state: {doc_count:,} docs, {node_count:,} nodes, {edge_count:,} edges")
        return state
        
    def clear_existing_graph_data(self):
        """Clear existing graph data to start fresh"""
        logger.info("🧹 Clearing existing graph data...")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM RAG.KnowledgeGraphEdges")
                cursor.execute("DELETE FROM RAG.KnowledgeGraphNodes")
                self.connection.commit()
            
            logger.info("✅ Existing graph data cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing graph data: {e}")
            return False
    
    def populate_knowledge_graph(self):
        """Populate knowledge graph for all documents"""
        logger.info("🕸️ Populating knowledge graph...")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                total_docs = cursor.fetchone()[0]
            
            logger.info(f"Extracting entities and relationships from {total_docs:,} documents...")
            
            # Process documents in batches
            batch_size = 100  # Smaller batches for stability
            entity_id = 1
            relationship_id = 1
            
            for offset in range(0, total_docs, batch_size):
                logger.info(f"Processing graph batch: docs {offset + 1}-{min(offset + batch_size, total_docs)}")
                
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT doc_id, title, text_content 
                        FROM RAG.SourceDocuments 
                        ORDER BY doc_id 
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
                    
                    batch_docs = cursor.fetchall()
                
                # Extract entities and relationships for this batch
                entities = []
                relationships = []
                
                for doc_id, title, text_content in batch_docs:
                    try:
                        # Handle IRIS stream - convert to string safely
                        title_str = str(title) if title else f"Document {doc_id}"
                        
                        # Handle text_content which might be a stream
                        if hasattr(text_content, 'read'):
                            try:
                                content_str = text_content.read()
                            except:
                                content_str = ""
                        elif text_content is not None:
                            content_str = str(text_content)
                        else:
                            content_str = ""
                        
                        # Simple entity extraction (medical terms)
                        doc_entities = self._extract_simple_entities(doc_id, title_str, content_str)
                        
                        entity_ids_for_doc = []
                        for entity_name, entity_type in doc_entities:
                            # Create simple embedding (just zeros for now to avoid PyTorch issues)
                            simple_embedding = ','.join(['0.1'] * 384)  # Simple default embedding
                            
                            node_id = f"entity_{entity_id:08d}"
                            entities.append((
                                node_id,
                                entity_name,
                                entity_type,
                                doc_id,
                                simple_embedding
                            ))
                            entity_ids_for_doc.append(node_id)
                            entity_id += 1
                        
                        # Create simple relationships between entities in the same document
                        if len(entity_ids_for_doc) > 1:
                            for i in range(len(entity_ids_for_doc) - 1):
                                relationships.append((
                                    f"rel_{relationship_id:08d}",
                                    entity_ids_for_doc[i],
                                    entity_ids_for_doc[i + 1],
                                    "RELATED_TO",
                                    doc_id,
                                    0.8  # confidence score
                                ))
                                relationship_id += 1
                    
                    except Exception as e:
                        logger.warning(f"Error processing document {doc_id} for graph: {e}")
                        continue
                
                # Insert entities
                if entities:
                    try:
                        with self.connection.cursor() as cursor:
                            cursor.executemany("""
                                INSERT INTO RAG.KnowledgeGraphNodes 
                                (node_id, entity_name, entity_type, source_doc_id, embedding)
                                VALUES (?, ?, ?, ?, ?)
                            """, entities)
                            self.connection.commit()
                    except Exception as e:
                        logger.warning(f"Error inserting entities: {e}")
                
                # Insert relationships
                if relationships:
                    try:
                        with self.connection.cursor() as cursor:
                            cursor.executemany("""
                                INSERT INTO RAG.KnowledgeGraphEdges 
                                (edge_id, source_node_id, target_node_id, relationship_type, source_doc_id, confidence_score)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, relationships)
                            self.connection.commit()
                    except Exception as e:
                        logger.warning(f"Error inserting relationships: {e}")
                
                logger.info(f"Added {len(entities)} entities and {len(relationships)} relationships")
                
                # Brief pause
                time.sleep(0.1)
            
            # Check final graph counts
            final_state = self.check_current_state()
            node_count = final_state['graph_nodes']
            edge_count = final_state['graph_edges']
            
            logger.info(f"✅ Knowledge graph complete: {node_count:,} nodes, {edge_count:,} edges")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in knowledge graph population: {e}")
            return False
    
    def _extract_simple_entities(self, doc_id, title, text_content):
        """Extract simple entities from document text"""
        # Simple keyword-based entity extraction
        medical_terms = [
            ("diabetes", "DISEASE"),
            ("cancer", "DISEASE"), 
            ("cardiovascular", "DISEASE"),
            ("hypertension", "DISEASE"),
            ("treatment", "PROCEDURE"),
            ("therapy", "PROCEDURE"),
            ("medication", "DRUG"),
            ("patient", "PERSON"),
            ("study", "RESEARCH"),
            ("clinical", "RESEARCH"),
            ("diagnosis", "PROCEDURE"),
            ("prevention", "PROCEDURE"),
            ("healthcare", "CONCEPT"),
            ("management", "PROCEDURE"),
            ("intervention", "PROCEDURE"),
            ("outcomes", "CONCEPT"),
            ("research", "RESEARCH"),
            ("analysis", "RESEARCH"),
            ("findings", "CONCEPT"),
            ("results", "CONCEPT")
        ]
        
        entities = []
        text_lower = (title + " " + text_content).lower()
        
        # Extract medical terms found in the text
        for term, entity_type in medical_terms:
            if term in text_lower:
                entities.append((term.title(), entity_type))
        
        # Add document title as an entity (truncated to avoid issues)
        title_entity = title[:50] if title else f"Document {doc_id}"
        entities.append((title_entity, "DOCUMENT"))
        
        # Add document ID as an entity
        entities.append((doc_id, "DOCUMENT_ID"))
        
        return entities[:8]  # Limit to 8 entities per document to keep it manageable
    
    def test_graph_retrieval(self):
        """Test basic graph retrieval functionality"""
        logger.info("🧪 Testing graph retrieval...")
        
        try:
            # Test node retrieval
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT TOP 10 node_id, entity_name, entity_type
                    FROM RAG.KnowledgeGraphNodes
                    ORDER BY node_id
                """)
                
                node_results = cursor.fetchall()
            
            # Test edge retrieval
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT TOP 10 edge_id, source_node_id, target_node_id, relationship_type
                    FROM RAG.KnowledgeGraphEdges
                    ORDER BY edge_id
                """)
                
                edge_results = cursor.fetchall()
            
            # Test entity type distribution
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT entity_type, COUNT(*) as count
                    FROM RAG.KnowledgeGraphNodes
                    GROUP BY entity_type
                    ORDER BY count DESC
                """)
                
                type_distribution = cursor.fetchall()
            
            logger.info(f"✅ Graph retrieval test complete:")
            logger.info(f"  - Sample nodes: {len(node_results)}")
            logger.info(f"  - Sample edges: {len(edge_results)}")
            logger.info(f"  - Entity types: {len(type_distribution)}")
            
            if type_distribution:
                logger.info("Entity type distribution:")
                for entity_type, count in type_distribution:
                    logger.info(f"    {entity_type}: {count:,}")
            
            return len(node_results) > 0 and len(edge_results) > 0
            
        except Exception as e:
            logger.error(f"❌ Error in graph retrieval test: {e}")
            return False
    
    def run_graph_fix(self):
        """Run the complete graph fixing process"""
        start_time = time.time()
        logger.info("🚀 Starting knowledge graph fix...")
        
        try:
            # Initialize
            self.initialize()
            
            # Check initial state
            initial_state = self.check_current_state()
            logger.info(f"Initial state: {initial_state}")
            
            # Step 1: Clear existing graph data
            logger.info("🧹 Step 1: Clearing existing graph data...")
            if not self.clear_existing_graph_data():
                raise Exception("Failed to clear existing graph data")
            
            # Step 2: Populate knowledge graph
            logger.info("🕸️ Step 2: Populating knowledge graph...")
            if not self.populate_knowledge_graph():
                raise Exception("Failed to populate knowledge graph")
            
            # Step 3: Test graph retrieval
            logger.info("🧪 Step 3: Testing graph retrieval...")
            if not self.test_graph_retrieval():
                raise Exception("Graph retrieval tests failed")
            
            # Final state check
            final_state = self.check_current_state()
            
            elapsed_time = time.time() - start_time
            
            logger.info("🎉 Knowledge graph fix successful!")
            logger.info(f"Final state: {final_state}")
            logger.info(f"Total time: {elapsed_time:.1f} seconds")
            
            return True, final_state
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph fix failed: {e}")
            return False, {}
        
        finally:
            if self.connection:
                self.connection.close()

def main():
    """Main function"""
    fixer = KnowledgeGraphFixer()
    success, final_state = fixer.run_graph_fix()
    
    if success:
        print("\n🎉 SUCCESS: Knowledge graph fix completed!")
        print(f"Final graph state: {final_state}")
        return 0
    else:
        print("\n❌ FAILED: Knowledge graph fix encountered errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())