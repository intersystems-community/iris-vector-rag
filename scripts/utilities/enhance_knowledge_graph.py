#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Population

This script will:
1. Extract more comprehensive entities from document content
2. Create richer relationships between entities
3. Add semantic embeddings for better graph traversal
4. Populate with medical domain knowledge

Usage:
    python scripts/enhance_knowledge_graph.py
"""

import os
import sys
import time
import logging
import re
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhance_knowledge_graph.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraphPopulator:
    """Enhanced knowledge graph population with rich entity extraction"""
    
    def __init__(self):
        self.connection = None
        self.embedding_func = None
        
        # Enhanced medical entity patterns
        self.entity_patterns = {
            'DISEASE': [
                r'\b(cancer|carcinoma|tumor|malignancy)\b',
                r'\b(diabetes|diabetic)\b',
                r'\b(hypertension|high blood pressure)\b',
                r'\b(covid|coronavirus|sars-cov-2)\b',
                r'\b(alzheimer|dementia)\b',
                r'\b(depression|anxiety)\b',
                r'\b(asthma|copd)\b',
                r'\b(arthritis|osteoarthritis)\b',
                r'\b(stroke|cerebrovascular)\b',
                r'\b(heart disease|cardiovascular)\b'
            ],
            'TREATMENT': [
                r'\b(chemotherapy|radiation|surgery)\b',
                r'\b(medication|drug|pharmaceutical)\b',
                r'\b(therapy|treatment|intervention)\b',
                r'\b(vaccine|vaccination|immunization)\b',
                r'\b(rehabilitation|physiotherapy)\b',
                r'\b(counseling|psychotherapy)\b'
            ],
            'ANATOMY': [
                r'\b(brain|heart|lung|liver|kidney)\b',
                r'\b(blood|plasma|serum)\b',
                r'\b(cell|tissue|organ)\b',
                r'\b(gene|dna|rna|protein)\b',
                r'\b(muscle|bone|nerve)\b'
            ],
            'RESEARCH': [
                r'\b(study|trial|research|investigation)\b',
                r'\b(clinical trial|randomized)\b',
                r'\b(meta-analysis|systematic review)\b',
                r'\b(cohort|case-control)\b',
                r'\b(biomarker|endpoint)\b'
            ],
            'MEASUREMENT': [
                r'\b(\d+\s*mg|\d+\s*ml|\d+\s*%)\b',
                r'\b(p-value|confidence interval|odds ratio)\b',
                r'\b(sensitivity|specificity|accuracy)\b',
                r'\b(prevalence|incidence|mortality)\b'
            ]
        }
        
    def initialize(self):
        """Initialize connections and functions"""
        logger.info("🚀 Initializing Enhanced Knowledge Graph Populator...")
        
        # Get database connection
        self.connection = get_iris_connection()
        if not self.connection:
            raise Exception("Failed to connect to IRIS database")
        
        # Get embedding function
        self.embedding_func = get_embedding_func()
        
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
        
    def clear_and_rebuild_graph(self):
        """Clear existing graph and rebuild with enhanced data"""
        logger.info("🧹 Clearing existing graph for enhanced rebuild...")
        
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
    
    def populate_enhanced_knowledge_graph(self):
        """Populate knowledge graph with enhanced entity extraction"""
        logger.info("🕸️ Populating enhanced knowledge graph...")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                total_docs = cursor.fetchone()[0]
            
            logger.info(f"Processing {total_docs:,} documents with enhanced extraction...")
            
            # Process documents in batches
            batch_size = 25  # Smaller batches for more intensive processing
            node_id = 1
            edge_id = 1
            
            for offset in range(0, total_docs, batch_size):
                logger.info(f"Processing enhanced batch: docs {offset + 1}-{min(offset + batch_size, total_docs)}")
                
                # Get document chunks for richer content
                with self.connection.cursor() as cursor:
                    # First get the document IDs for this batch
                    cursor.execute("""
                        SELECT doc_id FROM RAG.SourceDocuments
                        ORDER BY doc_id
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
                    
                    doc_ids = [row[0] for row in cursor.fetchall()]
                    
                    if not doc_ids:
                        continue
                    
                    # Create placeholders for the IN clause
                    placeholders = ','.join(['?' for _ in doc_ids])
                    
                    # Now get documents and chunks
                    cursor.execute(f"""
                        SELECT DISTINCT s.doc_id, s.title, c.chunk_text
                        FROM RAG.SourceDocuments s
                        LEFT JOIN RAG.DocumentChunks c ON s.doc_id = c.doc_id
                        WHERE s.doc_id IN ({placeholders})
                        ORDER BY s.doc_id, c.chunk_index
                    """, doc_ids)
                    
                    batch_data = cursor.fetchall()
                
                # Group by document
                doc_data = {}
                for doc_id, title, chunk_text in batch_data:
                    if doc_id not in doc_data:
                        doc_data[doc_id] = {'title': title, 'chunks': []}
                    if chunk_text:
                        # Handle IRIS streams
                        if hasattr(chunk_text, 'read'):
                            try:
                                chunk_str = chunk_text.read()
                            except:
                                chunk_str = ""
                        else:
                            chunk_str = str(chunk_text) if chunk_text else ""
                        
                        if chunk_str and len(chunk_str.strip()) > 20:
                            doc_data[doc_id]['chunks'].append(chunk_str)
                
                # Extract entities and relationships for this batch
                nodes = []
                edges = []
                
                for doc_id, data in doc_data.items():
                    try:
                        title_str = str(data['title']) if data['title'] else f"Document {doc_id}"
                        all_text = title_str + " " + " ".join(data['chunks'])
                        
                        # Enhanced entity extraction
                        doc_entities = self._extract_enhanced_entities(doc_id, title_str, all_text)
                        
                        node_ids_for_doc = []
                        entity_groups = {}  # Group entities by type for better relationships
                        
                        for entity_content, entity_type, confidence in doc_entities:
                            # Create semantic embedding for the entity
                            try:
                                entity_embedding = self.embedding_func(entity_content)
                                entity_embedding_str = ','.join(map(str, entity_embedding))
                            except:
                                entity_embedding_str = ','.join(['0.1'] * 384)
                            
                            current_node_id = f"node_{node_id:08d}"
                            
                            # Enhanced metadata
                            metadata = {
                                "source_doc": doc_id,
                                "confidence": confidence,
                                "extraction_method": "enhanced_pattern_matching",
                                "created_at": datetime.now().isoformat()
                            }
                            
                            nodes.append((
                                current_node_id,
                                entity_content,
                                entity_type,
                                entity_embedding_str,
                                str(metadata).replace("'", '"')
                            ))
                            
                            node_ids_for_doc.append(current_node_id)
                            
                            # Group by type for relationship creation
                            if entity_type not in entity_groups:
                                entity_groups[entity_type] = []
                            entity_groups[entity_type].append(current_node_id)
                            
                            node_id += 1
                        
                        # Create enhanced relationships
                        relationships = self._create_enhanced_relationships(
                            entity_groups, doc_id, edge_id
                        )
                        edges.extend(relationships)
                        edge_id += len(relationships)
                    
                    except Exception as e:
                        logger.warning(f"Error processing document {doc_id}: {e}")
                        continue
                
                # Insert nodes
                if nodes:
                    try:
                        with self.connection.cursor() as cursor:
                            cursor.executemany("""
                                INSERT INTO RAG.KnowledgeGraphNodes 
                                (node_id, content, node_type, embedding, metadata)
                                VALUES (?, ?, ?, ?, ?)
                            """, nodes)
                            self.connection.commit()
                    except Exception as e:
                        logger.warning(f"Error inserting nodes: {e}")
                
                # Insert edges
                if edges:
                    try:
                        with self.connection.cursor() as cursor:
                            cursor.executemany("""
                                INSERT INTO RAG.KnowledgeGraphEdges 
                                (edge_id, source_node_id, target_node_id, edge_type, weight)
                                VALUES (?, ?, ?, ?, ?)
                            """, edges)
                            self.connection.commit()
                    except Exception as e:
                        logger.warning(f"Error inserting edges: {e}")
                
                logger.info(f"Added {len(nodes)} enhanced nodes and {len(edges)} relationships")
                
                # Brief pause
                time.sleep(0.2)
            
            # Check final graph counts
            final_state = self.check_current_state()
            node_count = final_state['graph_nodes']
            edge_count = final_state['graph_edges']
            
            logger.info(f"✅ Enhanced knowledge graph complete: {node_count:,} nodes, {edge_count:,} edges")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced knowledge graph population: {e}")
            return False
    
    def _extract_enhanced_entities(self, doc_id, title, text):
        """Extract enhanced entities using pattern matching and NLP techniques"""
        entities = []
        text_lower = text.lower()
        
        # Extract entities by type using regex patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 2:
                        # Calculate confidence based on context
                        confidence = self._calculate_entity_confidence(entity_text, text_lower, entity_type)
                        entities.append((entity_text.title(), entity_type, confidence))
        
        # Add document-level entities
        entities.append((title[:100], "DOCUMENT", 1.0))
        entities.append((doc_id, "DOCUMENT_ID", 1.0))
        
        # Extract key phrases (simple approach)
        key_phrases = self._extract_key_phrases(text)
        for phrase in key_phrases:
            entities.append((phrase, "KEY_PHRASE", 0.7))
        
        # Remove duplicates and sort by confidence
        unique_entities = {}
        for content, etype, conf in entities:
            key = (content.lower(), etype)
            if key not in unique_entities or unique_entities[key][2] < conf:
                unique_entities[key] = (content, etype, conf)
        
        return sorted(unique_entities.values(), key=lambda x: x[2], reverse=True)[:15]
    
    def _calculate_entity_confidence(self, entity, text, entity_type):
        """Calculate confidence score for entity extraction"""
        # Base confidence
        confidence = 0.5
        
        # Boost confidence based on frequency
        frequency = text.count(entity.lower())
        confidence += min(frequency * 0.1, 0.3)
        
        # Boost confidence based on context
        if entity_type == "DISEASE" and any(word in text for word in ["patient", "treatment", "diagnosis"]):
            confidence += 0.2
        elif entity_type == "TREATMENT" and any(word in text for word in ["therapy", "medication", "intervention"]):
            confidence += 0.2
        elif entity_type == "RESEARCH" and any(word in text for word in ["study", "trial", "analysis"]):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_key_phrases(self, text):
        """Extract key phrases from text (simple approach)"""
        # Simple key phrase extraction based on common medical patterns
        phrases = []
        
        # Look for noun phrases with medical relevance
        medical_adjectives = ["clinical", "medical", "therapeutic", "diagnostic", "preventive"]
        medical_nouns = ["study", "trial", "treatment", "therapy", "intervention", "outcome"]
        
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] in medical_adjectives and words[i+1] in medical_nouns:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        return phrases[:5]  # Limit to top 5 phrases
    
    def _create_enhanced_relationships(self, entity_groups, doc_id, start_edge_id):
        """Create enhanced relationships between entities"""
        relationships = []
        edge_id = start_edge_id
        
        # Define relationship types and weights
        relationship_rules = [
            ("DISEASE", "TREATMENT", "TREATED_BY", 0.9),
            ("DISEASE", "ANATOMY", "AFFECTS", 0.8),
            ("TREATMENT", "MEASUREMENT", "MEASURED_BY", 0.7),
            ("RESEARCH", "DISEASE", "STUDIES", 0.8),
            ("RESEARCH", "TREATMENT", "EVALUATES", 0.8),
            ("DOCUMENT", "DISEASE", "DISCUSSES", 0.6),
            ("DOCUMENT", "TREATMENT", "DESCRIBES", 0.6),
        ]
        
        # Create relationships based on rules
        for source_type, target_type, rel_type, weight in relationship_rules:
            if source_type in entity_groups and target_type in entity_groups:
                for source_node in entity_groups[source_type]:
                    for target_node in entity_groups[target_type]:
                        if source_node != target_node:
                            relationships.append((
                                f"edge_{edge_id:08d}",
                                source_node,
                                target_node,
                                rel_type,
                                weight
                            ))
                            edge_id += 1
        
        # Create co-occurrence relationships within same type
        for entity_type, nodes in entity_groups.items():
            if len(nodes) > 1:
                for i in range(len(nodes) - 1):
                    relationships.append((
                        f"edge_{edge_id:08d}",
                        nodes[i],
                        nodes[i + 1],
                        "CO_OCCURS_WITH",
                        0.5
                    ))
                    edge_id += 1
        
        return relationships
    
    def test_enhanced_graph(self):
        """Test the enhanced graph functionality"""
        logger.info("🧪 Testing enhanced graph...")
        
        try:
            # Test node type distribution
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT node_type, COUNT(*) as count
                    FROM RAG.KnowledgeGraphNodes
                    GROUP BY node_type
                    ORDER BY count DESC
                """)
                
                type_distribution = cursor.fetchall()
            
            # Test relationship types
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT edge_type, COUNT(*) as count
                    FROM RAG.KnowledgeGraphEdges
                    GROUP BY edge_type
                    ORDER BY count DESC
                """)
                
                rel_distribution = cursor.fetchall()
            
            logger.info("✅ Enhanced graph test results:")
            logger.info("Node type distribution:")
            for node_type, count in type_distribution:
                logger.info(f"    {node_type}: {count:,}")
            
            logger.info("Relationship type distribution:")
            for edge_type, count in rel_distribution:
                logger.info(f"    {edge_type}: {count:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing enhanced graph: {e}")
            return False
    
    def run_enhancement(self):
        """Run the complete graph enhancement process"""
        start_time = time.time()
        logger.info("🚀 Starting enhanced knowledge graph population...")
        
        try:
            # Initialize
            self.initialize()
            
            # Check initial state
            initial_state = self.check_current_state()
            logger.info(f"Initial state: {initial_state}")
            
            # Step 1: Clear and rebuild
            logger.info("🧹 Step 1: Clearing existing graph...")
            if not self.clear_and_rebuild_graph():
                raise Exception("Failed to clear existing graph")
            
            # Step 2: Populate enhanced graph
            logger.info("🕸️ Step 2: Populating enhanced knowledge graph...")
            if not self.populate_enhanced_knowledge_graph():
                raise Exception("Failed to populate enhanced graph")
            
            # Step 3: Test enhanced graph
            logger.info("🧪 Step 3: Testing enhanced graph...")
            if not self.test_enhanced_graph():
                logger.warning("Enhanced graph tests had issues, but continuing...")
            
            # Final state check
            final_state = self.check_current_state()
            
            elapsed_time = time.time() - start_time
            
            logger.info("🎉 Enhanced knowledge graph population successful!")
            logger.info(f"Final state: {final_state}")
            logger.info(f"Total time: {elapsed_time:.1f} seconds")
            
            return True, final_state
            
        except Exception as e:
            logger.error(f"❌ Enhanced graph population failed: {e}")
            return False, {}
        
        finally:
            if self.connection:
                self.connection.close()

def main():
    """Main function"""
    populator = EnhancedKnowledgeGraphPopulator()
    success, final_state = populator.run_enhancement()
    
    if success:
        print("\n🎉 SUCCESS: Enhanced knowledge graph population completed!")
        print(f"Final enhanced graph state: {final_state}")
        return 0
    else:
        print("\n❌ FAILED: Enhanced knowledge graph population encountered errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())