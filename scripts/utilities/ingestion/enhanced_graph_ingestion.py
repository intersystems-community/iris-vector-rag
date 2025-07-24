#!/usr/bin/env python3
"""
Enhanced ingestion script that populates GraphRAG tables with entities and relationships.
"""

import sys
import logging
import re
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGIngestionPipeline:
    """Enhanced ingestion pipeline that extracts entities and relationships for GraphRAG"""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.embedding_func = get_embedding_func()
        
    def extract_entities_from_text(self, text: str, doc_id: str) -> List[Dict]:
        """
        Extract entities from text using simple NLP patterns.
        In production, this would use spaCy, NLTK, or a dedicated NER model.
        """
        entities = []
        
        # Simple patterns for medical/scientific entities
        patterns = {
            'DISEASE': r'\b(?:diabetes|cancer|hypertension|asthma|pneumonia|infection|syndrome|disorder)\b',
            'TREATMENT': r'\b(?:therapy|treatment|medication|drug|surgery|procedure|intervention)\b',
            'PROTEIN': r'\b[A-Z][a-z]*[0-9]*\b(?=\s+protein|\s+enzyme|\s+receptor)',
            'GENE': r'\b[A-Z]{2,}[0-9]*\b(?=\s+gene|\s+expression)',
            'CHEMICAL': r'\b(?:insulin|glucose|cortisol|dopamine|serotonin|acetylcholine)\b',
            'ORGAN': r'\b(?:heart|brain|liver|kidney|lung|pancreas|stomach)\b',
            'CELL_TYPE': r'\b(?:neuron|lymphocyte|macrophage|fibroblast|hepatocyte)\b'
        }
        
        entity_id_counter = 0
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().lower()
                entity_id = f"{doc_id}_{entity_type}_{entity_id_counter}"
                
                entities.append({
                    'entity_id': entity_id,
                    'entity_name': entity_name,
                    'entity_type': entity_type,
                    'source_doc_id': doc_id,
                    'description': f"{entity_type.lower()} mentioned in document {doc_id}"
                })
                entity_id_counter += 1
        
        return entities
    
    def extract_relationships_from_entities(self, entities: List[Dict], doc_id: str) -> List[Dict]:
        """
        Extract relationships between entities based on co-occurrence and patterns.
        """
        relationships = []
        relationship_id_counter = 0
        
        # Create relationships between entities of different types
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1['entity_type'] != entity2['entity_type']:
                    # Create relationship based on entity types
                    rel_type = self._determine_relationship_type(
                        entity1['entity_type'], 
                        entity2['entity_type']
                    )
                    
                    if rel_type:
                        rel_id = f"{doc_id}_rel_{relationship_id_counter}"
                        relationships.append({
                            'relationship_id': rel_id,
                            'source_entity_id': entity1['entity_id'],
                            'target_entity_id': entity2['entity_id'],
                            'relationship_type': rel_type,
                            'description': f"{rel_type} relationship in {doc_id}",
                            'strength': 1.0,
                            'source_doc_id': doc_id
                        })
                        relationship_id_counter += 1
        
        return relationships
    
    def _determine_relationship_type(self, type1: str, type2: str) -> str:
        """Determine relationship type based on entity types"""
        relationship_rules = {
            ('DISEASE', 'TREATMENT'): 'TREATED_BY',
            ('DISEASE', 'ORGAN'): 'AFFECTS',
            ('TREATMENT', 'DISEASE'): 'TREATS',
            ('PROTEIN', 'GENE'): 'ENCODED_BY',
            ('CHEMICAL', 'ORGAN'): 'PRODUCED_BY',
            ('DISEASE', 'CHEMICAL'): 'INVOLVES',
            ('TREATMENT', 'CHEMICAL'): 'CONTAINS',
            ('CELL_TYPE', 'ORGAN'): 'PART_OF'
        }
        
        # Check both directions
        key1 = (type1, type2)
        key2 = (type2, type1)
        
        return relationship_rules.get(key1) or relationship_rules.get(key2)
    
    def populate_graph_tables(self, batch_size: int = 1000):
        """
        Populate GraphRAG tables by processing existing documents.
        """
        logger.info("🚀 Starting GraphRAG table population...")
        
        cursor = self.connection.cursor()
        
        try:
            # Get total document count
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE text_content IS NOT NULL")
            total_docs = cursor.fetchone()[0]
            logger.info(f"📊 Processing {total_docs:,} documents for graph extraction")
            
            # Process documents in batches
            processed = 0
            total_entities = 0
            total_relationships = 0
            
            for offset in range(0, total_docs, batch_size):
                logger.info(f"📋 Processing batch {offset//batch_size + 1} (docs {offset+1}-{min(offset+batch_size, total_docs)})")
                
                # Get batch of documents
                cursor.execute("""
                    SELECT doc_id, text_content 
                    FROM RAG.SourceDocuments 
                    WHERE text_content IS NOT NULL 
                    ORDER BY doc_id 
                    LIMIT ? OFFSET ?
                """, (batch_size, offset))
                
                docs = cursor.fetchall()
                
                batch_entities = []
                batch_relationships = []
                
                for doc_id, text_content in docs:
                    if text_content and len(text_content) > 100:  # Skip very short documents
                        # Extract entities
                        entities = self.extract_entities_from_text(text_content, doc_id)
                        batch_entities.extend(entities)
                        
                        # Extract relationships
                        relationships = self.extract_relationships_from_entities(entities, doc_id)
                        batch_relationships.extend(relationships)
                
                # Insert entities
                if batch_entities:
                    for entity in batch_entities:
                        try:
                            cursor.execute("""
                                INSERT INTO RAG.Entities (entity_id, entity_name, entity_type, description, source_doc_id)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                entity['entity_id'],
                                entity['entity_name'],
                                entity['entity_type'],
                                entity['description'],
                                entity['source_doc_id']
                            ))
                        except Exception as e:
                            # Skip duplicates
                            if "duplicate" not in str(e).lower():
                                logger.warning(f"Entity insert error: {e}")
                
                # Insert relationships
                if batch_relationships:
                    for rel in batch_relationships:
                        try:
                            cursor.execute("""
                                INSERT INTO RAG.Relationships (relationship_id, source_entity_id, target_entity_id, 
                                                              relationship_type, description, strength, source_doc_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                rel['relationship_id'],
                                rel['source_entity_id'],
                                rel['target_entity_id'],
                                rel['relationship_type'],
                                rel['description'],
                                rel['strength'],
                                rel['source_doc_id']
                            ))
                        except Exception as e:
                            # Skip duplicates or foreign key errors
                            if "duplicate" not in str(e).lower() and "foreign key" not in str(e).lower():
                                logger.warning(f"Relationship insert error: {e}")
                
                processed += len(docs)
                total_entities += len(batch_entities)
                total_relationships += len(batch_relationships)
                
                logger.info(f"✅ Batch complete: +{len(batch_entities)} entities, +{len(batch_relationships)} relationships")
                
                # Progress update
                if processed % (batch_size * 5) == 0:
                    logger.info(f"📈 Progress: {processed:,}/{total_docs:,} docs ({processed/total_docs*100:.1f}%)")
            
            logger.info(f"🎉 Graph population complete!")
            logger.info(f"📊 Total entities created: {total_entities:,}")
            logger.info(f"📊 Total relationships created: {total_relationships:,}")
            
            # Verify final counts
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            final_entities = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
            final_relationships = cursor.fetchone()[0]
            
            logger.info(f"✅ Final database counts:")
            logger.info(f"   - Entities: {final_entities:,}")
            logger.info(f"   - Relationships: {final_relationships:,}")
            
        except Exception as e:
            logger.error(f"❌ Error during graph population: {e}")
        finally:
            cursor.close()

def main():
    """Main function to run the enhanced graph ingestion"""
    logger.info("🚀 Starting Enhanced GraphRAG Ingestion Pipeline")
    
    pipeline = GraphRAGIngestionPipeline()
    pipeline.populate_graph_tables(batch_size=500)  # Smaller batches for better progress tracking
    
    logger.info("🎉 Enhanced GraphRAG ingestion completed!")

if __name__ == "__main__":
    main()