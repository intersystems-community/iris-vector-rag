#!/usr/bin/env python3
"""
Populate existing RAG.Entities table for GraphRAG using schema manager.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import re
import hashlib
from typing import List, Dict, Any
from common.database_schema_manager import get_schema_manager
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityPopulator:
    def __init__(self):
        self.schema = get_schema_manager()
        self.connection = get_iris_connection()
        self.embedding_func = get_embedding_func()
        
        # Simple biomedical patterns
        self.patterns = {
            "GENE": re.compile(r'\b([A-Z][A-Z0-9]{2,}|BRCA[12]|TP53|EGFR|KRAS)\b'),
            "DISEASE": re.compile(r'\b(cancer|diabetes|heart disease|alzheimer|covid)\b', re.IGNORECASE),
            "DRUG": re.compile(r'\b(\w+mab|\w+nib|aspirin|insulin|metformin)\b', re.IGNORECASE)
        }
    
    def extract_entities(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Extract simple entities from text."""
        entities = []
        seen = set()
        
        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(text[:2000])  # First 2000 chars
            for match in matches[:5]:  # Limit to 5 per type
                if isinstance(match, tuple):
                    match = match[0]
                entity_text = match.strip()
                if len(entity_text) > 2 and entity_text.lower() not in seen:
                    seen.add(entity_text.lower())
                    entity_id = hashlib.md5(f"{doc_id}_{entity_text}".encode()).hexdigest()[:16]
                    
                    entities.append({
                        'entity_id': entity_id,
                        'source_doc_id': doc_id,
                        'entity_name': entity_text,
                        'entity_type': entity_type,
                        'description': f"{entity_type}: {entity_text}",
                        'embedding': None  # Will compute if needed
                    })
        
        return entities[:10]  # Max 10 entities per document
    
    def populate_entities(self, limit: int = 100):
        """Populate entities table."""
        logger.info(f"Populating entities for up to {limit} documents...")
        
        cursor = self.connection.cursor()
        
        # Get documents
        docs_table = self.schema.get_table_name('source_documents', fully_qualified=True)
        cursor.execute(f"SELECT doc_id, title, text_content FROM {docs_table} LIMIT {limit}")
        documents = cursor.fetchall()
        
        logger.info(f"Processing {len(documents)} documents...")
        
        entities_table = self.schema.get_table_name('document_entities', fully_qualified=True)
        
        # Clear existing entities
        cursor.execute(f"DELETE FROM {entities_table}")
        logger.info(f"Cleared existing entities")
        
        total_entities = 0
        for i, (doc_id, title, content) in enumerate(documents):
            if i % 50 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")
            
            # Extract entities
            text = f"{title} {content}"
            entities = self.extract_entities(text, doc_id)
            
            # Insert entities
            for entity in entities:
                try:
                    cursor.execute(f"""
                        INSERT INTO {entities_table} 
                        (entity_id, source_doc_id, entity_name, entity_type, description) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        entity['entity_id'],
                        entity['source_doc_id'], 
                        entity['entity_name'],
                        entity['entity_type'],
                        entity['description']
                    ))
                    total_entities += 1
                except Exception as e:
                    logger.warning(f"Failed to insert entity {entity['entity_name']}: {e}")
            
            if i % 100 == 0:
                self.connection.commit()
        
        self.connection.commit()
        logger.info(f"✅ Populated {total_entities} entities for {len(documents)} documents")
        cursor.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=970, help='Number of documents to process')
    args = parser.parse_args()
    
    populator = EntityPopulator()
    populator.populate_entities(args.limit)