#!/usr/bin/env python3
"""
Populate DocumentEntities table for GraphRAG with biomedical entities.

This script uses pattern matching and heuristics to extract biomedical entities
from PMC documents since we don't have scispacy installed.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import re
import hashlib
from typing import List, Dict, Any
from common.iris_connection_manager import get_iris_connection
from common.utils import get_embedding_func
from common.db_vector_utils import insert_vector
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedicalEntityExtractor:
    """Extract biomedical entities using patterns and heuristics."""
    
    def __init__(self):
        self.embedding_func = get_embedding_func()
        
        # Define biomedical patterns
        self.gene_pattern = re.compile(r'\b([A-Z][A-Z0-9]{1,}\d+[A-Z]?|BRCA[12]|TP53|EGFR|KRAS|PTEN|APC|PIK3CA|CDKN2A|MLH1|MSH[26])\b')
        self.protein_pattern = re.compile(r'\b(p53|HER2|PD-L1|CD\d+|IL-\d+|TNF-?\w*|VEGF[A-Z]?|mTOR|AKT\d?|ERK\d?|MAPK\d*)\b', re.IGNORECASE)
        self.disease_pattern = re.compile(r'\b(cancer|carcinoma|lymphoma|leukemia|sarcoma|melanoma|glioma|adenoma|tumor|tumour|neoplasm|malignancy|metastasis|metastases)\b', re.IGNORECASE)
        self.drug_pattern = re.compile(r'\b(\w*(mab|nib|ib|cillin|cycline|statin|prazole|azole|mycin|floxacin|vir|ine|ate|ide)\b)', re.IGNORECASE)
        self.pathway_pattern = re.compile(r'\b(\w+\s+(?:pathway|signaling|cascade)|(?:PI3K|MAPK|WNT|NOTCH|HEDGEHOG|TGF-?β|NF-?κB)\s*(?:pathway|signaling)?)\b', re.IGNORECASE)
        self.mutation_pattern = re.compile(r'\b([A-Z]\d+[A-Z]|(?:mutation|variant|deletion|insertion|amplification|translocation|fusion))\b')
        self.cell_type_pattern = re.compile(r'\b(T[- ]?cells?|B[- ]?cells?|NK[- ]?cells?|macrophages?|neutrophils?|lymphocytes?|monocytes?|dendritic[- ]?cells?|stem[- ]?cells?)\b', re.IGNORECASE)
        
    def extract_entities(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        entities = []
        seen_entities = set()
        
        # Extract different entity types
        entity_extractors = [
            (self.gene_pattern, "GENE"),
            (self.protein_pattern, "PROTEIN"),
            (self.disease_pattern, "DISEASE"),
            (self.drug_pattern, "DRUG"),
            (self.pathway_pattern, "PATHWAY"),
            (self.mutation_pattern, "MUTATION"),
            (self.cell_type_pattern, "CELL_TYPE")
        ]
        
        for pattern, entity_type in entity_extractors:
            for match in pattern.finditer(text):
                entity_text = match.group(0).strip()
                
                # Normalize and filter
                entity_text_norm = entity_text.upper()
                
                # Skip very short entities or common words
                if len(entity_text) < 3 or entity_text_norm in ['THE', 'AND', 'FOR', 'WITH', 'FROM']:
                    continue
                
                # Skip if already seen (case-insensitive)
                if entity_text_norm in seen_entities:
                    continue
                
                seen_entities.add(entity_text_norm)
                
                # Generate unique entity ID
                entity_id = hashlib.md5(f"{doc_id}_{entity_text_norm}_{entity_type}".encode()).hexdigest()[:16]
                
                entities.append({
                    "entity_id": entity_id,
                    "doc_id": doc_id,
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "position": match.start()
                })
        
        return entities

def populate_entities(limit: int = 100):
    """Populate DocumentEntities table with extracted entities using schema manager."""
    logger.info("Populating DocumentEntities table for GraphRAG...")
    
    # Initialize schema manager to get proper table structure
    config_manager = ConfigurationManager()
    schema_manager = SchemaManager(config_manager)
    
    connection = get_iris_connection()
    cursor = connection.cursor()
    extractor = BiomedicalEntityExtractor()
    
    try:
        # Ensure DocumentEntities table exists and is properly structured
        schema_manager.ensure_table_ready("DocumentEntities")
        # Get documents that don't have entities yet
        cursor.execute("""
            SELECT d.doc_id, d.title, d.text_content 
            FROM RAG.SourceDocuments d
            WHERE d.doc_id NOT IN (
                SELECT DISTINCT doc_id FROM RAG.DocumentEntities
            )
            AND d.text_content IS NOT NULL
            LIMIT ?
        """, [limit])
        
        documents = cursor.fetchall()
        logger.info(f"Found {len(documents)} documents without entities")
        
        total_entities = 0
        
        for i, (doc_id, title, content) in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}...")
            
            # Combine title and content for entity extraction
            full_text = f"{title or ''} {content or ''}"
            
            # Extract entities
            entities = extractor.extract_entities(full_text, doc_id)
            
            # Store entities
            for entity in entities:
                try:
                    # Generate embedding for entity text
                    embedding = extractor.embedding_func(entity["entity_text"])
                    
                    # Insert entity with embedding
                    success = insert_vector(
                        cursor=cursor,
                        table_name="RAG.DocumentEntities",
                        vector_column_name="embedding",
                        vector_data=embedding,
                        target_dimension=384,
                        key_columns={"entity_id": entity["entity_id"]},
                        additional_data={
                            "doc_id": entity["doc_id"],
                            "entity_text": entity["entity_text"],
                            "entity_type": entity["entity_type"],
                            "position": entity["position"]
                        }
                    )
                    
                    if success:
                        total_entities += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to insert entity {entity['entity_text']}: {e}")
            
            # Commit periodically
            if (i + 1) % 10 == 0:
                connection.commit()
        
        # Final commit
        connection.commit()
        
        logger.info(f"\n✅ Successfully populated {total_entities} entities")
        
        # Show statistics
        cursor.execute("""
            SELECT entity_type, COUNT(*) as entity_count
            FROM RAG.DocumentEntities
            GROUP BY entity_type
            ORDER BY entity_count DESC
        """)
        
        logger.info("\nEntity type distribution:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]} entities")
        
        # Show sample entities
        cursor.execute("""
            SELECT entity_text, entity_type
            FROM RAG.DocumentEntities
            WHERE entity_type IN ('GENE', 'DISEASE', 'DRUG')
            GROUP BY entity_text, entity_type
            ORDER BY COUNT(*) DESC
            LIMIT 20
        """)
        
        logger.info("\nMost common biomedical entities:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]} ({row[1]})")
            
    except Exception as e:
        logger.error(f"Error populating entities: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of documents to process")
    args = parser.parse_args()
    
    logger.info("Populating DocumentEntities table for GraphRAG...")
    populate_entities(args.limit)