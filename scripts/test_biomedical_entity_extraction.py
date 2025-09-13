#!/usr/bin/env python3
"""
Test biomedical entity extraction from PMC documents.

This script tests document parsing and entity recognition to ensure
we're getting good data from biomedical documents.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import json
from typing import List, Dict, Any
from common.iris_connection_manager import get_iris_connection
from data.pmc_processor import process_pmc_files
import spacy
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedicalEntityExtractor:
    """Extract biomedical entities from documents."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        # Try to load biomedical NER model
        try:
            # Try scispacy models
            self.nlp = spacy.load("en_core_sci_sm")
            logger.info("Loaded scispacy model: en_core_sci_sm")
        except:
            try:
                # Try biobert model
                self.nlp = spacy.load("en_ner_bc5cdr_md")
                logger.info("Loaded biomedical model: en_ner_bc5cdr_md")
            except:
                # Fallback to general model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.warning("Using general spacy model - biomedical entity recognition may be limited")
                    logger.info("Install scispacy for better results: pip install scispacy")
                    logger.info("Then: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz")
                except:
                    logger.error("No spacy model found. Install with: python -m spacy download en_core_web_sm")
                    self.nlp = None
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if not self.nlp:
            return []
            
        doc = self.nlp(text[:1000000])  # Limit text length for processing
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def extract_biomedical_terms(self, text: str) -> List[str]:
        """Extract biomedical terms using pattern matching."""
        import re
        
        # Common biomedical patterns
        patterns = [
            r'\b[A-Z][A-Z0-9]{2,}\d*\b',  # Gene/protein names (e.g., BRCA1, TP53)
            r'\b(?:cancer|carcinoma|tumor|tumour|malignant|benign)\b',
            r'\b(?:mutation|variant|polymorphism|deletion|insertion)\b',
            r'\b(?:protein|receptor|enzyme|kinase|phosphatase)\b',
            r'\b(?:pathway|signaling|cascade|regulation)\b',
            r'\b(?:expression|overexpression|downregulation|upregulation)\b',
            r'\b(?:therapy|treatment|drug|inhibitor|agonist|antagonist)\b',
            r'\b(?:diagnosis|prognosis|biomarker|marker)\b',
            r'\bmiR-\d+[a-z]?\b',  # microRNAs
            r'\b(?:DNA|RNA|mRNA|miRNA|siRNA|lncRNA)\b',
            r'\b(?:chromosome|chromosomal|genomic|genetic)\b',
            r'\b(?:cell|cellular|tissue|organ)\b',
            r'\b(?:disease|disorder|syndrome|condition)\b',
            r'\b(?:patient|clinical|trial|study)\b'
        ]
        
        terms = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))  # Remove duplicates

def analyze_sample_documents():
    """Analyze sample PMC documents for entity extraction quality."""
    
    logger.info("Loading sample PMC documents...")
    
    # Process a few sample documents
    documents = list(process_pmc_files("data/pmc_oas_downloaded", limit=5))
    
    if not documents:
        logger.error("No documents found")
        return
    
    logger.info(f"Loaded {len(documents)} documents for analysis")
    
    # Initialize entity extractor
    extractor = BiomedicalEntityExtractor()
    
    # Analyze each document
    all_entities = []
    all_terms = []
    
    for i, doc in enumerate(documents):
        logger.info(f"\n{'='*60}")
        logger.info(f"Document {i+1}: {doc.get('doc_id', 'Unknown')}")
        logger.info(f"Title: {doc.get('title', 'No title')[:100]}...")
        
        # Get text content
        text = doc.get('content') or doc.get('abstract') or doc.get('title', '')
        
        if not text:
            logger.warning("No text content found")
            continue
        
        logger.info(f"Text length: {len(text)} characters")
        
        # Extract entities using NER
        if extractor.nlp:
            entities = extractor.extract_entities(text)
            logger.info(f"\nExtracted {len(entities)} entities using NER:")
            
            # Group by type
            entity_types = Counter(e['type'] for e in entities)
            for ent_type, count in entity_types.most_common():
                logger.info(f"  {ent_type}: {count}")
            
            # Show sample entities
            logger.info("\nSample entities:")
            for entity in entities[:10]:
                logger.info(f"  - {entity['text']} ({entity['type']})")
            
            all_entities.extend(entities)
        
        # Extract biomedical terms using patterns
        terms = extractor.extract_biomedical_terms(text)
        logger.info(f"\nExtracted {len(terms)} biomedical terms using patterns")
        
        # Show sample terms
        logger.info("\nSample biomedical terms:")
        term_counts = Counter(terms)
        for term, count in term_counts.most_common(20):
            logger.info(f"  - {term}: {count} occurrences")
        
        all_terms.extend(terms)
        
        # Check for key biomedical concepts
        logger.info("\nChecking for key biomedical concepts:")
        key_concepts = {
            "genes": [t for t in terms if re.match(r'^[A-Z][A-Z0-9]{2,}\d*$', t)],
            "diseases": [t for t in terms if any(word in t.lower() for word in ['cancer', 'disease', 'disorder', 'syndrome'])],
            "mutations": [t for t in terms if any(word in t.lower() for word in ['mutation', 'variant', 'polymorphism'])],
            "treatments": [t for t in terms if any(word in t.lower() for word in ['therapy', 'treatment', 'drug'])]
        }
        
        for concept_type, concepts in key_concepts.items():
            if concepts:
                logger.info(f"  {concept_type}: {', '.join(concepts[:5])}{'...' if len(concepts) > 5 else ''}")
    
    # Overall statistics
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL STATISTICS")
    logger.info(f"{'='*60}")
    
    if all_entities:
        entity_types = Counter(e['type'] for e in all_entities)
        logger.info(f"\nTotal entities extracted: {len(all_entities)}")
        logger.info("Entity types distribution:")
        for ent_type, count in entity_types.most_common():
            logger.info(f"  {ent_type}: {count}")
    
    if all_terms:
        term_counts = Counter(all_terms)
        logger.info(f"\nTotal unique biomedical terms: {len(set(all_terms))}")
        logger.info("Most common biomedical terms:")
        for term, count in term_counts.most_common(30):
            logger.info(f"  {term}: {count} occurrences")

def check_stored_entities():
    """Check what entities are already stored in the database."""
    
    logger.info("\nChecking stored entities in database...")
    
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    try:
        # Check if any entities are stored
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
        count = cursor.fetchone()[0]
        logger.info(f"Total entities in database: {count}")
        
        if count > 0:
            # Sample some entities
            cursor.execute("""
                SELECT entity_text, entity_type, COUNT(*) as count
                FROM RAG.DocumentEntities
                GROUP BY entity_text, entity_type
                ORDER BY count DESC
                LIMIT 20
            """)
            
            logger.info("\nMost common entities in database:")
            for row in cursor.fetchall():
                logger.info(f"  {row[0]} ({row[1]}): {row[2]} occurrences")
            
            # Check entity types
            cursor.execute("""
                SELECT entity_type, COUNT(*) as count
                FROM RAG.DocumentEntities
                GROUP BY entity_type
                ORDER BY count DESC
            """)
            
            logger.info("\nEntity types in database:")
            for row in cursor.fetchall():
                logger.info(f"  {row[0]}: {row[1]} entities")
    
    except Exception as e:
        logger.error(f"Error checking stored entities: {e}")
    
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    import re
    
    logger.info("Testing biomedical entity extraction from PMC documents\n")
    
    # Analyze sample documents
    analyze_sample_documents()
    
    # Check what's already in the database
    check_stored_entities()
    
    logger.info("\n✅ Entity extraction analysis complete!")
    logger.info("\nRecommendations:")
    logger.info("1. Install scispacy for better biomedical entity recognition")
    logger.info("2. Consider using BioBERT or similar models for domain-specific extraction")
    logger.info("3. Implement custom rules for specific entity types (genes, diseases, drugs)")
    logger.info("4. Use the GraphRAG pipeline to populate DocumentEntities table")