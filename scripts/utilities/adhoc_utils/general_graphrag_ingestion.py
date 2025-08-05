#!/usr/bin/env python3
"""
General-purpose GraphRAG ingestion with comprehensive entity extraction
Not specific to biomedical domain - works for any text
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
import re
from typing import List, Dict, Tuple, Set
import uuid
import time
from collections import defaultdict
import string

class GeneralEntityExtractor:
    """General-purpose entity extractor for any domain"""
    
    def __init__(self):
        # General entity patterns that work across domains
        self.entity_patterns = {
            'PERSON': [
                # Names with titles
                r'\b(?:Dr|Mr|Mrs|Ms|Prof|Professor|Sir|Lady|Lord|Judge|Senator|President|CEO|CTO|CFO)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                # Full names (First Last)
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
                # Names with initials
                r'\b[A-Z]\.\s*[A-Z][a-z]+\b',
            ],
            
            'ORGANIZATION': [
                # Companies with suffixes
                r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc|Corp|Corporation|LLC|Ltd|Limited|Company|Co|Group|Foundation|Institute|University|College|Hospital|Bank|Agency)\b',
                # Acronyms (3+ capital letters)
                r'\b[A-Z]{3,}\b',
                # Organizations with "of"
                r'\b(?:University|College|Institute|Department|Ministry|Bureau|Office|Board)\s+of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b',
            ],
            
            'LOCATION': [
                # Cities, States, Countries
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b',
                # Places with descriptors
                r'\b(?:North|South|East|West|Upper|Lower|New|Old)\s+[A-Z][a-z]+\b',
                # Geographic features
                r'\b[A-Z][a-z]+\s+(?:River|Mountain|Lake|Ocean|Sea|Bay|Island|Peninsula|Valley|Desert|Forest)\b',
            ],
            
            'DATE_TIME': [
                # Dates in various formats
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                # Years
                r'\b(?:19|20)\d{2}\b',
                # Time
                r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            ],
            
            'QUANTITY': [
                # Numbers with units
                r'\b\d+(?:\.\d+)?\s*(?:percent|%|dollars?|euros?|pounds?|yen|yuan)\b',
                r'\b\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b€\d+(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b£\d+(?:,\d{3})*(?:\.\d{2})?\b',
                # Measurements
                r'\b\d+(?:\.\d+)?\s*(?:meters?|kilometres?|miles?|feet|inches?|pounds?|kilograms?|grams?|liters?|gallons?)\b',
                # Percentages and fractions
                r'\b\d+(?:\.\d+)?%\b',
                r'\b\d+/\d+\b',
            ],
            
            'PRODUCT': [
                # Product names (often capitalized with model numbers)
                r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:v|V)?\d+(?:\.\d+)*\b',
                # Products with trademark symbols
                r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:™|®|©)\b',
            ],
            
            'EVENT': [
                # Events with years
                r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:19|20)\d{2}\b',
                # Common event patterns
                r'\b(?:Conference|Summit|Meeting|Symposium|Workshop|Seminar|Festival|Championship|Olympics|World Cup|Election)\s+(?:of|on|for)?\s*[A-Z][A-Za-z]+\b',
            ],
            
            'CONCEPT': [
                # Technical terms (capitalized multi-word)
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
                # Terms with hyphens
                r'\b[A-Za-z]+(?:-[A-Za-z]+)+\b',
                # Acronyms with explanation
                r'\b[A-Z]{2,}\s*\([^)]+\)\b',
            ],
            
            'IDENTIFIER': [
                # IDs, codes, references
                r'\b[A-Z]{2,}-\d+\b',
                r'\b\d{3,}-\d{3,}-\d{3,}\b',
                r'\b[A-Z]\d{2,}[A-Z]?\b',
                # URLs and emails
                r'\bhttps?://[^\s]+\b',
                r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            ],
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            combined_pattern = '|'.join(f'({p})' for p in patterns)
            self.compiled_patterns[entity_type] = re.compile(combined_pattern)
        
        # General relationship patterns
        self.relationship_patterns = [
            # Causal relationships
            (r'(\w+)\s+(?:causes?|leads?\s+to|results?\s+in|produces?|creates?)\s+(\w+)', 'CAUSES'),
            (r'(\w+)\s+(?:caused\s+by|due\s+to|resulting\s+from|produced\s+by)\s+(\w+)', 'CAUSED_BY'),
            
            # Part-whole relationships
            (r'(\w+)\s+(?:is\s+part\s+of|belongs?\s+to|is\s+in|is\s+within)\s+(\w+)', 'PART_OF'),
            (r'(\w+)\s+(?:contains?|includes?|comprises?|consists?\s+of|has)\s+(\w+)', 'CONTAINS'),
            
            # Comparison relationships
            (r'(\w+)\s+(?:is\s+similar\s+to|resembles?|is\s+like)\s+(\w+)', 'SIMILAR_TO'),
            (r'(\w+)\s+(?:differs?\s+from|is\s+different\s+from|contrasts?\s+with)\s+(\w+)', 'DIFFERENT_FROM'),
            
            # Temporal relationships
            (r'(\w+)\s+(?:before|precedes?|prior\s+to)\s+(\w+)', 'BEFORE'),
            (r'(\w+)\s+(?:after|follows?|subsequent\s+to)\s+(\w+)', 'AFTER'),
            (r'(\w+)\s+(?:during|while|at\s+the\s+same\s+time\s+as)\s+(\w+)', 'CONCURRENT'),
            
            # Association relationships
            (r'(\w+)\s+(?:is\s+associated\s+with|relates?\s+to|is\s+linked\s+to|correlates?\s+with)\s+(\w+)', 'ASSOCIATED_WITH'),
            (r'(\w+)\s+(?:depends?\s+on|requires?|needs?)\s+(\w+)', 'DEPENDS_ON'),
            
            # Action relationships
            (r'(\w+)\s+(?:uses?|utilizes?|employs?|applies?)\s+(\w+)', 'USES'),
            (r'(\w+)\s+(?:affects?|influences?|impacts?|modifies?)\s+(\w+)', 'AFFECTS'),
            
            # Hierarchical relationships
            (r'(\w+)\s+(?:is\s+a\s+type\s+of|is\s+a\s+kind\s+of|is\s+an?\s+)\s+(\w+)', 'IS_A'),
            (r'(\w+)\s+(?:such\s+as|including|for\s+example)\s+(\w+)', 'EXAMPLE_OF'),
        ]
        
        # Compile relationship patterns
        self.compiled_relationships = [
            (re.compile(pattern, re.IGNORECASE), rel_type) 
            for pattern, rel_type in self.relationship_patterns
        ]
        
        # Common words to filter out (stopwords)
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ])
    
    def is_valid_entity(self, text: str, entity_type: str) -> bool:
        """Check if extracted entity is valid"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Too short
        if len(text) < 2:
            return False
        
        # All lowercase (except for certain types)
        if entity_type not in ['QUANTITY', 'DATE_TIME', 'IDENTIFIER'] and text.islower():
            return False
        
        # Stopword
        if text.lower() in self.stopwords:
            return False
        
        # Just punctuation or numbers
        if all(c in string.punctuation + string.digits + ' ' for c in text):
            return False
        
        return True
    
    def extract_entities(self, text: str, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text"""
        entities = []
        entity_map = {}  # entity_text -> entity_id
        entity_positions = defaultdict(list)  # entity_text -> [(start, end)]
        
        # Extract entities by type
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entity_text = match.group(0).strip()
                
                # Validate entity
                if not self.is_valid_entity(entity_text, entity_type):
                    continue
                
                # Normalize entity text
                entity_key = entity_text.lower()
                
                # Record position for relationship extraction
                entity_positions[entity_key].append((match.start(), match.end()))
                
                # Create entity if not exists
                if entity_key not in entity_map:
                    entity_id = str(uuid.uuid4())
                    entity_map[entity_key] = entity_id
                    
                    entities.append({
                        'entity_id': entity_id,
                        'entity_name': entity_text,  # Keep original case
                        'entity_type': entity_type,
                        'source_doc_id': doc_id
                    })
        
        # Extract noun phrases as additional entities
        # Simple pattern for noun phrases
        noun_phrase_pattern = r'\b(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b'
        for match in re.finditer(noun_phrase_pattern, text):
            entity_text = match.group(0).strip()
            entity_key = entity_text.lower()
            
            if entity_key not in entity_map and self.is_valid_entity(entity_text, 'CONCEPT'):
                entity_id = str(uuid.uuid4())
                entity_map[entity_key] = entity_id
                
                entities.append({
                    'entity_id': entity_id,
                    'entity_name': entity_text,
                    'entity_type': 'CONCEPT',
                    'source_doc_id': doc_id
                })
                
                entity_positions[entity_key].append((match.start(), match.end()))
        
        # Extract relationships
        relationships = []
        for pattern, rel_type in self.compiled_relationships:
            for match in pattern.finditer(text):
                source_text = match.group(1).strip().lower()
                target_text = match.group(2).strip().lower()
                
                # Only create relationships between extracted entities
                if source_text in entity_map and target_text in entity_map:
                    relationships.append({
                        'relationship_id': str(uuid.uuid4()),
                        'source_entity_id': entity_map[source_text],
                        'target_entity_id': entity_map[target_text],
                        'relationship_type': rel_type,
                        'source_doc_id': doc_id
                    })
        
        # Add proximity-based relationships
        # Entities within 50 characters of each other
        sorted_entities = sorted(
            [(pos[0], entity_key) for entity_key, positions in entity_positions.items() for pos in positions]
        )
        
        for i in range(len(sorted_entities)):
            for j in range(i + 1, len(sorted_entities)):
                pos1, entity1 = sorted_entities[i]
                pos2, entity2 = sorted_entities[j]
                
                # If entities are close together
                if pos2 - pos1 < 100 and entity1 != entity2:
                    relationships.append({
                        'relationship_id': str(uuid.uuid4()),
                        'source_entity_id': entity_map[entity1],
                        'target_entity_id': entity_map[entity2],
                        'relationship_type': 'NEAR',
                        'source_doc_id': doc_id
                    })
                    break  # Only link to next closest entity
        
        return entities, relationships

def main():
    print("🚀 General-Purpose GraphRAG Ingestion")
    print("=" * 60)
    
    # Connect to database
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    # Get embedding model
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize entity extractor
    extractor = GeneralEntityExtractor()
    
    # Current state
    print("\n📊 Current state:")
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    current_entities = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    current_relationships = cursor.fetchone()[0]
    print(f"Entities: {current_entities}")
    print(f"Relationships: {current_relationships}")
    
    # Clear existing data
    print("\n🗑️  Clearing existing GraphRAG data...")
    cursor.execute("DELETE FROM RAG.Relationships")
    cursor.execute("DELETE FROM RAG.Entities")
    iris.commit()
    print("✅ Cleared existing data")
    
    # Get documents
    print("\n📄 Loading documents...")
    cursor.execute("""
        SELECT doc_id, title, text_content 
        FROM RAG.SourceDocuments 
        WHERE text_content IS NOT NULL
        ORDER BY doc_id
        LIMIT 10000  -- Process 10k documents
    """)
    
    documents = cursor.fetchall()
    total_docs = len(documents)
    print(f"Processing {total_docs:,} documents...")
    
    # Process documents
    batch_size = 100
    total_entities = 0
    total_relationships = 0
    unique_entities = set()
    entity_type_counts = defaultdict(int)
    
    print("\n🔄 Processing documents...")
    start_time = time.time()
    
    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_entities = []
        batch_relationships = []
        
        for doc_id, title, content in batch_docs:
            # Combine title and content
            full_text = f"{title or ''} {content or ''}"
            
            # Extract entities and relationships
            entities, relationships = extractor.extract_entities(full_text, doc_id)
            
            # Track statistics
            for entity in entities:
                unique_entities.add(entity['entity_name'].lower())
                entity_type_counts[entity['entity_type']] += 1
            
            # Add embeddings to entities
            if entities:
                entity_texts = [e['entity_name'] for e in entities]
                embeddings = embedding_model.encode(entity_texts)
                
                for entity, embedding in zip(entities, embeddings):
                    entity['embedding'] = embedding.tolist()
            
            batch_entities.extend(entities)
            batch_relationships.extend(relationships)
        
        # Insert batch
        if batch_entities:
            for entity in batch_entities:
                try:
                    cursor.execute("""
                        INSERT INTO RAG.Entities 
                        (entity_id, entity_name, entity_type, source_doc_id, embedding)
                        VALUES (?, ?, ?, ?, TO_VECTOR(?))
                    """, (
                        entity['entity_id'],
                        entity['entity_name'],
                        entity['entity_type'],
                        entity['source_doc_id'],
                        str(entity['embedding'])
                    ))
                    total_entities += 1
                except Exception as e:
                    # Skip duplicates
                    pass
        
        if batch_relationships:
            for rel in batch_relationships:
                try:
                    cursor.execute("""
                        INSERT INTO RAG.Relationships 
                        (relationship_id, source_entity_id, target_entity_id, 
                         relationship_type, source_doc_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        rel['relationship_id'],
                        rel['source_entity_id'],
                        rel['target_entity_id'],
                        rel['relationship_type'],
                        rel['source_doc_id']
                    ))
                    total_relationships += 1
                except Exception as e:
                    # Skip invalid relationships
                    pass
        
        # Commit batch
        iris.commit()
        
        # Progress update
        processed = min(i + batch_size, total_docs)
        pct = (processed / total_docs) * 100
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total_docs - processed) / rate if rate > 0 else 0
        
        print(f"\r[{processed:,}/{total_docs:,}] {pct:.1f}% - "
              f"Entities: {total_entities:,} (unique: {len(unique_entities):,}), "
              f"Relationships: {total_relationships:,} - "
              f"Rate: {rate:.0f} docs/s - ETA: {eta/60:.1f} min", end='', flush=True)
    
    print("\n\n✅ Processing complete!")
    
    # Final counts
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    final_entities = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    final_relationships = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT source_doc_id) FROM RAG.Entities")
    docs_with_entities = cursor.fetchone()[0]
    
    print(f"\n📊 Final results:")
    print(f"Total entities: {final_entities:,}")
    print(f"Unique entity names: {len(unique_entities):,}")
    print(f"Total relationships: {final_relationships:,}")
    print(f"Documents with entities: {docs_with_entities:,} ({docs_with_entities/total_docs*100:.1f}%)")
    print(f"Average entities per document: {final_entities/total_docs:.1f}")
    print(f"Average relationships per document: {final_relationships/total_docs:.1f}")
    
    print("\n📈 Entity type distribution:")
    for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count:,}")
    
    # Sample entities
    print("\n📝 Sample entities:")
    cursor.execute("""
        SELECT entity_name, entity_type 
        FROM RAG.Entities 
        WHERE entity_name LIKE '%diabetes%' 
        OR entity_name LIKE '%treatment%'
        OR entity_name LIKE '%research%'
        LIMIT 10
    """)
    for name, type_ in cursor.fetchall():
        print(f"  - {name} ({type_})")
    
    # Close connection
    cursor.close()
    iris.close()
    
    print("\n🎉 General-purpose GraphRAG ingestion complete!")

if __name__ == "__main__":
    main()