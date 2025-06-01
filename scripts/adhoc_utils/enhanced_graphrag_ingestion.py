#!/usr/bin/env python3
"""
Enhanced GraphRAG ingestion with comprehensive entity extraction
Based on research findings for medical text processing
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

class MedicalEntityExtractor:
    """Enhanced entity extractor for medical/scientific text"""
    
    def __init__(self):
        # Comprehensive medical patterns based on research
        self.entity_patterns = {
            'DISEASE': [
                # Common diseases
                r'\b(?:diabetes|cancer|hypertension|asthma|arthritis|pneumonia|influenza|covid-19|coronavirus)',
                r'\b(?:alzheimer|parkinson|epilepsy|stroke|hepatitis|tuberculosis|malaria|hiv|aids)',
                r'\b(?:leukemia|lymphoma|melanoma|carcinoma|sarcoma|tumor|tumour)',
                # Disease patterns
                r'\b\w+(?:itis|osis|emia|opathy|syndrome|disease|disorder|deficiency|infection)\b',
                r'\b(?:acute|chronic|severe|mild|moderate)\s+\w+',
                r'\b\w+\s+(?:syndrome|disease|disorder|condition|infection)\b',
            ],
            
            'DRUG': [
                # Common drugs
                r'\b(?:insulin|metformin|aspirin|ibuprofen|acetaminophen|penicillin|amoxicillin)',
                r'\b(?:atorvastatin|simvastatin|lisinopril|amlodipine|metoprolol|omeprazole)',
                # Drug patterns
                r'\b\w+(?:mab|nib|tide|vir|cin|mycin|cillin|azole|pril|sartan|statin|olol)\b',
                r'\b(?:anti|beta|alpha|selective)\s*[-]?\s*\w+',
                r'\b\w+\s+(?:inhibitor|blocker|agonist|antagonist|antibody|vaccine)\b',
            ],
            
            'CHEMICAL': [
                # Biological molecules
                r'\b(?:glucose|cholesterol|hemoglobin|insulin|cortisol|testosterone|estrogen)',
                r'\b(?:dopamine|serotonin|norepinephrine|acetylcholine|gaba|glutamate)',
                # Chemical patterns
                r'\b(?:protein|enzyme|hormone|cytokine|antibody|antigen|receptor|ligand)',
                r'\b\w+(?:ase|ine|ate|ide|ose|ol)\b',
                r'\b(?:alpha|beta|gamma|delta|omega)[-\s]?\w+',
            ],
            
            'ANATOMY': [
                # Organs and systems
                r'\b(?:heart|liver|kidney|lung|brain|pancreas|stomach|intestine|colon|spleen)',
                r'\b(?:artery|vein|nerve|muscle|bone|joint|tissue|gland|duct|vessel)',
                # Anatomical patterns
                r'\b(?:cardiovascular|respiratory|nervous|digestive|endocrine|immune)\s+system\b',
                r'\b(?:left|right|anterior|posterior|superior|inferior)\s+\w+',
                r'\b\w+\s+(?:lobe|cortex|nucleus|ganglion|plexus|tract)\b',
            ],
            
            'SYMPTOM': [
                # Common symptoms
                r'\b(?:pain|fever|cough|headache|nausea|vomiting|diarrhea|fatigue|weakness)',
                r'\b(?:dyspnea|tachycardia|bradycardia|hypotension|hypertension|edema)',
                # Symptom patterns
                r'\b(?:acute|chronic|severe|mild|intermittent)\s+(?:pain|discomfort)',
                r'\b\w+(?:algia|odynia|itis|pnea|cardia|tension|emia)\b',
            ],
            
            'PROCEDURE': [
                # Medical procedures
                r'\b(?:surgery|biopsy|transplant|resection|excision|ablation|catheterization)',
                r'\b(?:mri|ct scan|x-ray|ultrasound|ecg|eeg|endoscopy|colonoscopy)',
                # Procedure patterns
                r'\b\w+(?:ectomy|otomy|oscopy|graphy|plasty|pexy|rrhaphy)\b',
                r'\b(?:diagnostic|therapeutic|surgical|minimally invasive)\s+\w+',
            ],
            
            'MEASUREMENT': [
                # Measurements with units
                r'\b\d+(?:\.\d+)?\s*(?:mg|g|kg|mcg|μg|ml|l|dl|mmol|mol|mEq|IU|units?)\b',
                r'\b\d+(?:\.\d+)?\s*(?:mmHg|bpm|breaths?/min|°[CF]|%|percent)\b',
                # Ranges
                r'\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:mg|ml|mmHg|%)',
            ],
            
            'GENE_PROTEIN': [
                # Gene/protein patterns
                r'\b[A-Z][A-Z0-9]{1,5}\b(?![a-z])',  # e.g., TP53, BRCA1
                r'\b(?:p53|bcl-2|her2|egfr|vegf|tnf|il-\d+|cd\d+)\b',
                r'\b\w+\s+(?:gene|protein|receptor|kinase|phosphatase)\b',
            ],
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            combined_pattern = '|'.join(f'({p})' for p in patterns)
            self.compiled_patterns[entity_type] = re.compile(combined_pattern, re.IGNORECASE)
        
        # Relationship patterns
        self.relationship_patterns = [
            # Causal relationships
            (r'(\w+)\s+(?:causes?|leads?\s+to|results?\s+in|induces?)\s+(\w+)', 'CAUSES'),
            (r'(\w+)\s+(?:caused\s+by|due\s+to|resulting\s+from)\s+(\w+)', 'CAUSED_BY'),
            
            # Treatment relationships
            (r'(\w+)\s+(?:treats?|cures?|manages?|controls?|alleviates?)\s+(\w+)', 'TREATS'),
            (r'(\w+)\s+(?:treated\s+with|managed\s+with|controlled\s+by)\s+(\w+)', 'TREATED_WITH'),
            
            # Mechanism relationships
            (r'(\w+)\s+(?:inhibits?|blocks?|suppresses?|reduces?)\s+(\w+)', 'INHIBITS'),
            (r'(\w+)\s+(?:activates?|stimulates?|enhances?|increases?)\s+(\w+)', 'ACTIVATES'),
            (r'(\w+)\s+(?:regulates?|modulates?|controls?)\s+(\w+)', 'REGULATES'),
            
            # Association relationships
            (r'(\w+)\s+(?:associated\s+with|linked\s+to|correlated\s+with)\s+(\w+)', 'ASSOCIATED_WITH'),
            (r'(\w+)\s+(?:risk\s+factor\s+for|predisposes?\s+to)\s+(\w+)', 'RISK_FACTOR'),
            
            # Diagnostic relationships
            (r'(\w+)\s+(?:indicates?|suggests?|diagnostic\s+of)\s+(\w+)', 'INDICATES'),
            (r'(\w+)\s+(?:marker\s+for|biomarker\s+for|sign\s+of)\s+(\w+)', 'MARKER_FOR'),
        ]
        
        # Compile relationship patterns
        self.compiled_relationships = [
            (re.compile(pattern, re.IGNORECASE), rel_type) 
            for pattern, rel_type in self.relationship_patterns
        ]
    
    def extract_entities(self, text: str, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text"""
        entities = []
        entity_map = {}  # entity_text -> entity_id
        entity_positions = defaultdict(list)  # entity_text -> [(start, end)]
        
        # Extract entities by type
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entity_text = match.group(0).strip().lower()
                
                # Skip very short entities
                if len(entity_text) < 3:
                    continue
                
                # Skip pure numbers for non-measurement types
                if entity_type != 'MEASUREMENT' and entity_text.replace('.', '').isdigit():
                    continue
                
                # Record position for relationship extraction
                entity_positions[entity_text].append((match.start(), match.end()))
                
                # Create entity if not exists
                if entity_text not in entity_map:
                    entity_id = str(uuid.uuid4())
                    entity_map[entity_text] = entity_id
                    
                    entities.append({
                        'entity_id': entity_id,
                        'entity_name': entity_text,
                        'entity_type': entity_type,
                        'source_doc_id': doc_id
                    })
        
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
        
        # Add co-occurrence relationships for entities in same sentence
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_entities = []
            
            # Find entities in this sentence
            for entity_text, entity_id in entity_map.items():
                if entity_text in sentence_lower:
                    sentence_entities.append((entity_text, entity_id))
            
            # Create co-occurrence relationships
            for i in range(len(sentence_entities)):
                for j in range(i + 1, len(sentence_entities)):
                    relationships.append({
                        'relationship_id': str(uuid.uuid4()),
                        'source_entity_id': sentence_entities[i][1],
                        'target_entity_id': sentence_entities[j][1],
                        'relationship_type': 'CO_OCCURS',
                        'source_doc_id': doc_id
                    })
        
        return entities, relationships

def main():
    print("🚀 Enhanced GraphRAG Ingestion")
    print("=" * 60)
    
    # Connect to database
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    # Get embedding model
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize entity extractor
    extractor = MedicalEntityExtractor()
    
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
        SELECT doc_id, title, full_text 
        FROM RAG.SourceDocuments 
        WHERE full_text IS NOT NULL
        ORDER BY doc_id
        LIMIT 5000  -- Start with 5k documents for testing
    """)
    
    documents = cursor.fetchall()
    total_docs = len(documents)
    print(f"Processing {total_docs:,} documents...")
    
    # Process documents
    batch_size = 50
    total_entities = 0
    total_relationships = 0
    unique_entities = set()
    
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
            
            # Track unique entities
            for entity in entities:
                unique_entities.add(entity['entity_name'])
            
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
        
        print(f"\r[{processed:,}/{total_docs:,}] {pct:.1f}% - "
              f"Entities: {total_entities:,} (unique: {len(unique_entities):,}), "
              f"Relationships: {total_relationships:,} - "
              f"Rate: {rate:.0f} docs/s", end='', flush=True)
    
    print("\n\n✅ Processing complete!")
    
    # Final counts
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    final_entities = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    final_relationships = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT source_doc_id) FROM RAG.Entities")
    docs_with_entities = cursor.fetchone()[0]
    
    # Entity type distribution
    cursor.execute("""
        SELECT entity_type, COUNT(*) as cnt
        FROM RAG.Entities
        GROUP BY entity_type
        ORDER BY cnt DESC
    """)
    
    print(f"\n📊 Final results:")
    print(f"Total entities: {final_entities:,}")
    print(f"Unique entity names: {len(unique_entities):,}")
    print(f"Total relationships: {final_relationships:,}")
    print(f"Documents with entities: {docs_with_entities:,} ({docs_with_entities/total_docs*100:.1f}%)")
    print(f"Average entities per document: {final_entities/total_docs:.1f}")
    print(f"Average relationships per document: {final_relationships/total_docs:.1f}")
    
    print("\n📈 Entity type distribution:")
    for entity_type, count in cursor.fetchall():
        print(f"  {entity_type}: {count:,}")
    
    # Close connection
    cursor.close()
    iris.close()
    
    print("\n🎉 Enhanced GraphRAG ingestion complete!")
    print(f"Expected ~50 entities/doc × 5,000 docs = ~250,000 entities")
    print(f"Actual extraction rate: {final_entities/total_docs:.1f} entities/doc")

if __name__ == "__main__":
    main()