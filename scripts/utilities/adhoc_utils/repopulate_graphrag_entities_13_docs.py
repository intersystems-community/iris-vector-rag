#!/usr/bin/env python3
"""
Script to re-populate GraphRAG entities and relationships for 13 documents.
Based on enhanced_graphrag_ingestion.py
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
                r'\b(?:diabetes|cancer|hypertension|asthma|arthritis|pneumonia|influenza|covid-19|coronavirus)',
                r'\b(?:alzheimer|parkinson|epilepsy|stroke|hepatitis|tuberculosis|malaria|hiv|aids)',
                r'\b(?:leukemia|lymphoma|melanoma|carcinoma|sarcoma|tumor|tumour)',
                r'\b\w+(?:itis|osis|emia|opathy|syndrome|disease|disorder|deficiency|infection)\b',
                r'\b(?:acute|chronic|severe|mild|moderate)\s+\w+',
                r'\b\w+\s+(?:syndrome|disease|disorder|condition|infection)\b',
            ],
            'DRUG': [
                r'\b(?:insulin|metformin|aspirin|ibuprofen|acetaminophen|penicillin|amoxicillin)',
                r'\b(?:atorvastatin|simvastatin|lisinopril|amlodipine|metoprolol|omeprazole)',
                r'\b\w+(?:mab|nib|tide|vir|cin|mycin|cillin|azole|pril|sartan|statin|olol)\b',
                r'\b(?:anti|beta|alpha|selective)\s*[-]?\s*\w+',
                r'\b\w+\s+(?:inhibitor|blocker|agonist|antagonist|antibody|vaccine)\b',
            ],
            'CHEMICAL': [
                r'\b(?:glucose|cholesterol|hemoglobin|insulin|cortisol|testosterone|estrogen)',
                r'\b(?:dopamine|serotonin|norepinephrine|acetylcholine|gaba|glutamate)',
                r'\b(?:protein|enzyme|hormone|cytokine|antibody|antigen|receptor|ligand)',
                r'\b\w+(?:ase|ine|ate|ide|ose|ol)\b',
                r'\b(?:alpha|beta|gamma|delta|omega)[-\s]?\w+',
            ],
            'ANATOMY': [
                r'\b(?:heart|liver|kidney|lung|brain|pancreas|stomach|intestine|colon|spleen)',
                r'\b(?:artery|vein|nerve|muscle|bone|joint|tissue|gland|duct|vessel)',
                r'\b(?:cardiovascular|respiratory|nervous|digestive|endocrine|immune)\s+system\b',
                r'\b(?:left|right|anterior|posterior|superior|inferior)\s+\w+',
                r'\b\w+\s+(?:lobe|cortex|nucleus|ganglion|plexus|tract)\b',
            ],
            'SYMPTOM': [
                r'\b(?:pain|fever|cough|headache|nausea|vomiting|diarrhea|fatigue|weakness)',
                r'\b(?:dyspnea|tachycardia|bradycardia|hypotension|hypertension|edema)',
                r'\b(?:acute|chronic|severe|mild|intermittent)\s+(?:pain|discomfort)',
                r'\b\w+(?:algia|odynia|itis|pnea|cardia|tension|emia)\b',
            ],
            'PROCEDURE': [
                r'\b(?:surgery|biopsy|transplant|resection|excision|ablation|catheterization)',
                r'\b(?:mri|ct scan|x-ray|ultrasound|ecg|eeg|endoscopy|colonoscopy)',
                r'\b\w+(?:ectomy|otomy|oscopy|graphy|plasty|pexy|rrhaphy)\b',
                r'\b(?:diagnostic|therapeutic|surgical|minimally invasive)\s+\w+',
            ],
            'MEASUREMENT': [
                r'\b\d+(?:\.\d+)?\s*(?:mg|g|kg|mcg|μg|ml|l|dl|mmol|mol|mEq|IU|units?)\b',
                r'\b\d+(?:\.\d+)?\s*(?:mmHg|bpm|breaths?/min|°[CF]|%|percent)\b',
                r'\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:mg|ml|mmHg|%)',
            ],
            'GENE_PROTEIN': [
                r'\b[A-Z][A-Z0-9]{1,5}\b(?![a-z])',  # e.g., TP53, BRCA1
                r'\b(?:p53|bcl-2|her2|egfr|vegf|tnf|il-\d+|cd\d+)\b',
                r'\b\w+\s+(?:gene|protein|receptor|kinase|phosphatase)\b',
            ],
        }
        self.compiled_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            combined_pattern = '|'.join(f'({p})' for p in patterns)
            self.compiled_patterns[entity_type] = re.compile(combined_pattern, re.IGNORECASE)
        
        self.relationship_patterns = [
            (r'(\w+)\s+(?:causes?|leads?\s+to|results?\s+in|induces?)\s+(\w+)', 'CAUSES'),
            (r'(\w+)\s+(?:caused\s+by|due\s+to|resulting\s+from)\s+(\w+)', 'CAUSED_BY'),
            (r'(\w+)\s+(?:treats?|cures?|manages?|controls?|alleviates?)\s+(\w+)', 'TREATS'),
            (r'(\w+)\s+(?:treated\s+with|managed\s+with|controlled\s+by)\s+(\w+)', 'TREATED_WITH'),
            (r'(\w+)\s+(?:inhibits?|blocks?|suppresses?|reduces?)\s+(\w+)', 'INHIBITS'),
            (r'(\w+)\s+(?:activates?|stimulates?|enhances?|increases?)\s+(\w+)', 'ACTIVATES'),
            (r'(\w+)\s+(?:regulates?|modulates?|controls?)\s+(\w+)', 'REGULATES'),
            (r'(\w+)\s+(?:associated\s+with|linked\s+to|correlated\s+with)\s+(\w+)', 'ASSOCIATED_WITH'),
            (r'(\w+)\s+(?:risk\s+factor\s+for|predisposes?\s+to)\s+(\w+)', 'RISK_FACTOR'),
            (r'(\w+)\s+(?:indicates?|suggests?|diagnostic\s+of)\s+(\w+)', 'INDICATES'),
            (r'(\w+)\s+(?:marker\s+for|biomarker\s+for|sign\s+of)\s+(\w+)', 'MARKER_FOR'),
        ]
        self.compiled_relationships = [
            (re.compile(pattern, re.IGNORECASE), rel_type) 
            for pattern, rel_type in self.relationship_patterns
        ]
    
    def extract_entities(self, text: str, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
        entities = []
        entity_map = {} 
        entity_positions = defaultdict(list) 
        
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entity_text = match.group(0).strip().lower()
                if len(entity_text) < 3:
                    continue
                if entity_type != 'MEASUREMENT' and entity_text.replace('.', '').isdigit():
                    continue
                entity_positions[entity_text].append((match.start(), match.end()))
                if entity_text not in entity_map:
                    entity_id = str(uuid.uuid4())
                    entity_map[entity_text] = entity_id
                    entities.append({
                        'entity_id': entity_id,
                        'entity_name': entity_text,
                        'entity_type': entity_type,
                        'source_doc_id': doc_id
                    })
        
        relationships = []
        for pattern, rel_type in self.compiled_relationships:
            for match in pattern.finditer(text):
                source_text = match.group(1).strip().lower()
                target_text = match.group(2).strip().lower()
                if source_text in entity_map and target_text in entity_map:
                    relationships.append({
                        'relationship_id': str(uuid.uuid4()),
                        'source_entity_id': entity_map[source_text],
                        'target_entity_id': entity_map[target_text],
                        'relationship_type': rel_type,
                        'source_doc_id': doc_id
                    })
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_entities = []
            for entity_text, entity_id in entity_map.items():
                if entity_text in sentence_lower:
                    sentence_entities.append((entity_text, entity_id))
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
    print("🚀 Re-populating GraphRAG entities for 13 documents")
    print("=" * 60)
    
    iris = get_iris_connection()
    cursor = iris.cursor()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    extractor = MedicalEntityExtractor()
    
    print("\n🗑️  Clearing existing GraphRAG data (RAG.Entities, RAG.Relationships)...")
    cursor.execute("DELETE FROM RAG.Relationships")
    cursor.execute("DELETE FROM RAG.Entities")
    iris.commit()
    print("✅ Cleared existing data")
    
    print("\n📄 Loading 13 documents...")
    cursor.execute("""
        SELECT doc_id, title, text_content
        FROM RAG.SourceDocuments
        WHERE text_content IS NOT NULL
        ORDER BY doc_id
        LIMIT 13
    """)
    documents = cursor.fetchall()
    total_docs = len(documents)
    print(f"Processing {total_docs} documents...")
    
    total_entities_processed = 0
    total_relationships_processed = 0
    
    print("\n🔄 Processing documents...")
    start_time = time.time()
    
    for idx, (doc_id, title, content) in enumerate(documents):
        print(f"Processing doc {idx+1}/{total_docs}: {doc_id}")
        full_text = f"{title or ''} {content or ''}"
        entities, relationships = extractor.extract_entities(full_text, doc_id)
        
        if entities:
            entity_texts = [e['entity_name'] for e in entities]
            embeddings = embedding_model.encode(entity_texts)
            for entity, embedding in zip(entities, embeddings):
                entity['embedding'] = embedding.tolist()
            
            for entity in entities:
                try:
                    cursor.execute("""
                        INSERT INTO RAG.Entities
                        (entity_id, entity_name, entity_type, source_doc_id, embedding)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        entity['entity_id'],
                        entity['entity_name'],
                        entity['entity_type'],
                        entity['source_doc_id'],
                        ','.join(map(str, entity['embedding'])) # Store as comma-separated string
                    ))
                    total_entities_processed += 1
                except Exception as e:
                    print(f"Error inserting entity {entity['entity_name']} for doc {doc_id}: {e}")
        
        if relationships:
            for rel in relationships:
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
                    total_relationships_processed += 1
                except Exception as e:
                    print(f"Error inserting relationship for doc {doc_id}: {e}")
        
        iris.commit()
            
    elapsed_time = time.time() - start_time
    print(f"\n✅ Processing complete in {elapsed_time:.2f} seconds.")
    
    print(f"\n📊 Final results for {total_docs} documents:")
    print(f"Total entities processed: {total_entities_processed}")
    print(f"Total relationships processed: {total_relationships_processed}")
    
    cursor.close()
    iris.close()
    print("\n🎉 Re-population script finished.")

if __name__ == "__main__":
    main()