#!/usr/bin/env python3
"""
Simple graph ingestion without external NLP dependencies
"""

import sys
import os # Added for path manipulation
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
import re
from typing import List, Dict, Tuple
import uuid

def extract_medical_entities(text: str, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract medical entities and relationships using regex patterns"""
    entities = []
    entity_map = {}
    
    # Define expanded medical patterns
    patterns = [
        # Diseases - expanded
        (r'\b(diabetes|cancer|hypertension|asthma|arthritis|pneumonia|influenza|covid-19|coronavirus|'
         r'alzheimer|parkinson|epilepsy|stroke|hepatitis|tuberculosis|malaria|hiv|aids|'
         r'leukemia|lymphoma|melanoma|carcinoma|sarcoma|tumor|tumour|infection|syndrome|'
         r'disorder|disease|condition|illness|injury|trauma|fracture)\b', 'DISEASE'),
        
        # Drugs/Medications - expanded
        (r'\b(insulin|metformin|aspirin|ibuprofen|acetaminophen|antibiotics|vaccine|medication|'
         r'drug|medicine|pharmaceutical|antibiotic|antiviral|antifungal|analgesic|anesthetic|'
         r'steroid|hormone|vitamin|supplement|inhibitor|blocker|agonist|antagonist|'
         r'chemotherapy|immunotherapy|therapy)\b', 'DRUG'),
        
        # Substances/Chemicals - expanded
        (r'\b(glucose|cholesterol|hemoglobin|protein|oxygen|carbon dioxide|sodium|potassium|'
         r'calcium|iron|zinc|magnesium|phosphorus|nitrogen|hydrogen|enzyme|hormone|'
         r'neurotransmitter|cytokine|antibody|antigen|receptor|ligand|substrate|metabolite|'
         r'lipid|carbohydrate|amino acid|nucleotide|peptide|molecule|compound)\b', 'SUBSTANCE'),
        
        # Organs/Body Parts - expanded
        (r'\b(heart|liver|kidney|lung|brain|pancreas|stomach|intestine|blood|muscle|'
         r'bone|skin|eye|ear|nose|throat|mouth|tooth|teeth|tongue|esophagus|'
         r'bladder|prostate|ovary|uterus|breast|thyroid|adrenal|pituitary|spleen|'
         r'artery|vein|nerve|spine|joint|tissue|cell|organ)\b', 'ORGAN'),
        
        # Treatments/Procedures - expanded
        (r'\b(treatment|therapy|surgery|medication|diagnosis|examination|test|procedure|'
         r'operation|transplant|transfusion|injection|infusion|radiation|chemotherapy|'
         r'immunotherapy|physiotherapy|psychotherapy|rehabilitation|screening|biopsy|'
         r'scan|imaging|x-ray|mri|ct scan|ultrasound|endoscopy|colonoscopy)\b', 'TREATMENT'),
        
        # Symptoms - expanded
        (r'\b(pain|fever|cough|fatigue|nausea|headache|dizziness|weakness|'
         r'vomiting|diarrhea|constipation|bleeding|swelling|inflammation|rash|'
         r'itching|numbness|tingling|shortness of breath|chest pain|abdominal pain|'
         r'back pain|joint pain|muscle pain|loss of appetite|weight loss|weight gain)\b', 'SYMPTOM'),
        
        # Medical Measurements - expanded
        (r'\b(blood pressure|blood sugar|temperature|heart rate|pulse|weight|height|'
         r'bmi|body mass index|cholesterol level|glucose level|oxygen saturation|'
         r'white blood cell count|red blood cell count|platelet count|hemoglobin level|'
         r'creatinine|bilirubin|alt|ast|blood test|lab result)\b', 'MEASUREMENT'),
        
        # Medical Professionals
        (r'\b(doctor|physician|surgeon|nurse|specialist|oncologist|cardiologist|neurologist|'
         r'psychiatrist|psychologist|therapist|pharmacist|radiologist|pathologist|'
         r'anesthesiologist|dermatologist|pediatrician|gynecologist|urologist)\b', 'PROFESSIONAL'),
        
        # Medical Concepts
        (r'\b(gene|genome|dna|rna|chromosome|mutation|expression|pathway|mechanism|'
         r'metabolism|immune system|nervous system|cardiovascular system|respiratory system|'
         r'digestive system|endocrine system|reproductive system|musculoskeletal system)\b', 'CONCEPT')
    ]
    
    # Extract entities
    text_lower = text.lower()
    for pattern, entity_type in patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            entity_name = match.group(1)
            if entity_name not in entity_map:
                entity_id = str(uuid.uuid4())
                entity_map[entity_name] = entity_id
                entities.append({
                    'entity_id': entity_id,
                    'entity_name': entity_name,
                    'entity_type': entity_type,
                    'source_doc_id': doc_id
                })
    
    # Extract relationships based on sentence co-occurrence
    relationships = []
    sentences = re.split(r'[.!?]+', text_lower)
    
    # Process first 50 sentences
    for sent in sentences[:50]:
        if len(sent) < 20:  # Skip very short sentences
            continue
            
        entities_in_sent = []
        for entity_name, entity_id in entity_map.items():
            if entity_name in sent:
                entities_in_sent.append((entity_name, entity_id))
        
        # Create relationships between co-occurring entities
        for i in range(len(entities_in_sent)):
            for j in range(i + 1, len(entities_in_sent)):
                source_name, source_id = entities_in_sent[i]
                target_name, target_id = entities_in_sent[j]
                
                # Determine relationship type based on keywords
                rel_type = 'RELATED_TO'
                if any(word in sent for word in ['treat', 'therapy', 'cure']):
                    rel_type = 'TREATS'
                elif any(word in sent for word in ['cause', 'lead to', 'result in']):
                    rel_type = 'CAUSES'
                elif any(word in sent for word in ['affect', 'impact', 'influence']):
                    rel_type = 'AFFECTS'
                elif any(word in sent for word in ['produce', 'secrete', 'generate']):
                    rel_type = 'PRODUCES'
                elif any(word in sent for word in ['regulate', 'control', 'manage']):
                    rel_type = 'REGULATES'
                elif any(word in sent for word in ['symptom', 'sign', 'indicate']):
                    rel_type = 'SYMPTOM_OF'
                
                relationships.append({
                    'relationship_id': str(uuid.uuid4()),
                    'source_entity_id': source_id,
                    'target_entity_id': target_id,
                    'relationship_type': rel_type,
                    'source_doc_id': doc_id
                })
    
    return entities, relationships

def run_simple_graph_ingestion(limit: int = 10):
    """Run simple graph ingestion on documents"""
    iris = get_iris_connection()
    cursor = iris.cursor()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    print(f"=== Running Simple Graph Ingestion (limit={limit}) ===\n")
    
    # Get documents to process
    cursor.execute(f"""
        SELECT TOP {limit} doc_id, title, text_content
        FROM RAG.SourceDocuments
        WHERE text_content IS NOT NULL
        ORDER BY doc_id
    """)
    documents = cursor.fetchall()
    
    print(f"Processing {len(documents)} documents...")
    
    total_entities = 0
    total_relationships = 0
    
    for idx, (doc_id, title, content) in enumerate(documents, 1):
        print(f"\n[{idx}/{len(documents)}] Processing: {title[:50]}...")
        
        # Extract entities and relationships
        entities, relationships = extract_medical_entities(content[:50000], doc_id)  # Limit content size
        
        # Insert entities
        entities_added = 0
        entity_id_map = {}  # Map old IDs to actual IDs
        
        for entity in entities:
            # Check if entity already exists
            cursor.execute("""
                SELECT entity_id FROM RAG.Entities
                WHERE entity_name = ? AND entity_type = ?
            """, [entity['entity_name'], entity['entity_type']])
            
            existing = cursor.fetchone()
            if not existing:
                # Generate embedding
                embedding = embedding_model.encode([entity['entity_name']])[0]
                embedding_str = ','.join([f'{x:.10f}' for x in embedding])
                
                # Insert entity
                cursor.execute("""
                    INSERT INTO RAG.Entities
                    (entity_id, entity_name, entity_type, source_doc_id, embedding)
                    VALUES (?, ?, ?, ?, ?)
                """, [entity['entity_id'], entity['entity_name'], entity['entity_type'],
                      entity['source_doc_id'], embedding_str])
                total_entities += 1
                entities_added += 1
                entity_id_map[entity['entity_id']] = entity['entity_id']
            else:
                # Map old ID to existing ID
                entity_id_map[entity['entity_id']] = existing[0]
        
        # Insert relationships
        relationships_added = 0
        for rel in relationships[:20]:  # Limit relationships per document
            # Map entity IDs to actual database IDs
            source_id = entity_id_map.get(rel['source_entity_id'])
            target_id = entity_id_map.get(rel['target_entity_id'])
            
            # Only insert if both entities exist
            if source_id and target_id:
                # Check if relationship already exists
                cursor.execute("""
                    SELECT COUNT(*) FROM RAG.Relationships
                    WHERE source_entity_id = ? AND target_entity_id = ?
                    AND relationship_type = ?
                """, [source_id, target_id, rel['relationship_type']])
                
                if cursor.fetchone()[0] == 0:
                    cursor.execute("""
                        INSERT INTO RAG.Relationships
                        (relationship_id, source_entity_id, target_entity_id,
                         relationship_type, source_doc_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, [rel['relationship_id'], source_id, target_id,
                          rel['relationship_type'], rel['source_doc_id']])
                    total_relationships += 1
                    relationships_added += 1
        
        # Commit after each document
        iris.commit()
        print(f"  Found {len(entities)} entities ({entities_added} new), {len(relationships)} relationships ({relationships_added} new)")
    
    # Final statistics
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    final_entities = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    final_relationships = cursor.fetchone()[0]
    
    print(f"\n=== Ingestion Complete ===")
    print(f"Total entities in database: {final_entities}")
    print(f"Total relationships in database: {final_relationships}")
    print(f"New entities added: {total_entities}")
    print(f"New relationships added: {total_relationships}")
    
    cursor.close()
    iris.close()

def test_graphrag_after_ingestion():
    """Test GraphRAG after ingestion"""
    from src.deprecated.graphrag.pipeline_v2 import GraphRAGPipelineV2 # Updated import
    
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the knowledge graph and medical documents: {prompt[:100]}..."
    
    print("\n=== Testing GraphRAG After Ingestion ===\n")
    
    graphrag = GraphRAGPipelineV2(iris, embedding_func, llm_func)
    
    queries = [
        "What is diabetes and how is insulin related?",
        "What are the symptoms of hypertension?",
        "How does the pancreas produce insulin?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        try:
            result = graphrag.run(query, top_k=3)
            
            print(f"✅ Success!")
            print(f"Entities found: {len(result['entities'])}")
            print(f"Relationships found: {len(result['relationships'])}")
            print(f"Documents retrieved: {len(result['retrieved_documents'])}")
            
            if result['entities']:
                print("\nTop entities:")
                for i, ent in enumerate(result['entities'][:5], 1):
                    print(f"  {i}. {ent['entity_name']} ({ent['entity_type']}) - Score: {ent['similarity']:.3f}")
            
            if result['relationships']:
                print("\nTop relationships:")
                for i, rel in enumerate(result['relationships'][:5], 1):
                    print(f"  {i}. {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
            
            print(f"\nAnswer preview: {result['answer'][:150]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    iris.close()

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Run simple graph ingestion')
    parser.add_argument('--limit', type=int, default=10, 
                        help='Number of documents to process (default: 10)')
    parser.add_argument('--test', action='store_true', 
                        help='Run test queries after ingestion')
    args = parser.parse_args()
    
    # Run ingestion
    run_simple_graph_ingestion(limit=args.limit)
    
    # Optionally test
    if args.test:
        test_graphrag_after_ingestion()
    
    print("\n✅ Simple graph ingestion complete!")

if __name__ == "__main__":
    main()