#!/usr/bin/env python3
"""
Run enhanced graph ingestion to populate entities and relationships from documents
"""

import sys
import os # Added for path manipulation
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
import spacy
import re
from typing import List, Dict, Tuple
import uuid

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_entities_and_relationships(cursor, text: str, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from text using NLP, reusing existing entity IDs."""
    doc = nlp(text[:1000000])  # Limit text size for spaCy
    
    entities_to_insert = [] # Entities that are new and need insertion
    entity_map = {} # Maps entity_name to its entity_id (either existing or new UUID)
    
    # Process entities (both NER and regex patterns)
    # First pass: identify all potential entity names and their types
    potential_entities = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DISEASE', 'DRUG', 'CHEMICAL']:
            potential_entities.append({'name': ent.text.lower().strip(), 'type': ent.label_})

    medical_patterns = [
        (r'\b(diabetes|cancer|hypertension|asthma|arthritis)\b', 'DISEASE'),
        (r'\b(insulin|metformin|aspirin|ibuprofen)\b', 'DRUG'),
        (r'\b(glucose|cholesterol|hemoglobin|protein)\b', 'SUBSTANCE'),
        (r'\b(heart|liver|kidney|lung|brain)\b', 'ORGAN'),
        (r'\b(treatment|therapy|surgery|medication)\b', 'TREATMENT')
    ]
    for pattern, entity_type in medical_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            potential_entities.append({'name': match.group(1).lower().strip(), 'type': entity_type})

    # Second pass: check existence, assign ID (existing or new), and prepare for insertion if new
    processed_entity_names = set() # To handle duplicates within the same document text

    for pe in potential_entities:
        entity_name = pe['name']
        entity_type = pe['type']

        if entity_name in processed_entity_names:
            continue # Already decided on an ID for this name in this document pass
        
        processed_entity_names.add(entity_name)

        # Check if entity (name, type) already exists in DB
        cursor.execute("SELECT entity_id FROM RAG.Entities WHERE entity_name = ? AND entity_type = ?", (entity_name, entity_type))
        existing_entity_row = cursor.fetchone()
        
        current_entity_id_for_map = None
        if existing_entity_row:
            current_entity_id_for_map = existing_entity_row[0]
            # This entity already exists, no need to add to entities_to_insert
        else:
            new_entity_id = str(uuid.uuid4())
            current_entity_id_for_map = new_entity_id
            entities_to_insert.append({
                'entity_id': new_entity_id,
                'entity_name': entity_name,
                'entity_type': entity_type,
                'source_doc_id': doc_id
            })
        
        entity_map[entity_name] = current_entity_id_for_map
            
    # Extract medical terms using patterns
    medical_patterns = [
        (r'\b(diabetes|cancer|hypertension|asthma|arthritis)\b', 'DISEASE'),
        (r'\b(insulin|metformin|aspirin|ibuprofen)\b', 'DRUG'),
        (r'\b(glucose|cholesterol|hemoglobin|protein)\b', 'SUBSTANCE'),
        (r'\b(heart|liver|kidney|lung|brain)\b', 'ORGAN'),
        (r'\b(treatment|therapy|surgery|medication)\b', 'TREATMENT')
    ]
    
    # Extract relationships based on sentence co-occurrence using the entity_map
    relationships = []
    sentences = [sent.text.lower() for sent in doc.sents]
    
    # Find entities that appear in the same sentence
    for sent in sentences[:100]:  # Limit to first 100 sentences
        entities_in_sent = []
        for entity_name, entity_id in entity_map.items():
            if entity_name in sent:
                entities_in_sent.append((entity_name, entity_id))
        
        # Create relationships between co-occurring entities
        for i in range(len(entities_in_sent)):
            for j in range(i + 1, len(entities_in_sent)):
                source_name, source_id = entities_in_sent[i]
                target_name, target_id = entities_in_sent[j]
                
                # Determine relationship type based on context
                rel_type = 'RELATED_TO'
                if 'treat' in sent:
                    rel_type = 'TREATS'
                elif 'cause' in sent:
                    rel_type = 'CAUSES'
                elif 'affect' in sent:
                    rel_type = 'AFFECTS'
                elif 'produc' in sent:
                    rel_type = 'PRODUCES'
                
                relationships.append({
                    'relationship_id': str(uuid.uuid4()),
                    'source_entity_id': source_id,
                    'target_entity_id': target_id,
                    'relationship_type': rel_type,
                    'source_doc_id': doc_id
                })
    
    return entities_to_insert, relationships # Return only new entities for insertion

def run_enhanced_graph_ingestion(limit: int = 10):
    """Run enhanced graph ingestion on documents"""
    iris = get_iris_connection()
    cursor = iris.cursor()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    print(f"=== Running Enhanced Graph Ingestion (limit={limit}) ===\n")
    
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
    
    for idx, (doc_id, title, raw_content) in enumerate(documents, 1): # Renamed content to raw_content
        print(f"\n[{idx}/{len(documents)}] Processing: {title[:50]}...")
        
        content_str = ""
        if hasattr(raw_content, 'read'):  # Check if it's a Java-style InputStream
            try:
                byte_list = []
                while True:
                    byte_val = raw_content.read()
                    if byte_val == -1:
                        break
                    byte_list.append(byte_val)
                if byte_list:
                    content_bytes = bytes(byte_list)
                    content_str = content_bytes.decode('utf-8', errors='replace')
                else:
                    content_str = "" # Handle case where stream is empty
            except Exception as e_read:
                print(f"Warning: Could not read content stream for doc_id {doc_id}: {e_read}")
                continue # Skip this document if content cannot be read
        elif isinstance(raw_content, str):
            content_str = raw_content
        elif isinstance(raw_content, bytes):
            try:
                content_str = raw_content.decode('utf-8', errors='replace')
            except Exception as e_decode:
                print(f"Warning: Could not decode bytes content for doc_id {doc_id}: {e_decode}")
                continue # Skip this document
        elif raw_content is None:
            content_str = ""
        else:
            print(f"Warning: Unexpected content type for doc_id {doc_id}: {type(raw_content)}. Skipping.")
            continue

        if not content_str.strip():
            print(f"Warning: Empty content for doc_id {doc_id} after processing. Skipping.")
            continue

        # Extract entities and relationships, passing the cursor
        # entities_to_insert will only contain entities not already in the DB by name/type
        entities_to_insert, relationships = extract_entities_and_relationships(cursor, content_str, doc_id)
        
        # Insert new entities
        for entity_data in entities_to_insert:
            # Embedding is generated only for new entities
            embedding = embedding_model.encode([entity_data['entity_name']])[0]
            # Ensure embedding_str is bracketed for TO_VECTOR(?)
            embedding_str = "[" + ','.join([f'{x:.10f}' for x in embedding]) + "]"
            
            # Insert entity
            cursor.execute("""
                INSERT INTO RAG.Entities
                (entity_id, entity_name, entity_type, source_doc_id, embedding)
                VALUES (?, ?, ?, ?, TO_VECTOR(?))
            """, [entity_data['entity_id'], entity_data['entity_name'], entity_data['entity_type'],
                  entity_data['source_doc_id'], embedding_str])
            total_entities += 1
        
        # Insert relationships
        for rel in relationships:
            # Check if relationship already exists
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.EntityRelationships
                WHERE source_entity_id = ? AND target_entity_id = ?
                AND relationship_type = ?
            """, [rel['source_entity_id'], rel['target_entity_id'], rel['relationship_type']])
            
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO RAG.EntityRelationships
                    (relationship_id, source_entity_id, target_entity_id,
                     relationship_type, source_doc_id)
                    VALUES (?, ?, ?, ?, ?)
                """, [rel['relationship_id'], rel['source_entity_id'], 
                      rel['target_entity_id'], rel['relationship_type'], 
                      rel['source_doc_id']])
                total_relationships += 1
        
        # Commit after each document
        iris.commit()
        print(f"  Added {len(entities_to_insert)} entities, {len(relationships)} relationships") # Corrected variable name
    
    # Final statistics
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    final_entities = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships") # Corrected table name
    final_relationships = cursor.fetchone()[0]
    
    print(f"\n=== Ingestion Complete ===")
    print(f"Total entities in database: {final_entities}")
    print(f"Total relationships in database: {final_relationships}")
    print(f"New entities added: {total_entities}")
    print(f"New relationships added: {total_relationships}")
    
    cursor.close()
    iris.close()

def test_enhanced_graphrag():
    """Test GraphRAG after enhanced ingestion"""
    from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
    
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the knowledge graph: {prompt[:100]}..."
    
    print("\n=== Testing Enhanced GraphRAG ===\n")
    
    graphrag = GraphRAGPipeline(iris, embedding_func, llm_func)
    
    queries = [
        "What is diabetes and its treatment?",
        "How does insulin work?",
        "What are cancer treatments?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = graphrag.run(query, top_k=3)
        
        print(f"Entities: {len(result['entities'])}")
        print(f"Relationships: {len(result['relationships'])}")
        print(f"Documents: {len(result['retrieved_documents'])}")
        
        if result['entities']:
            print("Top entities:")
            for i, ent in enumerate(result['entities'][:3], 1):
                print(f"  {i}. {ent['entity_name']} ({ent['entity_type']})")
        
        if result['relationships']:
            print("Top relationships:")
            for i, rel in enumerate(result['relationships'][:3], 1):
                print(f"  {i}. {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
    
    iris.close()

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Run enhanced graph ingestion')
    parser.add_argument('--limit', type=int, default=10, 
                        help='Number of documents to process (default: 10)')
    parser.add_argument('--test', action='store_true', 
                        help='Run test queries after ingestion')
    args = parser.parse_args()
    
    # Run ingestion
    run_enhanced_graph_ingestion(limit=args.limit)
    
    # Optionally test
    if args.test:
        test_enhanced_graphrag()
    
    print("\n✅ Enhanced graph ingestion complete!")

if __name__ == "__main__":
    main()