#!/usr/bin/env python3
"""
Run enhanced graph ingestion to populate entities and relationships from documents
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
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

def extract_entities_and_relationships(text: str, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from text using NLP"""
    doc = nlp(text[:1000000])  # Limit text size for spaCy
    
    entities = []
    entity_map = {}
    
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DISEASE', 'DRUG', 'CHEMICAL']:
            entity_id = str(uuid.uuid4())
            entity_name = ent.text.lower().strip()
            
            # Skip if already processed
            if entity_name in entity_map:
                continue
                
            entity_map[entity_name] = entity_id
            entities.append({
                'entity_id': entity_id,
                'entity_name': entity_name,
                'entity_type': ent.label_,
                'source_doc_id': doc_id
            })
    
    # Extract medical terms using patterns
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
    
    return entities, relationships

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
    
    for idx, (doc_id, title, content) in enumerate(documents, 1):
        print(f"\n[{idx}/{len(documents)}] Processing: {title[:50]}...")
        
        # Extract entities and relationships
        entities, relationships = extract_entities_and_relationships(content, doc_id)
        
        # Insert entities
        for entity in entities:
            # Check if entity already exists
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.Entities 
                WHERE entity_name = ? AND entity_type = ?
            """, [entity['entity_name'], entity['entity_type']])
            
            if cursor.fetchone()[0] == 0:
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
        
        # Insert relationships
        for rel in relationships:
            # Check if relationship already exists
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.Relationships 
                WHERE source_entity_id = ? AND target_entity_id = ? 
                AND relationship_type = ?
            """, [rel['source_entity_id'], rel['target_entity_id'], rel['relationship_type']])
            
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO RAG.Relationships 
                    (relationship_id, source_entity_id, target_entity_id, 
                     relationship_type, source_doc_id)
                    VALUES (?, ?, ?, ?, ?)
                """, [rel['relationship_id'], rel['source_entity_id'], 
                      rel['target_entity_id'], rel['relationship_type'], 
                      rel['source_doc_id']])
                total_relationships += 1
        
        # Commit after each document
        iris.commit()
        print(f"  Added {len(entities)} entities, {len(relationships)} relationships")
    
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

def test_enhanced_graphrag():
    """Test GraphRAG after enhanced ingestion"""
    from graphrag.pipeline_v2 import GraphRAGPipelineV2
    
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the knowledge graph: {prompt[:100]}..."
    
    print("\n=== Testing Enhanced GraphRAG ===\n")
    
    graphrag = GraphRAGPipelineV2(iris, embedding_func, llm_func)
    
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