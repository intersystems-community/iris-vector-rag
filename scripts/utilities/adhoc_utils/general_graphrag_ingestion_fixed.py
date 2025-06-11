#!/usr/bin/env python3
"""
General-purpose GraphRAG ingestion with proper embedding format
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
from general_graphrag_ingestion import GeneralEntityExtractor
import time
import uuid
from collections import defaultdict

def main():
    print("🚀 GraphRAG Ingestion with Proper Embedding Format")
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
            
            batch_entities.extend(entities)
            batch_relationships.extend(relationships)
        
        # Add embeddings and insert entities
        if batch_entities:
            # Get embeddings for all entities in batch
            entity_texts = [e['entity_name'] for e in batch_entities]
            embeddings = embedding_model.encode(entity_texts)
            
            for entity, embedding in zip(batch_entities, embeddings):
                try:
                    # Format embedding properly for IRIS
                    embedding_str = f"[{','.join([f'{x:.10f}' for x in embedding])}]"
                    
                    cursor.execute("""
                        INSERT INTO RAG.Entities 
                        (entity_id, entity_name, entity_type, source_doc_id, embedding)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        entity['entity_id'],
                        entity['entity_name'],
                        entity['entity_type'],
                        entity['source_doc_id'],
                        embedding_str
                    ))
                    total_entities += 1
                except Exception as e:
                    # Skip duplicates
                    if "duplicate" not in str(e).lower():
                        pass
        
        # Insert relationships
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
    
    # Test vector search
    print("\n🔍 Testing vector search...")
    query = "diabetes treatment"
    query_embedding = embedding_model.encode([query])[0]
    query_embedding_str = f"[{','.join([f'{x:.10f}' for x in query_embedding])}]"
    
    cursor.execute("""
        SELECT TOP 5
            entity_name,
            entity_type,
            VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
        FROM RAG.Entities
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
    """, [query_embedding_str])
    
    print(f"Top entities for '{query}':")
    for name, type_, sim in cursor.fetchall():
        print(f"  - {name} ({type_}): {sim:.4f}")
    
    # Close connection
    cursor.close()
    iris.close()
    
    print("\n🎉 GraphRAG ingestion complete with proper embedding format!")

if __name__ == "__main__":
    main()