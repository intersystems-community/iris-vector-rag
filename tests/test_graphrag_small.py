#!/usr/bin/env python3
"""
Test GraphRAG with a small number of documents to ensure everything works
"""

import sys
import os # Added for path manipulation
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.iris_connector import get_iris_connection # Updated import
from src.common.embedding_utils import get_embedding_model # Updated import
from general_graphrag_ingestion import GeneralEntityExtractor # Assuming this is a local/root script
import time

def main():
    print("🧪 Testing GraphRAG with Small Dataset")
    print("=" * 60)
    
    # Connect to database
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    # Get embedding model
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize entity extractor
    extractor = GeneralEntityExtractor()
    
    # Get just 10 documents for testing
    print("\n📄 Loading 10 test documents...")
    cursor.execute("""
        SELECT TOP 10 doc_id, title, text_content 
        FROM RAG.SourceDocuments 
        WHERE text_content IS NOT NULL
        ORDER BY doc_id
    """)
    
    documents = cursor.fetchall()
    print(f"Loaded {len(documents)} documents")
    
    # Process documents
    print("\n🔄 Processing documents...")
    total_entities = 0
    total_relationships = 0
    
    for doc_id, title, content in documents:
        # Combine title and content
        full_text = f"{title or ''} {content or ''}"
        
        # Extract entities and relationships
        entities, relationships = extractor.extract_entities(full_text, doc_id)
        
        print(f"\nDoc: {title[:50]}...")
        print(f"  Found {len(entities)} entities, {len(relationships)} relationships")
        
        # Insert entities with embeddings
        for entity in entities:
            try:
                # Generate embedding
                embedding = embedding_model.encode([entity['entity_name']])[0]
                embedding_str = str(embedding.tolist())
                
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
                if "duplicate" not in str(e).lower():
                    print(f"    Error inserting entity: {e}")
        
        # Insert relationships
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
                total_relationships += 1
            except:
                pass
        
        iris.commit()
    
    print(f"\n✅ Inserted {total_entities} entities and {total_relationships} relationships")
    
    # Test vector search
    print("\n🔍 Testing vector search...")
    query = "diabetes treatment"
    query_embedding = embedding_model.encode([query])[0]
    query_embedding_str = str(query_embedding.tolist())
    
    try:
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
        
        print("\n✅ Vector search working correctly!")
        
    except Exception as e:
        print(f"\n❌ Vector search error: {e}")
        print("\nTrying alternative approach...")
        
        # Try without vector operations
        cursor.execute("""
            SELECT entity_name, entity_type
            FROM RAG.Entities
            WHERE LOWER(entity_name) LIKE '%diabetes%'
               OR LOWER(entity_name) LIKE '%treatment%'
            LIMIT 5
        """)
        
        print(f"Text search results:")
        for name, type_ in cursor.fetchall():
            print(f"  - {name} ({type_})")
    
    # Test GraphRAG pipeline
    print("\n🧪 Testing GraphRAG pipeline...")
    from src.deprecated.graphrag.pipeline_v2 import GraphRAGPipelineV2 # Updated import
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the context: {prompt[:100]}..."
    
    try:
        graphrag = GraphRAGPipelineV2(iris, embedding_func, llm_func)
        result = graphrag.run(query, top_k=3)
        
        print(f"✅ GraphRAG pipeline executed successfully!")
        print(f"  Entities found: {len(result['entities'])}")
        print(f"  Relationships found: {len(result['relationships'])}")
        print(f"  Documents retrieved: {len(result['retrieved_documents'])}")
        
    except Exception as e:
        print(f"❌ GraphRAG pipeline error: {e}")
        import traceback
        traceback.print_exc()
    
    # Close connection
    cursor.close()
    iris.close()
    
    print("\n🎯 Test complete!")

if __name__ == "__main__":
    main()