#!/usr/bin/env python3
"""
Test GraphRAG pipeline step by step to identify issues
"""

import sys
import os # Added for path manipulation
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import

def test_graphrag_data():
    """Test if GraphRAG tables have data"""
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    print("=== Testing GraphRAG Data Availability ===\n")
    
    # Check Entities table
    print("1. Checking Entities table...")
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    entity_count = cursor.fetchone()[0]
    print(f"   Total entities: {entity_count}")
    
    if entity_count > 0:
        cursor.execute("SELECT TOP 5 entity_id, entity_name, entity_type FROM RAG.Entities")
        print("   Sample entities:")
        for row in cursor.fetchall():
            print(f"     - {row[1]} ({row[2]})")
    
    # Check Relationships table
    print("\n2. Checking Relationships table...")
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    rel_count = cursor.fetchone()[0]
    print(f"   Total relationships: {rel_count}")
    
    if rel_count > 0:
        cursor.execute("""
            SELECT TOP 5 r.relationship_type, e1.entity_name, e2.entity_name
            FROM RAG.Relationships r
            JOIN RAG.Entities e1 ON r.source_entity_id = e1.entity_id
            JOIN RAG.Entities e2 ON r.target_entity_id = e2.entity_id
        """)
        print("   Sample relationships:")
        for row in cursor.fetchall():
            print(f"     - {row[1]} --[{row[0]}]--> {row[2]}")
    
    # Check embeddings
    print("\n3. Checking entity embeddings...")
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE embedding IS NOT NULL")
    embedded_count = cursor.fetchone()[0]
    print(f"   Entities with embeddings: {embedded_count}")
    
    cursor.close()
    iris.close()
    
    return entity_count > 0 and rel_count > 0 and embedded_count > 0

def test_entity_retrieval():
    """Test entity retrieval specifically"""
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    print("\n=== Testing Entity Retrieval ===\n")
    
    # Create pipeline
    graphrag = GraphRAGPipeline(iris, embedding_func, lambda x: x)
    
    # Test query
    query = "diabetes"
    print(f"Query: {query}")
    
    try:
        entities = graphrag.retrieve_entities(query, top_k=5)
        print(f"\nRetrieved {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity['entity_name']} ({entity['entity_type']}) - Score: {entity['similarity']:.4f}")
        return True
    except Exception as e:
        print(f"Error retrieving entities: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        iris.close()

def test_full_pipeline():
    """Test the full GraphRAG pipeline"""
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the provided context, this is a response to: {prompt[:100]}..."
    
    print("\n=== Testing Full GraphRAG Pipeline ===\n")
    
    # Create pipeline
    graphrag = GraphRAGPipeline(iris, embedding_func, llm_func)
    
    # Test query
    query = "What is diabetes and how is it treated?"
    print(f"Query: {query}")
    
    try:
        result = graphrag.run(query, top_k=3)
        
        print(f"\n✅ GraphRAG Pipeline executed successfully!")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Entities found: {len(result['entities'])}")
        print(f"Relationships found: {len(result['relationships'])}")
        print(f"Documents retrieved: {len(result['retrieved_documents'])}")
        
        # Show some entities
        if result['entities']:
            print(f"\nTop entities:")
            for i, entity in enumerate(result['entities'][:3], 1):
                print(f"  {i}. {entity['entity_name']} ({entity['entity_type']}) - Score: {entity['similarity']:.4f}")
        
        return True
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        iris.close()

def main():
    """Run all tests"""
    print("="*60)
    print("GraphRAG Step-by-Step Testing")
    print("="*60)
    
    # Test 1: Check data
    if not test_graphrag_data():
        print("\n❌ GraphRAG data not available. Need to run graph ingestion first.")
        return
    
    # Test 2: Entity retrieval
    if not test_entity_retrieval():
        print("\n❌ Entity retrieval failed.")
        return
    
    # Test 3: Full pipeline
    if test_full_pipeline():
        print("\n🎉 GraphRAG is FULLY OPERATIONAL!")
    else:
        print("\n❌ Full pipeline test failed.")

if __name__ == "__main__":
    main()