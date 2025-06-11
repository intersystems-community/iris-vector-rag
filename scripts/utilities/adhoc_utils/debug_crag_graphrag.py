#!/usr/bin/env python3
"""
Debug CRAG and GraphRAG issues with RAGAS evaluation
"""

import sys
import os # Added for path manipulation
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
from common.utils import get_llm_func # Updated import

from src.deprecated.crag.pipeline_v2 import CRAGPipelineV2 # Updated import
from src.deprecated.graphrag.pipeline_v2 import GraphRAGPipelineV2 # Updated import

def test_crag():
    """Test CRAG with detailed output"""
    print("\n" + "="*60)
    print("Testing CRAG Pipeline")
    print("="*60)
    
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Use real LLM
    llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
    
    # Initialize pipeline
    pipeline = CRAGPipelineV2(iris, embedding_func, llm_func)
    
    # Test query
    query = "What is diabetes and how is it treated?"
    
    # Run pipeline
    result = pipeline.run(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Documents retrieved: {len(result['retrieved_documents'])}")
    print(f"\nAnswer (first 500 chars):\n{result['answer'][:500]}...")
    
    # Check the actual prompt being sent
    docs = result['retrieved_documents']
    if docs and len(docs) > 0:
        first_doc = docs[0]
        if hasattr(first_doc, 'score'):
            print(f"\nFirst document score: {first_doc.score}")
        if hasattr(first_doc, 'content'):
            print(f"First document preview: {first_doc.content[:200]}...")
        elif isinstance(first_doc, dict) and 'content' in first_doc:
            print(f"First document preview: {first_doc['content'][:200]}...")
    
    iris.close()
    return result

def test_graphrag():
    """Test GraphRAG with detailed output"""
    print("\n" + "="*60)
    print("Testing GraphRAG Pipeline")
    print("="*60)
    
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Use real LLM
    llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
    
    # Initialize pipeline
    pipeline = GraphRAGPipelineV2(iris, embedding_func, llm_func)
    
    # Test query
    query = "What is diabetes and how is it treated?"
    
    # Run pipeline
    result = pipeline.run(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Documents retrieved: {len(result['retrieved_documents'])}")
    print(f"Entities found: {len(result['entities'])}")
    print(f"Relationships found: {len(result['relationships'])}")
    
    print(f"\nAnswer (first 500 chars):\n{result['answer'][:500]}...")
    
    # Show entities and relationships
    if result['entities']:
        print("\nTop entities:")
        for entity in result['entities'][:3]:
            print(f"  - {entity['entity_name']} ({entity['entity_type']})")
    
    if result['relationships']:
        print("\nTop relationships:")
        for rel in result['relationships'][:3]:
            print(f"  - {rel['source_name']} {rel['relationship_type']} {rel['target_name']}")
    
    iris.close()
    return result

def check_graphrag_data():
    """Check GraphRAG data quality"""
    print("\n" + "="*60)
    print("Checking GraphRAG Data Quality")
    print("="*60)
    
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    # Check entities
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    entity_count = cursor.fetchone()[0]
    print(f"Total entities: {entity_count}")
    
    # Check entity embeddings
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE embedding IS NOT NULL")
    entities_with_embeddings = cursor.fetchone()[0]
    print(f"Entities with embeddings: {entities_with_embeddings}")
    
    # Sample entities
    cursor.execute("""
        SELECT entity_name, entity_type 
        FROM RAG.Entities 
        WHERE entity_name LIKE '%diabet%' OR entity_name LIKE '%insulin%'
        LIMIT 10
    """)
    diabetes_entities = cursor.fetchall()
    
    if diabetes_entities:
        print("\nDiabetes-related entities:")
        for name, type_ in diabetes_entities:
            print(f"  - {name} ({type_})")
    else:
        print("\nNo diabetes-related entities found!")
    
    # Check relationships - first check column names
    cursor.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'RAG'
        AND TABLE_NAME = 'RELATIONSHIPS'
    """)
    rel_columns = [col[0] for col in cursor.fetchall()]
    print(f"\nRelationship columns: {rel_columns}")
    
    # Check total relationships
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    total_rels = cursor.fetchone()[0]
    print(f"Total relationships: {total_rels}")
    
    cursor.close()
    iris.close()

def main():
    """Run all tests"""
    print("🔍 Debugging CRAG and GraphRAG Issues")
    
    # Check GraphRAG data first
    check_graphrag_data()
    
    # Test CRAG
    crag_result = test_crag()
    
    # Test GraphRAG
    graphrag_result = test_graphrag()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    print("\nCRAG:")
    print(f"  - Answer length: {len(crag_result['answer'])}")
    print(f"  - Starts with question? {crag_result['answer'].lower().startswith('what')}")
    
    print("\nGraphRAG:")
    print(f"  - Answer length: {len(graphrag_result['answer'])}")
    print(f"  - Uses entity context? {'entities' in graphrag_result['answer'].lower()}")

if __name__ == "__main__":
    main()