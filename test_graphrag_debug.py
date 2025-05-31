import sys
sys.path.append('.')
from graphrag.pipeline_v2 import GraphRAGPipelineV2
from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

try:
    print('=== Testing GraphRAG Pipeline Step by Step ===')
    
    # Initialize components
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f'Based on the provided context, this is a response to: {prompt[:100]}...'
    
    # Create GraphRAG pipeline
    graphrag = GraphRAGPipelineV2(iris, embedding_func, llm_func)
    
    # Test query
    query = 'What is diabetes and how is it treated?'
    print(f'\nQuery: {query}')
    
    # Step 1: Test entity retrieval
    print('\n1. Testing entity retrieval...')
    try:
        entities = graphrag.retrieve_entities(query, top_k=5)
        print(f'   ✓ Retrieved {len(entities)} entities')
        for i, entity in enumerate(entities[:3], 1):
            print(f'     {i}. {entity["entity_name"]} ({entity["entity_type"]}) - Score: {entity["similarity"]:.4f}')
    except Exception as e:
        print(f'   ✗ Entity retrieval failed: {e}')
        import traceback
        traceback.print_exc()
        entities = []
    
    # Step 2: Test relationship retrieval
    print('\n2. Testing relationship retrieval...')
    if entities:
        entity_ids = [e['entity_id'] for e in entities]
        try:
            relationships = graphrag.retrieve_relationships(entity_ids)
            print(f'   ✓ Retrieved {len(relationships)} relationships')
            for i, rel in enumerate(relationships[:3], 1):
                print(f'     {i}. {rel["source_name"]} {rel["relationship_type"]} {rel["target_name"]}')
        except Exception as e:
            print(f'   ✗ Relationship retrieval failed: {e}')
            relationships = []
    else:
        print('   - Skipping (no entities found)')
        relationships = []
    
    # Step 3: Test document retrieval
    print('\n3. Testing document retrieval...')
    try:
        documents = graphrag.retrieve_documents_from_entities(entities, top_k=3)
        print(f'   ✓ Retrieved {len(documents)} documents')
        for i, doc in enumerate(documents, 1):
            print(f'     {i}. Doc ID: {doc.id}, Score: {doc.score:.4f}')
    except Exception as e:
        print(f'   ✗ Document retrieval failed: {e}')
        documents = []
    
    # Step 4: Test answer generation
    print('\n4. Testing answer generation...')
    try:
        answer = graphrag.generate_answer(query, documents, entities, relationships)
        print(f'   ✓ Generated answer: {answer[:200]}...')
    except Exception as e:
        print(f'   ✗ Answer generation failed: {e}')
    
    # Step 5: Test full pipeline
    print('\n5. Testing full pipeline run...')
    try:
        result = graphrag.run(query, top_k=3)
        print(f'   ✓ Pipeline completed successfully!')
        print(f'   - Answer length: {len(result["answer"])} chars')
        print(f'   - Entities found: {len(result["entities"])}')
        print(f'   - Relationships found: {len(result["relationships"])}')
        print(f'   - Documents retrieved: {len(result["retrieved_documents"])}')
    except Exception as e:
        print(f'   ✗ Full pipeline failed: {e}')
        import traceback
        traceback.print_exc()
    
    iris.close()
    print('\n=== Test Complete ===')
    
except Exception as e:
    print(f'❌ Fatal error: {e}')
    import traceback
    traceback.print_exc()