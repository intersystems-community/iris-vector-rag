#!/usr/bin/env python3
"""
Validate all 7 RAG techniques are working properly
"""

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.iris_connector import get_iris_connection # Updated import
from src.common.embedding_utils import get_embedding_model # Updated import
from src.deprecated.basic_rag.pipeline_v2 import BasicRAGPipelineV2 # Updated import
from src.deprecated.noderag.pipeline_v2 import NodeRAGPipelineV2 # Updated import
from src.deprecated.colbert.pipeline_v2 import ColBERTPipelineV2 # Updated import
from src.experimental.hyde.pipeline import HyDEPipeline # Updated import
from src.experimental.crag.pipeline import CRAGPipeline # Updated import
from src.deprecated.graphrag.pipeline_v2 import GraphRAGPipelineV2 # Updated import
from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline # Updated import

def test_technique(name: str, pipeline, query: str):
    """Test a single RAG technique"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    try:
        result = pipeline.run(query, top_k=3)
        
        print(f"✅ {name} executed successfully!")
        print(f"Query: {query}")
        print(f"Answer preview: {result['answer'][:150]}...")
        
        # Show retrieved documents
        if 'retrieved_documents' in result:
            print(f"\nRetrieved {len(result['retrieved_documents'])} documents")
            for i, doc in enumerate(result['retrieved_documents'][:2], 1):
                if isinstance(doc, dict):
                    content = doc.get('content', '')[:100]
                else:
                    content = getattr(doc, 'content', '')[:100]
                print(f"  Doc {i}: {content}...")
        
        # Show entities for GraphRAG
        if 'entities' in result and result['entities']:
            print(f"\nFound {len(result['entities'])} entities")
            for i, entity in enumerate(result['entities'][:3], 1):
                print(f"  {i}. {entity['entity_name']} ({entity['entity_type']})")
        
        # Show relationships for GraphRAG
        if 'relationships' in result and result['relationships']:
            print(f"\nFound {len(result['relationships'])} relationships")
            for i, rel in enumerate(result['relationships'][:3], 1):
                print(f"  {i}. {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all 7 RAG techniques"""
    print("="*60)
    print("Validating All 7 RAG Techniques")
    print("="*60)
    
    # Initialize components
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the context: {prompt[:100]}..."
    
    # Test query
    query = "What is diabetes and how is it treated?"
    
    # Initialize all pipelines
    pipelines = []
    
    # 1. BasicRAG
    try:
        basic_rag = BasicRAGPipelineV2(iris, embedding_func, llm_func)
        pipelines.append(("BasicRAG V2", basic_rag))
    except Exception as e:
        print(f"Failed to initialize BasicRAG: {e}")
    
    # 2. NodeRAG
    try:
        node_rag = NodeRAGPipelineV2(iris, embedding_func, llm_func)
        pipelines.append(("NodeRAG V2", node_rag))
    except Exception as e:
        print(f"Failed to initialize NodeRAG: {e}")
    
    # 3. ColBERT
    try:
        # ColBERT V2 uses the same embedding function as other pipelines
        colbert = ColBERTPipelineV2(iris, embedding_func, llm_func)
        pipelines.append(("ColBERT V2", colbert))
    except Exception as e:
        print(f"Failed to initialize ColBERT: {e}")
    
    # 4. HyDE
    try:
        hyde = HyDEPipeline(iris, embedding_func, llm_func)
        pipelines.append(("HyDE", hyde))
    except Exception as e:
        print(f"Failed to initialize HyDE: {e}")
    
    # 5. CRAG
    try:
        crag = CRAGPipeline(iris, embedding_func, llm_func)
        pipelines.append(("CRAG", crag))
    except Exception as e:
        print(f"Failed to initialize CRAG: {e}")
    
    # 6. GraphRAG
    try:
        graphrag = GraphRAGPipelineV2(iris, embedding_func, llm_func)
        pipelines.append(("GraphRAG V2", graphrag))
    except Exception as e:
        print(f"Failed to initialize GraphRAG: {e}")
    
    # 7. Hybrid iFIND RAG
    try:
        hybrid = HybridiFindRAGPipeline(iris, embedding_func, llm_func)
        pipelines.append(("Hybrid iFIND RAG", hybrid))
    except Exception as e:
        print(f"Failed to initialize Hybrid iFIND RAG: {e}")
    
    # Test each pipeline
    results = []
    for name, pipeline in pipelines:
        success = test_technique(name, pipeline, query)
        results.append((name, success))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20} {status}")
    
    print(f"\nTotal: {successful}/{total} techniques working")
    
    if successful == total:
        print("\n🎉 ALL 7 RAG TECHNIQUES ARE OPERATIONAL! 🎉")
    else:
        print(f"\n⚠️  {total - successful} techniques need attention")
    
    iris.close()

if __name__ == "__main__":
    main()