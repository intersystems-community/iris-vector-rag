#!/usr/bin/env python3
"""
Test all 7 RAG techniques to ensure they are fully operational.

This script tests:
1. BasicRAG
2. NodeRAG  
3. GraphRAG
4. ColBERT
5. HyDE
6. CRAG
7. Hybrid iFindRAG

Goal: Achieve 100% success rate with all 7 techniques working.
"""

import sys
import logging
from typing import Dict, Any
import os # Added

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) # Assuming script is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_rag():
    """Test BasicRAG pipeline."""
    try:
        from iris_rag.pipelines import BasicRAGPipeline # Updated import
        
        iris = get_iris_connection()
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        def embedding_func(texts):
            return embedding_model.encode(texts)
        
        def llm_func(prompt):
            return f"BasicRAG response to: {prompt[:50]}..."
        
        pipeline = BasicRAGPipeline(iris, embedding_func, llm_func)
        result = pipeline.run("diabetes treatment", top_k=3)
        
        iris.close()
        return True, f"Retrieved {len(result['retrieved_documents'])} documents"
        
    except Exception as e:
        return False, str(e)

def test_node_rag():
    """Test NodeRAG pipeline."""
    try:
        from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
        
        iris = get_iris_connection()
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        def embedding_func(texts):
            return embedding_model.encode(texts)
        
        def llm_func(prompt):
            return f"NodeRAG response to: {prompt[:50]}..."
        
        pipeline = NodeRAGPipeline(iris, embedding_func, llm_func)
        result = pipeline.run("diabetes treatment", top_k=3)
        
        iris.close()
        return True, f"Retrieved {len(result['retrieved_documents'])} chunks"
        
    except Exception as e:
        return False, str(e)

def test_graph_rag():
    """Test GraphRAG pipeline."""
    try:
        from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
        
        iris = get_iris_connection()
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        def embedding_func(texts):
            return embedding_model.encode(texts)
        
        def llm_func(prompt):
            return f"GraphRAG response to: {prompt[:50]}..."
        
        pipeline = GraphRAGPipeline(iris, embedding_func, llm_func)
        result = pipeline.run("diabetes treatment", top_k=3)
        
        iris.close()
        return True, f"Retrieved {len(result['entities'])} entities, {len(result['relationships'])} relationships"
        
    except Exception as e:
        return False, str(e)

def test_colbert():
    """Test ColBERT pipeline."""
    try:
        from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
        
        iris = get_iris_connection()
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        def embedding_func(texts):
            return embedding_model.encode(texts)
        
        def llm_func(prompt):
            return f"ColBERT response to: {prompt[:50]}..."
        
        pipeline = ColBERTRAGPipeline(iris, embedding_func, llm_func)
        result = pipeline.run("diabetes treatment", top_k=3)
        
        iris.close()
        return True, f"Retrieved {len(result['retrieved_documents'])} documents with token-level matching"
        
    except Exception as e:
        return False, str(e)

def test_hyde():
    """Test HyDE pipeline."""
    try:
        from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
        
        iris = get_iris_connection()
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        def embedding_func(texts):
            return embedding_model.encode(texts)
        
        def llm_func(prompt):
            return f"HyDE response to: {prompt[:50]}..."
        
        pipeline = HyDERAGPipeline(iris, embedding_func, llm_func)
        result = pipeline.run("diabetes treatment", top_k=3)
        
        iris.close()
        return True, f"Generated hypothetical document and retrieved {len(result['retrieved_documents'])} documents"
        
    except Exception as e:
        return False, str(e)

def test_crag():
    """Test CRAG pipeline."""
    try:
        from iris_rag.pipelines.crag import CRAGPipeline # Updated import
        
        iris = get_iris_connection()
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        def embedding_func(texts):
            return embedding_model.encode(texts)
        
        def llm_func(prompt):
            return f"CRAG response to: {prompt[:50]}..."
        
        pipeline = CRAGPipeline(iris, embedding_func, llm_func)
        result = pipeline.run("diabetes treatment", top_k=3)
        
        iris.close()
        return True, f"Performed corrective retrieval with {len(result['retrieved_documents'])} documents"
        
    except Exception as e:
        return False, str(e)

def test_hybrid_ifind_rag():
    """Test Hybrid iFindRAG pipeline."""
    try:
        from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import
        
        iris = get_iris_connection()
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        def embedding_func(texts):
            return embedding_model.encode(texts)
        
        def llm_func(prompt):
            return f"Hybrid iFindRAG response to: {prompt[:50]}..."
        
        pipeline = HybridIFindRAGPipeline(iris, embedding_func, llm_func)
        result = pipeline.run("diabetes treatment", top_k=3)
        
        iris.close()
        return True, f"Combined multiple retrieval strategies with {len(result['retrieved_documents'])} documents"
        
    except Exception as e:
        return False, str(e)

def main():
    """Test all 7 RAG techniques."""
    
    print("🧪 Testing All 7 RAG Techniques for 100% Success Rate")
    print("=" * 60)
    
    techniques = [
        ("BasicRAG", test_basic_rag),
        ("NodeRAG", test_node_rag),
        ("GraphRAG", test_graph_rag),
        ("ColBERT", test_colbert),
        ("HyDE", test_hyde),
        ("CRAG", test_crag),
        ("Hybrid iFindRAG", test_hybrid_ifind_rag),
    ]
    
    results = {}
    successful = 0
    
    for name, test_func in techniques:
        print(f"\n🔍 Testing {name}...")
        try:
            success, message = test_func()
            if success:
                print(f"✅ {name}: SUCCESS - {message}")
                successful += 1
            else:
                print(f"❌ {name}: FAILED - {message}")
            results[name] = (success, message)
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            results[name] = (False, str(e))
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    for name, (success, message) in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{name:20} {status}")
    
    success_rate = (successful / len(techniques)) * 100
    print(f"\n🎯 Success Rate: {successful}/{len(techniques)} ({success_rate:.1f}%)")
    
    if successful == len(techniques):
        print("🎉 ALL 7 RAG TECHNIQUES ARE FULLY OPERATIONAL!")
        print("🚀 Enterprise RAG system is ready for comprehensive evaluation!")
        return True
    else:
        print(f"⚠️  {len(techniques) - successful} techniques still need fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)