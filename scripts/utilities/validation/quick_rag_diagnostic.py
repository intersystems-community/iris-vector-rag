#!/usr/bin/env python3
"""
Quick RAG Diagnostic - Test each technique individually
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # Corrected path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

def test_technique(name, pipeline_class, *args):
    """Test a single RAG technique"""
    print(f"\n🔍 Testing {name}...")
    try:
        pipeline = pipeline_class(*args)
        print(f"✅ {name} initialized successfully")
        
        # Test with a simple query
        result = pipeline.run("diabetes treatment", top_k=5, similarity_threshold=0.1)
        
        docs = result.get('retrieved_documents', [])
        answer = result.get('answer', '')
        
        print(f"📊 {name} Results:")
        print(f"   Documents retrieved: {len(docs)}")
        print(f"   Answer length: {len(answer)} chars")
        print(f"   Answer preview: {answer[:100]}...")
        
        return True, len(docs), len(answer)
        
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return False, 0, 0

def main():
    print("🚀 Quick RAG Diagnostic")
    print("=" * 50)
    
    # Initialize common components
    connection = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func(provider="stub")  # Use stub for speed
    
    results = {}
    
    # Test each technique
    techniques = [
        ("BasicRAG", BasicRAGPipeline, connection, embedding_func, llm_func, "RAG"),
        ("HyDE", HyDERAGPipeline, connection, embedding_func, llm_func),
        ("CRAG", CRAGPipeline, connection, embedding_func, llm_func),
        ("NodeRAG", NodeRAGPipeline, connection, embedding_func, llm_func),
        ("GraphRAG", GraphRAGPipeline, connection, embedding_func, llm_func),
        ("HybridiFindRAG", HybridIFindRAGPipeline, connection, embedding_func, llm_func),
    ]
    
    for technique_info in techniques:
        name = technique_info[0]
        pipeline_class = technique_info[1]
        args = technique_info[2:]
        
        success, docs, answer_len = test_technique(name, pipeline_class, *args)
        results[name] = {
            'success': success,
            'documents': docs,
            'answer_length': answer_len
        }
    
    # Summary
    print(f"\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    working_count = 0
    for name, result in results.items():
        status = "✅ WORKING" if result['success'] else "❌ FAILED"
        docs = result['documents']
        answer_len = result['answer_length']
        
        print(f"{name:15} {status:12} Docs: {docs:3d} Answer: {answer_len:3d} chars")
        
        if result['success']:
            working_count += 1
    
    print(f"\n🎯 Working techniques: {working_count}/{len(techniques)}")
    
    # Identify issues
    print(f"\n🔧 ISSUES TO FIX:")
    for name, result in results.items():
        if not result['success']:
            print(f"❌ {name}: Failed to initialize or run")
        elif result['documents'] == 0:
            print(f"⚠️ {name}: No documents retrieved")

if __name__ == "__main__":
    main()