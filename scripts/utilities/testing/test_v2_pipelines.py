#!/usr/bin/env python3
"""
Test script for V2 pipelines with HNSW support
"""

import sys
import time
from typing import Dict, Any

import os # Added for path manipulation
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

def test_pipeline(pipeline_class, pipeline_name: str, query: str = "What are the symptoms of diabetes?") -> Dict[str, Any]:
    """Test a single pipeline and return results"""
    print(f"\n{'='*60}")
    print(f"Testing {pipeline_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize pipeline
        iris_connector = get_iris_connection()
        embedding_func = get_embedding_func()
        llm_func = get_llm_func()
        
        pipeline = pipeline_class(
            iris_connector=iris_connector,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        # Run pipeline
        start_time = time.time()
        result = pipeline.query(query, top_k=3)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Display results
        print(f"\n✅ SUCCESS: {pipeline_name} completed in {execution_time:.2f}s")
        print(f"📊 Answer preview: {result.get('answer', 'No answer')[:100]}...")
        print(f"📊 Documents retrieved: {len(result.get('retrieved_documents', []))}")
        print(f"📊 Metadata: {result.get('metadata', {})}")
        
        return {
            "success": True,
            "pipeline": pipeline_name,
            "execution_time": execution_time,
            "num_documents": len(result.get('retrieved_documents', [])),
            "has_answer": bool(result.get('answer')),
            "uses_hnsw": result.get('metadata', {}).get('uses_hnsw', False)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {pipeline_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "pipeline": pipeline_name,
            "error": str(e)
        }

def main():
    """Test all V2 pipelines"""
    print("🚀 Testing V2 RAG Pipelines with HNSW Support")
    print("=" * 80)
    
    # Check if migration is complete
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE document_embedding_vector IS NOT NULL")
        v2_count = cursor.fetchone()[0]
        print(f"\n📊 SourceDocuments_V2 records with VECTOR embeddings: {v2_count:,}")
        
        if v2_count == 0:
            print("\n⚠️  WARNING: No data in _V2 tables yet. Migration may still be running.")
            print("The V2 pipelines will fail until migration completes.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    except Exception as e:
        print(f"\n❌ Error checking V2 tables: {e}")
        return
    finally:
        cursor.close()
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    print(f"\n🔍 Test Query: {test_query}")
    
    results = []
    
    # Test BasicRAG V2
    try:
        from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
        result = test_pipeline(BasicRAGPipeline, "BasicRAG V2", test_query)
        results.append(result)
    except ImportError as e:
        print(f"\n❌ Could not import BasicRAG V2: {e}")
    
    # Test CRAG V2
    try:
        from iris_rag.pipelines.crag import CRAGPipeline # Updated import
        result = test_pipeline(CRAGPipeline, "CRAG V2", test_query)
        results.append(result)
    except ImportError as e:
        print(f"\n❌ Could not import CRAG V2: {e}")
    
    # Test HyDE V2
    try:
        from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
        result = test_pipeline(HyDERAGPipeline, "HyDE V2", test_query)
        results.append(result)
    except ImportError as e:
        print(f"\n❌ Could not import HyDE V2: {e}")
    
    # Test NodeRAG V2
    try:
        from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
        result = test_pipeline(NodeRAGPipeline, "NodeRAG V2", test_query)
        results.append(result)
    except ImportError as e:
        print(f"\n❌ Could not import NodeRAG V2: {e}")
    
    # Test GraphRAG V2
    try:
        from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
        result = test_pipeline(GraphRAGPipeline, "GraphRAG V2", test_query)
        results.append(result)
    except ImportError as e:
        print(f"\n❌ Could not import GraphRAG V2: {e}")
    
    # Test HybridiFindRAG V2
    try:
        from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import
        result = test_pipeline(HybridIFindRAGPipeline, "HybridiFindRAG V2", test_query)
        results.append(result)
    except ImportError as e:
        print(f"\n❌ Could not import HybridiFindRAG V2: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY OF V2 PIPELINE TESTS")
    print("="*80)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    if successful:
        print(f"\n✅ Successful pipelines ({len(successful)}):")
        for result in sorted(successful, key=lambda x: x.get('execution_time', float('inf'))):
            print(f"  - {result['pipeline']}: {result['execution_time']:.2f}s")
            print(f"    Documents: {result.get('num_documents', 0)}, HNSW: {result.get('uses_hnsw', False)}")
    
    if failed:
        print(f"\n❌ Failed pipelines ({len(failed)}):")
        for result in failed:
            print(f"  - {result['pipeline']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📈 Overall: {len(successful)}/{len(results)} pipelines successful")
    
    if successful:
        # Compare with original performance
        print("\n📊 Performance Comparison (V2 with HNSW vs Original):")
        print("Original performance benchmarks:")
        print("  - GraphRAG: 0.76s")
        print("  - BasicRAG: 7.95s")
        print("  - CRAG: 8.26s")
        print("  - HyDE: 10.11s")
        print("  - NodeRAG: 15.34s")
        print("  - HybridiFindRAG: 23.88s")

if __name__ == "__main__":
    main()