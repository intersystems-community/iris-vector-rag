"""
Quick performance test comparing BasicRAG original vs V2
"""

import sys
import time
sys.path.insert(0, '.')

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
from basic_rag.pipeline import BasicRAGPipeline
from basic_rag.pipeline_v2 import BasicRAGPipelineV2

def test_basic_rag_performance():
    """Quick test of BasicRAG original vs V2"""
    print("🚀 BasicRAG Performance Comparison: Original vs V2")
    print("=" * 60)
    
    # Initialize
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipelines
    original = BasicRAGPipeline(iris_connector, embedding_func, llm_func)
    v2 = BasicRAGPipelineV2(iris_connector, embedding_func, llm_func)
    
    # Test query
    query = "What are the symptoms of diabetes?"
    
    print(f"\n📊 Testing query: '{query}'")
    print("-" * 60)
    
    # Test original pipeline
    print("\n🔍 Testing Original BasicRAG (VARCHAR columns)...")
    start = time.time()
    try:
        result_orig = original.run(query, top_k=5)
        time_orig = time.time() - start
        print(f"✅ Success: {time_orig:.2f}s")
        print(f"   Method: {result_orig.get('method', 'unknown')}")
        print(f"   Documents: {len(result_orig.get('retrieved_documents', []))}")
    except Exception as e:
        time_orig = time.time() - start
        print(f"❌ Failed: {e}")
        time_orig = 0
    
    # Test V2 pipeline
    print("\n🔍 Testing V2 BasicRAG (native VECTOR columns)...")
    start = time.time()
    try:
        result_v2 = v2.run(query, top_k=5)
        time_v2 = time.time() - start
        print(f"✅ Success: {time_v2:.2f}s")
        print(f"   Method: {result_v2.get('method', 'unknown')}")
        print(f"   Documents: {len(result_v2.get('retrieved_documents', []))}")
    except Exception as e:
        time_v2 = time.time() - start
        print(f"❌ Failed: {e}")
        time_v2 = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Original BasicRAG: {time_orig:.2f}s")
    print(f"V2 BasicRAG:       {time_v2:.2f}s")
    
    if time_orig > 0 and time_v2 > 0:
        speedup = time_orig / time_v2
        time_saved = time_orig - time_v2
        print(f"\n🚀 Speedup: {speedup:.2f}x faster")
        print(f"⏱️  Time saved: {time_saved:.2f}s per query")
        
        # Extrapolate to larger scale
        print(f"\n📊 At scale (1000 queries):")
        print(f"   Original would take: {time_orig * 1000:.0f}s ({time_orig * 1000 / 60:.1f} minutes)")
        print(f"   V2 would take:       {time_v2 * 1000:.0f}s ({time_v2 * 1000 / 60:.1f} minutes)")
        print(f"   Total time saved:    {time_saved * 1000:.0f}s ({time_saved * 1000 / 60:.1f} minutes)")

if __name__ == "__main__":
    test_basic_rag_performance()