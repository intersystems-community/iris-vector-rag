"""
Quick test to compare original vs V2 performance on the same query
"""

import sys
import time
import os # Added for path manipulation

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

# Import all original pipelines
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.experimental.crag.pipeline import CRAGPipeline # Updated import
from src.experimental.hyde.pipeline import HyDEPipeline # Updated import
from src.experimental.graphrag.pipeline import OriginalGraphRAGPipeline as FixedGraphRAGPipeline # Updated import

# Import all V2 pipelines
from src.deprecated.basic_rag.pipeline_v2 import BasicRAGPipelineV2 # Updated import
from src.deprecated.crag.pipeline_v2 import CRAGPipelineV2 # Updated import
from src.deprecated.hyde.pipeline_v2 import HyDEPipelineV2 # Updated import
from src.deprecated.graphrag.pipeline_v2 import GraphRAGPipelineV2 # Updated import

def test_pipeline_pair(name, original_class, v2_class, query="What are the symptoms of diabetes?"):
    """Test a pair of original and V2 pipelines"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    # Initialize
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Test original
    print(f"\n🔍 {name} Original:")
    try:
        original = original_class(iris_connector, embedding_func, llm_func)
        start = time.time()
        result_orig = original.run(query, top_k=5)
        time_orig = time.time() - start
        docs_orig = len(result_orig.get('retrieved_documents', []))
        print(f"   ✅ Time: {time_orig:.2f}s, Docs: {docs_orig}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}...")
        time_orig = 0
        docs_orig = 0
    
    # Test V2
    print(f"\n🔍 {name} V2:")
    try:
        v2 = v2_class(iris_connector, embedding_func, llm_func)
        start = time.time()
        result_v2 = v2.run(query, top_k=5)
        time_v2 = time.time() - start
        docs_v2 = len(result_v2.get('retrieved_documents', []))
        print(f"   ✅ Time: {time_v2:.2f}s, Docs: {docs_v2}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}...")
        time_v2 = 0
        docs_v2 = 0
    
    # Compare
    if time_orig > 0 and time_v2 > 0:
        if time_v2 < time_orig:
            speedup = time_orig / time_v2
            print(f"\n   🚀 V2 is {speedup:.2f}x faster!")
        else:
            slowdown = time_v2 / time_orig
            print(f"\n   ⚠️  V2 is {slowdown:.2f}x slower")
    
    return time_orig, time_v2

def main():
    print("🚀 Quick Original vs V2 Performance Test")
    print("Testing on actual data with same query\n")
    
    # First check if V2 tables have data
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE document_embedding_vector IS NOT NULL")
    v2_count = cursor.fetchone()[0]
    
    if v2_count == 0:
        print("⚠️  WARNING: V2 tables are empty!")
        print("The V2 pipelines will fail because data hasn't been migrated yet.")
        print("This is why they appear 'slower' - they're not actually running!\n")
    else:
        print(f"✅ V2 tables have {v2_count:,} documents with embeddings\n")
    
    # Test each pipeline pair
    results = []
    
    # BasicRAG
    orig_time, v2_time = test_pipeline_pair("BasicRAG", BasicRAGPipeline, BasicRAGPipelineV2)
    results.append(("BasicRAG", orig_time, v2_time))
    
    # CRAG
    orig_time, v2_time = test_pipeline_pair("CRAG", CRAGPipeline, CRAGPipelineV2)
    results.append(("CRAG", orig_time, v2_time))
    
    # HyDE
    orig_time, v2_time = test_pipeline_pair("HyDE", HyDEPipeline, HyDEPipelineV2)
    results.append(("HyDE", orig_time, v2_time))
    
    # GraphRAG
    orig_time, v2_time = test_pipeline_pair("GraphRAG", FixedGraphRAGPipeline, GraphRAGPipelineV2)
    results.append(("GraphRAG", orig_time, v2_time))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    for name, orig, v2 in results:
        if orig > 0 and v2 > 0:
            if v2 < orig:
                print(f"{name}: Original {orig:.2f}s → V2 {v2:.2f}s (🚀 {orig/v2:.2f}x faster)")
            else:
                print(f"{name}: Original {orig:.2f}s → V2 {v2:.2f}s (⚠️  {v2/orig:.2f}x slower)")
        elif orig > 0:
            print(f"{name}: Original {orig:.2f}s → V2 FAILED")
        else:
            print(f"{name}: Both FAILED")
    
    if v2_count == 0:
        print("\n⚠️  IMPORTANT: V2 tables have no data!")
        print("Run the migration script first: python scripts/comprehensive_vector_migration.py")
        print("Then re-run this test to see the real performance improvements.")

if __name__ == "__main__":
    main()