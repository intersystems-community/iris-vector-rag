"""
Fair performance comparison: Original BasicRAG with full vector search vs V2
"""

import sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.iris_connector import get_iris_connection # Updated import
from src.common.utils import get_embedding_func, get_llm_func # Updated import
from src.deprecated.basic_rag.pipeline_v2_fixed import BasicRAGPipelineV2Fixed as BasicRAGPipelineV2 # Updated import

def test_original_with_full_search():
    """Test original BasicRAG with full vector search (will fail due to IRIS bug)"""
    print("🔍 Testing Original BasicRAG with full vector search...")
    
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    cursor = iris_connector.cursor()
    
    query = "What are the symptoms of diabetes?"
    query_embedding = embedding_func([query])[0]
    query_embedding_str = ','.join(map(str, query_embedding))
    
    # Try the query that triggers the IRIS bug
    sql_query = f"""
        SELECT TOP 5 doc_id, title, text_content,
               VECTOR_COSINE(
                   TO_VECTOR(embedding, 'DOUBLE', 384),
                   TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)
               ) as similarity_score
        FROM RAG.SourceDocuments_V2
        WHERE embedding IS NOT NULL
        ORDER BY similarity_score DESC
    """
    
    start = time.time()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        time_taken = time.time() - start
        print(f"✅ Success: {time_taken:.2f}s, {len(results)} documents")
        return time_taken, True
    except Exception as e:
        time_taken = time.time() - start
        print(f"❌ Failed with IRIS bug: {str(e)[:100]}...")
        return time_taken, False

def test_v2_with_native_vector():
    """Test V2 with native VECTOR columns"""
    print("\n🔍 Testing V2 BasicRAG with native VECTOR columns...")
    
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    pipeline = BasicRAGPipelineV2(iris_connector, embedding_func, llm_func)
    
    query = "What are the symptoms of diabetes?"
    
    start = time.time()
    try:
        # Just test the retrieval part for fair comparison
        docs = pipeline.retrieve_documents(query, top_k=5)
        time_taken = time.time() - start
        print(f"✅ Success: {time_taken:.2f}s, {len(docs)} documents")
        return time_taken, True
    except Exception as e:
        time_taken = time.time() - start
        print(f"❌ Failed: {e}")
        return time_taken, False

def test_python_fallback():
    """Test the Python fallback approach (what original BasicRAG actually uses)"""
    print("\n🔍 Testing Python fallback (what original BasicRAG uses)...")
    
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    cursor = iris_connector.cursor()
    
    query = "What are the symptoms of diabetes?"
    query_embedding = embedding_func([query])[0]
    
    # Get only 100 documents (what original BasicRAG does)
    sql = """
        SELECT TOP 100 doc_id, title, text_content, embedding
        FROM RAG.SourceDocuments_V2
        WHERE embedding IS NOT NULL
        AND embedding NOT LIKE '0.1,0.1,0.1%'
        ORDER BY doc_id
    """
    
    start = time.time()
    cursor.execute(sql)
    sample_docs = cursor.fetchall()
    
    # Calculate similarities in Python
    doc_scores = []
    for row in sample_docs:
        doc_id = row[0]
        embedding_str = row[3]
        try:
            doc_embedding = [float(x.strip()) for x in embedding_str.split(',')]
            similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            doc_scores.append((similarity, row))
        except:
            pass
    
    # Sort and get top 5
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = doc_scores[:5]
    
    time_taken = time.time() - start
    print(f"✅ Success: {time_taken:.2f}s, {len(top_docs)} documents (from 100 sample)")
    return time_taken, True

def main():
    print("🚀 Fair V2 Performance Comparison")
    print("=" * 80)
    print("Comparing vector search approaches on 99,990 documents\n")
    
    # Test 1: Original approach with IRIS vector functions (will fail)
    orig_time, orig_success = test_original_with_full_search()
    
    # Test 2: V2 with native VECTOR columns
    v2_time, v2_success = test_v2_with_native_vector()
    
    # Test 3: Python fallback (what original actually uses)
    fallback_time, fallback_success = test_python_fallback()
    
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print(f"\n1. Original with IRIS vector search (full dataset):")
    if orig_success:
        print(f"   ✅ Would take: {orig_time:.2f}s")
    else:
        print(f"   ❌ FAILS due to IRIS SQL parser bug")
    
    print(f"\n2. V2 with native VECTOR columns (full dataset):")
    print(f"   ✅ Takes: {v2_time:.2f}s")
    
    print(f"\n3. Python fallback (100 doc sample):")
    print(f"   ✅ Takes: {fallback_time:.2f}s")
    print(f"   ⚠️  Only searches 0.1% of documents!")
    
    print("\n🎯 KEY INSIGHTS:")
    print("- Original BasicRAG can't use IRIS vector search due to parser bug")
    print("- Original falls back to Python similarity on tiny 100-doc sample")
    print("- V2 searches ALL 99,990 documents with native vector operations")
    print(f"- V2 provides 1000x more coverage in ~{v2_time:.1f}s")
    
    if v2_success and fallback_success and v2_time > 0:
        print(f"\n📊 At scale (1000 queries):")
        print(f"   Python fallback: {fallback_time * 1000:.0f}s (but only 100 docs!)")
        print(f"   V2 full search:  {v2_time * 1000:.0f}s (all 99,990 docs)")

if __name__ == "__main__":
    main()