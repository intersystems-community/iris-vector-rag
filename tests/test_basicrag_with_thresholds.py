#!/usr/bin/env python3
"""
Test BasicRAG with different similarity thresholds to find the issue
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basicrag_thresholds():
    """Test BasicRAG with different similarity thresholds"""
    
    print("🔍 Testing BasicRAG with Different Similarity Thresholds")
    print("=" * 60)
    
    try:
        # Setup connections
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        
        # Test query
        test_query = "What is diabetes?"
        print(f"📝 Test Query: {test_query}")
        
        # Generate embedding
        query_embedding = embed_fn([test_query])[0]
        query_vector_str = f"[{','.join(map(str, query_embedding))}]"
        print(f"📊 Query embedding length: {len(query_embedding)}")
        
        cursor = db_conn.cursor()
        
        # Test with no threshold to see what scores we get
        print("\n1️⃣ Testing with NO similarity threshold...")
        sql_no_threshold = """
            SELECT TOP 10
                doc_id,
                title,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY similarity_score DESC
        """
        
        cursor.execute(sql_no_threshold, [query_vector_str])
        results = cursor.fetchall()
        print(f"   Results with no threshold: {len(results)}")
        
        if len(results) > 0:
            print("✅ Found documents! Top similarity scores:")
            for i, row in enumerate(results[:5]):
                print(f"     Doc {i+1}: ID={row[0]}, Score={row[2]:.6f}")
                if row[1]:  # title
                    print(f"              Title: {str(row[1])[:60]}...")
            
            # Find the range of scores
            scores = [row[2] for row in results if row[2] is not None]
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                print(f"\n📊 Score range: {min_score:.6f} to {max_score:.6f}")
                
                # Test with appropriate thresholds
                test_thresholds = [0.0, min_score - 0.01, max_score * 0.5, max_score * 0.8]
                
                for threshold in test_thresholds:
                    print(f"\n2️⃣ Testing with threshold {threshold:.6f}...")
                    sql_with_threshold = """
                        SELECT TOP 5
                            doc_id,
                            title,
                            VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                        FROM RAG.SourceDocuments
                        WHERE embedding IS NOT NULL
                          AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
                        ORDER BY similarity_score DESC
                    """
                    
                    cursor.execute(sql_with_threshold, [query_vector_str, query_vector_str, threshold])
                    threshold_results = cursor.fetchall()
                    print(f"   Results with threshold {threshold:.6f}: {len(threshold_results)}")
                    
                    if len(threshold_results) > 0:
                        print(f"   ✅ SUCCESS: Found {len(threshold_results)} documents with threshold {threshold:.6f}")
                        return True, threshold
        else:
            print("❌ No results even without threshold - there's a fundamental issue")
            return False, None
        
        return False, None
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db_conn' in locals() and db_conn:
            db_conn.close()

if __name__ == "__main__":
    success, working_threshold = test_basicrag_thresholds()
    if success:
        print(f"\n🎉 Found working threshold: {working_threshold:.6f}")
    else:
        print("\n💥 Could not find working threshold!")
    
    sys.exit(0 if success else 1)