#!/usr/bin/env python3
"""
Systematic test of different HNSW SQL syntax permutations for IRIS
"""

import sys
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

def test_sql_syntax(cursor, sql_query, description):
    """Test a specific SQL syntax"""
    print(f"\n🧪 Testing: {description}")
    print(f"SQL: {sql_query[:200]}...")
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        print(f"✅ SUCCESS! Retrieved {len(results)} results")
        if results:
            print(f"   First result: doc_id={results[0][0][:50]}..., score={results[0][2]:.4f}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("🔍 Systematic HNSW SQL Syntax Testing for IRIS")
    print("=" * 60)
    
    # Get connection and embedding
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    # Generate test embedding
    query = "diabetes treatment"
    query_embedding = embedding_func([query])[0]
    query_embedding_str = ','.join(map(str, query_embedding))
    
    print(f"📊 Query: '{query}'")
    print(f"📊 Embedding dimensions: {len(query_embedding)}")
    print(f"📊 Embedding string length: {len(query_embedding_str)}")
    
    # Test different SQL syntax permutations
    tests = [
        # Option 1: Direct embedding string in TO_VECTOR
        (
            f"""
            SELECT doc_id, text_content, 
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) AS similarity_score 
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL 
            AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) > 0.1
            ORDER BY similarity_score DESC
            """,
            "Option 1: Direct embedding string in TO_VECTOR"
        ),
        
        # Option 2: Using parameter placeholders
        (
            """
            SELECT doc_id, text_content, 
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE')) AS similarity_score 
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL 
            AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE')) > ?
            ORDER BY similarity_score DESC
            """,
            "Option 2: Using parameter placeholders"
        ),
        
        # Option 3: Without TO_VECTOR on the column (assuming it's already VECTOR type)
        (
            f"""
            SELECT doc_id, text_content, 
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}')) AS similarity_score 
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL 
            ORDER BY similarity_score DESC
            """,
            "Option 3: Without 'DOUBLE' parameter in TO_VECTOR"
        ),
        
        # Option 4: Using VECTOR_DOT_PRODUCT instead
        (
            f"""
            SELECT doc_id, text_content, 
            VECTOR_DOT_PRODUCT(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) AS similarity_score 
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL 
            ORDER BY similarity_score DESC
            """,
            "Option 4: Using VECTOR_DOT_PRODUCT"
        ),
        
        # Option 5: Simple query without WHERE clause on similarity
        (
            f"""
            SELECT TOP 10 doc_id, text_content, 
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) AS similarity_score 
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL 
            ORDER BY similarity_score DESC
            """,
            "Option 5: Using TOP 10 without similarity threshold"
        ),
    ]
    
    successful_options = []
    
    for sql, description in tests:
        if test_sql_syntax(cursor, sql, description):
            successful_options.append(description)
    
    # Test Option 2 with parameters
    print("\n🧪 Testing: Option 2 with actual parameters")
    try:
        sql = """
            SELECT doc_id, text_content, 
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE')) AS similarity_score 
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL 
            AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE')) > ?
            ORDER BY similarity_score DESC
        """
        cursor.execute(sql, (query_embedding_str, query_embedding_str, 0.1))
        results = cursor.fetchall()
        print(f"✅ SUCCESS with parameters! Retrieved {len(results)} results")
        if results:
            print(f"   First result: doc_id={results[0][0][:50]}..., score={results[0][2]:.4f}")
        successful_options.append("Option 2 with parameters")
    except Exception as e:
        print(f"❌ FAILED with parameters: {e}")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"✅ Successful options: {len(successful_options)}")
    for opt in successful_options:
        print(f"   - {opt}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()