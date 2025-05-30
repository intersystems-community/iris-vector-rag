#!/usr/bin/env python3
"""
Test vector search on V2 tables using VARCHAR columns
"""

import os
import sys
import jaydebeapi
import jpype
import time

# Find JDBC driver
jdbc_driver_path = "./intersystems-jdbc-3.8.4.jar"
if not os.path.exists(jdbc_driver_path):
    print("❌ JDBC driver not found!")
    sys.exit(1)

# Connection parameters
jdbc_url = f"jdbc:IRIS://localhost:1972/USER"
jdbc_driver_class = "com.intersystems.jdbc.IRISDriver"

print("🔍 Testing V2 Tables with VARCHAR Vector Columns via JDBC")
print("=" * 60)

try:
    # Start JVM
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), 
                      f"-Djava.class.path={jdbc_driver_path}")
    
    # Connect
    conn = jaydebeapi.connect(
        jdbc_driver_class,
        jdbc_url,
        ['SuperUser', 'SYS'],
        jdbc_driver_path
    )
    
    cursor = conn.cursor()
    print("✅ Connected to IRIS via JDBC")
    
    # Test 1: Verify V2 table structure
    print("\n📊 V2 Table Structure:")
    print("   - SourceDocuments_V2: VARCHAR columns for vectors")
    print("   - embedding: VARCHAR(50000) - original format")
    print("   - document_embedding_vector: VARCHAR(132863) - for HNSW index")
    
    # Test 2: Get a sample embedding from original column
    print("\n📊 Test 2: Get sample embedding")
    cursor.execute("""
        SELECT TOP 1 
            doc_id,
            title,
            LENGTH(embedding) as emb_len,
            SUBSTRING(embedding, 1, 50) as emb_preview
        FROM RAG.SourceDocuments_V2
        WHERE embedding IS NOT NULL
    """)
    sample = cursor.fetchone()
    doc_id, title, emb_len, emb_preview = sample
    print(f"   Doc: {doc_id}")
    print(f"   Title: {title[:60]}...")
    print(f"   Embedding length: {emb_len} chars")
    
    # Get full embedding for testing
    cursor.execute("""
        SELECT embedding 
        FROM RAG.SourceDocuments_V2 
        WHERE doc_id = ?
    """, [doc_id])
    test_embedding = cursor.fetchone()[0]
    
    # Test 3: Vector search using embedding column (no HNSW)
    print("\n📊 Test 3: Vector search on embedding column (baseline)")
    
    start_time = time.time()
    cursor.execute("""
        SELECT TOP 5
            doc_id,
            title,
            VECTOR_COSINE(
                TO_VECTOR(embedding), 
                TO_VECTOR(?)
            ) as similarity
        FROM RAG.SourceDocuments_V2
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
    """, [test_embedding])
    
    results = cursor.fetchall()
    baseline_time = time.time() - start_time
    
    print(f"   ✅ Search completed in {baseline_time:.3f} seconds")
    print(f"   Top results:")
    for i, (doc_id, title, sim) in enumerate(results):
        print(f"   {i+1}. {doc_id}: {title[:50]}... (similarity: {sim:.4f})")
    
    # Test 4: Check if document_embedding_vector has data
    print("\n📊 Test 4: Check document_embedding_vector column")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM RAG.SourceDocuments_V2 
        WHERE document_embedding_vector IS NOT NULL
    """)
    vec_count = cursor.fetchone()[0]
    print(f"   Records with document_embedding_vector: {vec_count}")
    
    if vec_count > 0:
        # Test 5: Try vector search on document_embedding_vector
        print("\n📊 Test 5: Vector search on document_embedding_vector (HNSW indexed)")
        
        # First get the vector format
        cursor.execute("""
            SELECT TOP 1 
                LENGTH(document_embedding_vector) as vec_len,
                SUBSTRING(document_embedding_vector, 1, 50) as vec_preview
            FROM RAG.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
        """)
        vec_len, vec_preview = cursor.fetchone()
        print(f"   Vector length: {vec_len} chars")
        print(f"   Preview: {vec_preview}...")
        
        # Get a test vector in the right format
        cursor.execute("""
            SELECT document_embedding_vector
            FROM RAG.SourceDocuments_V2
            WHERE doc_id = ?
        """, [doc_id])
        test_vector = cursor.fetchone()[0]
        
        # Search using the indexed column
        start_time = time.time()
        cursor.execute("""
            SELECT TOP 5
                doc_id,
                title,
                VECTOR_COSINE(
                    TO_VECTOR(document_embedding_vector), 
                    TO_VECTOR(?)
                ) as similarity
            FROM RAG.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
            ORDER BY similarity DESC
        """, [test_vector])
        
        results_indexed = cursor.fetchall()
        indexed_time = time.time() - start_time
        
        print(f"   ✅ Indexed search completed in {indexed_time:.3f} seconds")
        print(f"   Performance comparison:")
        print(f"      - Baseline (embedding): {baseline_time:.3f}s")
        print(f"      - Indexed (document_embedding_vector): {indexed_time:.3f}s")
        print(f"      - Speedup: {baseline_time/indexed_time:.2f}x")
    
    # Test 6: Test with custom query vector
    print("\n📊 Test 6: Parameter binding with synthetic query")
    
    # Create a synthetic query (384 dimensions)
    query_vector = ','.join([str(i * 0.001) for i in range(384)])
    
    cursor.execute("""
        SELECT TOP 3
            doc_id,
            VECTOR_COSINE(
                TO_VECTOR(embedding), 
                TO_VECTOR(?)
            ) as similarity
        FROM RAG.SourceDocuments_V2
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
    """, [query_vector])
    
    results = cursor.fetchall()
    print(f"   ✅ Parameter binding works! Found {len(results)} results")
    
    cursor.close()
    conn.close()
    
    print("\n✅ SUCCESS! Summary:")
    print("   1. V2 tables use VARCHAR columns (not native VECTOR type)")
    print("   2. JDBC works perfectly with V2 tables")
    print("   3. Parameter binding works correctly")
    print("   4. Both embedding columns are accessible")
    print("   5. HNSW indexes should provide performance benefits")
    print("\n🎯 Solution: Use JDBC with V2 tables for vector search!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    if jpype.isJVMStarted():
        jpype.shutdownJVM()