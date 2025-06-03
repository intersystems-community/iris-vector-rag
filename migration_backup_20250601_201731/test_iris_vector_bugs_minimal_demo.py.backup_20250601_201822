#!/usr/bin/env python3
"""
Minimal demonstration of IRIS vector search bugs using intersystems-irispython
Shows the issues with TO_VECTOR() on VARCHAR columns

Install: pip install intersystems-irispython
"""

import iris

def main():
    print("🔍 IRIS Vector Search Bug Demonstration")
    print("=" * 60)
    
    # Connection parameters
    args = {
        'hostname': '127.0.0.1', 
        'port': 1972,
        'namespace': 'USER', 
        'username': '_SYSTEM', 
        'password': 'SYS'
    }
    
    # Connect to IRIS
    print("\n📊 Connecting to IRIS...")
    conn = iris.connect(**args)
    cursor = conn.cursor()
    print("✅ Connected successfully")
    
    # Setup test environment
    setup_test_environment(cursor, conn)
    
    # Demonstrate the bugs
    print("\n" + "=" * 60)
    print("🐛 DEMONSTRATING IRIS VECTOR SEARCH BUGS")
    print("=" * 60)
    
    # Bug 1: Literal string works
    test_bug_1_literal_works(cursor)
    
    # Bug 2: Parameter marker fails
    test_bug_2_parameter_fails(cursor)
    
    # Bug 3: Long vectors fail
    test_bug_3_long_vectors_fail(cursor)
    
    # Bug 4: TOP clause cannot be parameterized
    test_bug_4_top_clause_fails(cursor)
    
    # Show the workaround
    show_workaround(cursor)
    
    # Cleanup
    cleanup(cursor, conn)
    
    print("\n✅ Demonstration complete!")

def setup_test_environment(cursor, conn):
    """Setup test schema and tables"""
    print("\n🔧 Setting up test environment...")
    
    # Create schema
    try:
        cursor.execute("CREATE SCHEMA TEST_VECTOR")
    except:
        pass  # Schema might already exist
    
    # Drop existing tables
    try:
        cursor.execute("DROP TABLE TEST_VECTOR.test_embeddings")
    except:
        pass
    
    try:
        cursor.execute("DROP TABLE TEST_VECTOR.test_embeddings_v2")
    except:
        pass
    
    # Create table with VARCHAR embedding column (like current RAG schema)
    cursor.execute("""
        CREATE TABLE TEST_VECTOR.test_embeddings (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            embedding VARCHAR(50000)
        )
    """)
    
    # Insert test data
    cursor.execute("""
        INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
        VALUES (1, 'test1', '0.1,0.2,0.3')
    """)
    
    cursor.execute("""
        INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
        VALUES (2, 'test2', '0.4,0.5,0.6')
    """)
    
    # Create a longer embedding for bug #3
    long_embedding = ','.join([str(i * 0.001) for i in range(384)])
    cursor.execute("""
        INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
        VALUES (3, 'test_long', ?)
    """, [long_embedding])
    
    conn.commit()
    print("✅ Test environment ready")

def test_bug_1_literal_works(cursor):
    """Bug #1: TO_VECTOR() with literal string works"""
    print("\n🧪 Bug #1: Testing TO_VECTOR() with literal string...")
    
    try:
        cursor.execute("""
            SELECT id, name, 
                   VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 3), 
                                 TO_VECTOR('0.1,0.2,0.3', 'DOUBLE', 3)) as similarity
            FROM TEST_VECTOR.test_embeddings
            WHERE id <= 2
        """)
        
        results = cursor.fetchall()
        print("✅ SUCCESS: Query with literal string works!")
        for row in results:
            print(f"   ID: {row[0]}, Name: {row[1]}, Similarity: {row[2]:.4f}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

def test_bug_2_parameter_fails(cursor):
    """Bug #2: TO_VECTOR() with parameter marker fails"""
    print("\n🧪 Bug #2: Testing TO_VECTOR() with parameter marker...")
    
    try:
        # This should fail with "colon found" error
        cursor.execute("""
            SELECT id, name, 
                   VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 3), 
                                 TO_VECTOR(?, 'DOUBLE', 3)) as similarity
            FROM TEST_VECTOR.test_embeddings
            WHERE id <= 2
        """, ['0.1,0.2,0.3'])
        
        results = cursor.fetchall()
        print("✅ UNEXPECTED: Query with parameter worked!")
    except Exception as e:
        print(f"❌ EXPECTED FAILURE: {e}")
        if "colon" in str(e).lower():
            print("   ⚠️  This is the 'colon found' bug!")

def test_bug_3_long_vectors_fail(cursor):
    """Bug #3: Long vectors fail even with string interpolation"""
    print("\n🧪 Bug #3: Testing TO_VECTOR() with long vectors...")
    
    # Generate a 384-dimensional vector (typical for sentence embeddings)
    long_vector = ','.join([str(i * 0.001) for i in range(384)])
    
    try:
        # Build query with string interpolation (no parameters)
        query = f"""
            SELECT id, name, 
                   VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 384), 
                                 TO_VECTOR('{long_vector}', 'DOUBLE', 384)) as similarity
            FROM TEST_VECTOR.test_embeddings
            WHERE id = 3
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        print("✅ SUCCESS: Long vector query worked!")
        for row in results:
            print(f"   ID: {row[0]}, Name: {row[1]}, Similarity: {row[2]:.4f}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        if "colon" in str(e).lower():
            print("   ⚠️  IRIS incorrectly interprets the long vector string as containing parameter markers!")

def test_bug_4_top_clause_fails(cursor):
    """Bug #4: TOP clause cannot be parameterized"""
    print("\n🧪 Bug #4: Testing parameterized TOP clause...")
    
    try:
        cursor.execute("SELECT TOP ? * FROM TEST_VECTOR.test_embeddings", [2])
        results = cursor.fetchall()
        print("✅ UNEXPECTED: Parameterized TOP worked!")
    except Exception as e:
        print(f"❌ EXPECTED FAILURE: {e}")
        print("   ⚠️  TOP clause does not accept parameter markers!")

def show_workaround(cursor):
    """Show the workaround that BasicRAG uses"""
    print("\n🔧 Workaround: Load embeddings and calculate similarity in Python")
    print("   (This is what BasicRAG does)")
    
    # Load all embeddings
    cursor.execute("""
        SELECT id, name, embedding 
        FROM TEST_VECTOR.test_embeddings
        WHERE embedding IS NOT NULL
    """)
    
    rows = cursor.fetchall()
    
    # Calculate cosine similarity in Python
    query_vector = [0.1, 0.2, 0.3]
    results = []
    
    for row in rows:
        doc_id, name, embedding_str = row
        # Parse embedding
        doc_vector = [float(x) for x in embedding_str.split(',')][:3]  # Take first 3 for comparison
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
        query_norm = sum(a * a for a in query_vector) ** 0.5
        doc_norm = sum(a * a for a in doc_vector) ** 0.5
        
        if query_norm > 0 and doc_norm > 0:
            similarity = dot_product / (query_norm * doc_norm)
            results.append((doc_id, name, similarity))
    
    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("\n✅ Python-calculated similarities:")
    for doc_id, name, similarity in results[:2]:
        print(f"   ID: {doc_id}, Name: {name}, Similarity: {similarity:.4f}")

def cleanup(cursor, conn):
    """Cleanup test environment"""
    print("\n🧹 Cleaning up...")
    try:
        cursor.execute("DROP TABLE TEST_VECTOR.test_embeddings")
        cursor.execute("DROP TABLE TEST_VECTOR.test_embeddings_v2")
        cursor.execute("DROP SCHEMA TEST_VECTOR")
        conn.commit()
    except:
        pass

if __name__ == "__main__":
    main()