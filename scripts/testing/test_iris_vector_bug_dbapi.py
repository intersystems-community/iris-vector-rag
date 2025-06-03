#!/usr/bin/env python3
"""
Send exact SQL queries to IRIS via dbapi to demonstrate vector search bugs
This sends queries as-is over the wire without any parameter substitution
"""

import iris

def execute_query(cursor, query, description):
    """Execute a query and handle the expected error"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"SQL: {query[:200]}..." if len(query) > 200 else f"SQL: {query}")
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print("✅ SUCCESS (unexpected!)")
        for row in results[:3]:  # Show first 3 rows
            print(f"   {row}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        if "colon" in str(e).lower() or ":%qpar" in str(e):
            print("   ⚠️  This is the 'colon found' bug!")
        return str(e)
    return None

def main():
    print("🔍 IRIS Vector Search Bug Demonstration via DB-API")
    print("Sending exact SQL queries over the wire")
    
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
    
    # Setup: Create test environment
    print("\n🔧 Setting up test environment...")
    
    # Create schema (ignore error if exists)
    try:
        cursor.execute("CREATE SCHEMA TEST_VECTOR")
    except:
        pass
    
    # Drop table if exists
    try:
        cursor.execute("DROP TABLE TEST_VECTOR.test_embeddings")
    except:
        pass
    
    # Create table
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
    
    conn.commit()
    print("✅ Test environment ready")
    
    # Test 1: Basic TO_VECTOR with literal string
    query1 = """
        SELECT id, name, 
               VECTOR_COSINE(TO_VECTOR(embedding, 'FLOAT', 3), 
                             TO_VECTOR('0.1,0.2,0.3', 'DOUBLE', 3)) as similarity
        FROM TEST_VECTOR.test_embeddings
        WHERE id <= 2
    """
    error1 = execute_query(cursor, query1, "Test 1: Basic TO_VECTOR with literal string")
    
    # Test 2: Just TO_VECTOR on column
    query2 = """
        SELECT id, name, TO_VECTOR(embedding, 'FLOAT', 3) as vector_result
        FROM TEST_VECTOR.test_embeddings
        WHERE id = 1
    """
    error2 = execute_query(cursor, query2, "Test 2: TO_VECTOR on column only")
    
    # Test 3: Try without quotes around DOUBLE
    query3 = """
        SELECT id, name, TO_VECTOR(embedding, DOUBLE, 3) as vector_result
        FROM TEST_VECTOR.test_embeddings
        WHERE id = 1
    """
    error3 = execute_query(cursor, query3, "Test 3: TO_VECTOR without quotes on DOUBLE")
    
    # Test 4: Direct VECTOR_COSINE on VARCHAR (should fail differently)
    query4 = """
        SELECT id, name, 
               VECTOR_COSINE(embedding, embedding) as similarity
        FROM TEST_VECTOR.test_embeddings
        WHERE id <= 2
    """
    error4 = execute_query(cursor, query4, "Test 4: Direct VECTOR_COSINE on VARCHAR")
    
    # Test 5: What BasicRAG does - just load the data
    query5 = """
        SELECT id, name, embedding 
        FROM TEST_VECTOR.test_embeddings
        WHERE embedding IS NOT NULL
    """
    execute_query(cursor, query5, "Test 5: BasicRAG approach - load embeddings as strings")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY OF RESULTS")
    print("="*60)
    
    if error1 and "colon" in error1.lower():
        print("✅ Confirmed: TO_VECTOR() has the 'colon found' bug")
        print("   IRIS incorrectly interprets 'DOUBLE' as containing a parameter marker")
    
    print("\n🔧 WORKAROUND:")
    print("   BasicRAG avoids TO_VECTOR() entirely")
    print("   Loads embeddings as strings and calculates similarity in Python")
    
    print("\n🚀 SOLUTION:")
    print("   Migration to native VECTOR columns (the _V2 tables)")
    print("   This will allow direct vector operations without TO_VECTOR()")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    try:
        cursor.execute("DROP TABLE TEST_VECTOR.test_embeddings")
        cursor.execute("DROP SCHEMA TEST_VECTOR")
        conn.commit()
    except:
        pass
    
    cursor.close()
    conn.close()
    print("✅ Done!")

if __name__ == "__main__":
    main()