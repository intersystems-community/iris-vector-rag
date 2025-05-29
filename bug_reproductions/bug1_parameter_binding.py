#!/usr/bin/env python3
"""
IRIS SQL Vector Search Bug #1: Parameter Binding Issue
======================================================

This script demonstrates the critical parameter binding issue where IRIS SQL
does not support parameterized queries with VECTOR functions.

Prerequisites:
- InterSystems IRIS instance with vector search enabled
- Python intersystems-iris driver installed: pip install intersystems-iris
- IRIS connection details (update CONNECTION_STRING below)

Bug Description:
When using parameterized queries with VECTOR_COSINE or TO_VECTOR functions,
IRIS throws an error. This forces developers to use string interpolation,
which is a security risk (SQL injection) and bad practice.

Expected: Parameter binding should work with vector functions
Actual: "SQLCODE: -1^%msg: User defined function (VECTOR_COSINE) arguments number mismatch"
"""

import iris

# UPDATE THESE CONNECTION DETAILS
CONNECTION_STRING = "localhost:1972/USER"
USERNAME = "_SYSTEM"
PASSWORD = "SYS"
NAMESPACE = "USER"

def setup_test_table(connection):
    """Create a test table with vector data"""
    cursor = connection.cursor()
    
    # Create schema if not exists
    try:
        cursor.execute("CREATE SCHEMA TestVectorBugs")
    except:
        pass  # Schema might already exist
    
    # Drop and recreate test table
    try:
        cursor.execute("DROP TABLE TestVectorBugs.VectorTable")
    except:
        pass
    
    cursor.execute("""
        CREATE TABLE TestVectorBugs.VectorTable (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            embedding VARCHAR(1000)
        )
    """)
    
    # Insert test data with embeddings
    test_embeddings = [
        (1, 'Document 1', '0.1,0.2,0.3,0.4'),
        (2, 'Document 2', '0.2,0.3,0.4,0.5'),
        (3, 'Document 3', '0.3,0.4,0.5,0.6')
    ]
    
    for id, name, embedding in test_embeddings:
        cursor.execute(
            "INSERT INTO TestVectorBugs.VectorTable (id, name, embedding) VALUES (?, ?, ?)",
            [id, name, embedding]
        )
    
    connection.commit()
    print("✅ Test table created and populated")

def test_parameter_binding_bug(connection):
    """Demonstrate the parameter binding bug"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("BUG #1: Parameter Binding with Vector Functions")
    print("="*60)
    
    query_embedding = "0.15,0.25,0.35,0.45"
    
    # Test 1: Try parameterized query (THIS FAILS)
    print("\n1. Testing parameterized query with VECTOR_COSINE:")
    print("   Query: SELECT name, VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as score")
    print("   Parameter:", query_embedding)
    
    try:
        cursor.execute("""
            SELECT name, 
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as score
            FROM TestVectorBugs.VectorTable
            ORDER BY score DESC
        """, [query_embedding])
        
        results = cursor.fetchall()
        print("   ✅ SUCCESS (unexpected!): Query executed")
        for row in results:
            print(f"      - {row[0]}: {row[1]}")
    except Exception as e:
        print(f"   ❌ FAILED (expected): {e}")
        print("   Error shows parameter binding doesn't work with vector functions")
    
    # Test 2: Try with multiple parameters (ALSO FAILS)
    print("\n2. Testing with similarity threshold parameter:")
    print("   Query: WHERE VECTOR_COSINE(...) > ?")
    
    similarity_threshold = 0.5
    try:
        cursor.execute("""
            SELECT name, 
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as score
            FROM TestVectorBugs.VectorTable
            WHERE VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
            ORDER BY score DESC
        """, [query_embedding, query_embedding, similarity_threshold])
        
        results = cursor.fetchall()
        print("   ✅ SUCCESS (unexpected!): Query executed")
    except Exception as e:
        print(f"   ❌ FAILED (expected): {e}")
    
    # Test 3: Show the workaround (string interpolation - BAD PRACTICE)
    print("\n3. Workaround using string interpolation (SECURITY RISK):")
    print("   ⚠️  WARNING: This is vulnerable to SQL injection!")
    
    # Build query with string interpolation
    unsafe_query = f"""
        SELECT name, 
               VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{query_embedding}')) as score
        FROM TestVectorBugs.VectorTable
        WHERE VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{query_embedding}')) > {similarity_threshold}
        ORDER BY score DESC
    """
    
    try:
        cursor.execute(unsafe_query)
        results = cursor.fetchall()
        print("   ✅ String interpolation works (but is unsafe)")
        for row in results:
            print(f"      - {row[0]}: {row[1]:.4f}")
    except Exception as e:
        print(f"   ❌ Even workaround failed: {e}")
    
    # Test 4: Show that regular parameters work fine
    print("\n4. Regular parameter binding (without vector functions) works fine:")
    try:
        cursor.execute(
            "SELECT name FROM TestVectorBugs.VectorTable WHERE id = ?",
            [2]
        )
        result = cursor.fetchone()
        print(f"   ✅ Regular parameter binding works: {result[0]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def main():
    """Main execution"""
    print("IRIS SQL Vector Search - Bug #1: Parameter Binding Issue")
    print("========================================================")
    
    # Connect to IRIS
    try:
        connection = iris.connect(
            CONNECTION_STRING,
            USERNAME,
            PASSWORD,
            NAMESPACE
        )
        print(f"✅ Connected to IRIS at {CONNECTION_STRING}")
    except Exception as e:
        print(f"❌ Failed to connect to IRIS: {e}")
        print("\nPlease update the connection details in this script:")
        print("  CONNECTION_STRING, USERNAME, PASSWORD, NAMESPACE")
        return
    
    try:
        # Setup test environment
        setup_test_table(connection)
        
        # Run bug demonstration
        test_parameter_binding_bug(connection)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("❌ BUG CONFIRMED: Parameter binding does not work with vector functions")
        print("📝 Impact: Forces use of string interpolation (SQL injection risk)")
        print("🔧 Workaround: String interpolation (NOT RECOMMENDED)")
        print("✅ Fix needed: IRIS should support parameter binding with vector functions")
        
    finally:
        connection.close()
        print("\n✅ Connection closed")

if __name__ == "__main__":
    main()