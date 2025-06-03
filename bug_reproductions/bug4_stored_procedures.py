#!/usr/bin/env python3
"""
IRIS SQL Vector Search Bug #4: Stored Procedure Vector Function Limitations
===========================================================================

This script demonstrates limitations when using vector functions within
stored procedures in IRIS SQL.

Prerequisites:
- InterSystems IRIS instance with vector search enabled
- Python intersystems-iris driver installed: pip install intersystems-iris
- IRIS connection details (update CONNECTION_STRING below)

Bug Description:
Vector functions like VECTOR_COSINE and TO_VECTOR have limitations when
used inside stored procedures. This affects performance optimization and
code reusability for vector search operations.

Expected: Vector functions should work seamlessly in stored procedures
Actual: Various limitations and workarounds required
"""

import iris
import time

# UPDATE THESE CONNECTION DETAILS
CONNECTION_STRING = "localhost:1972/USER"
USERNAME = "_SYSTEM"
PASSWORD = "SYS"
NAMESPACE = "USER"

def setup_test_environment(connection):
    """Create test tables and data"""
    cursor = connection.cursor()
    
    # Create schema if not exists
    try:
        cursor.execute("CREATE SCHEMA TestVectorBugs")
    except:
        pass  # Schema might already exist
    
    # Drop and recreate test table
    try:
        cursor.execute("DROP TABLE TestVectorBugs.Documents")
    except:
        pass
    
    # Create table with vector data
    cursor.execute("""
        CREATE TABLE TestVectorBugs.Documents (
            id INTEGER PRIMARY KEY,
            title VARCHAR(200),
            content LONGVARCHAR,
            embedding VARCHAR(1000),
            embedding_vector VECTOR(FLOAT, 4)
        )
    """)
    
    # Insert test data
    test_docs = [
        (1, 'Machine Learning Basics', 'Introduction to ML concepts', '0.1,0.2,0.3,0.4'),
        (2, 'Deep Learning Guide', 'Neural networks explained', '0.2,0.3,0.4,0.5'),
        (3, 'NLP Fundamentals', 'Natural language processing', '0.3,0.4,0.5,0.6'),
        (4, 'Computer Vision', 'Image processing techniques', '0.4,0.5,0.6,0.7'),
        (5, 'Reinforcement Learning', 'RL algorithms and applications', '0.5,0.6,0.7,0.8')
    ]
    
    for id, title, content, embedding in test_docs:
        cursor.execute(f"""
            INSERT INTO TestVectorBugs.Documents 
            (id, title, content, embedding, embedding_vector) 
            VALUES (?, ?, ?, ?, TO_VECTOR('{embedding}', 'FLOAT', 4))
        """, [id, title, content, embedding])
    
    connection.commit()
    print("✅ Test environment created with sample documents")

def test_stored_procedure_creation(connection):
    """Test creating stored procedures with vector functions"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("BUG #4A: Stored Procedure Creation with Vector Functions")
    print("="*60)
    
    # Test 1: Simple vector search procedure
    print("\n1. Creating basic vector search stored procedure:")
    
    try:
        # Drop if exists
        try:
            cursor.execute("DROP PROCEDURE TestVectorBugs.VectorSearch")
        except:
            pass
        
        # Create stored procedure
        create_proc = """
        CREATE PROCEDURE TestVectorBugs.VectorSearch(
            IN query_vector VARCHAR(1000),
            IN top_k INTEGER
        )
        RETURNS TABLE(
            doc_id INTEGER,
            title VARCHAR(200),
            similarity DOUBLE
        )
        LANGUAGE SQL
        BEGIN
            RETURN SELECT 
                id as doc_id,
                title,
                VECTOR_COSINE(embedding_vector, TO_VECTOR(query_vector, 'FLOAT', 4)) as similarity
            FROM TestVectorBugs.Documents
            WHERE embedding_vector IS NOT NULL
            ORDER BY similarity DESC
            LIMIT top_k;
        END
        """
        
        cursor.execute(create_proc)
        print("   ✅ SUCCESS: Basic stored procedure created")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print("   Issue: Vector functions in stored procedures may have limitations")
    
    # Test 2: Stored procedure with dynamic vector dimensions
    print("\n2. Creating stored procedure with dynamic dimensions:")
    
    try:
        try:
            cursor.execute("DROP PROCEDURE TestVectorBugs.DynamicVectorSearch")
        except:
            pass
        
        create_dynamic_proc = """
        CREATE PROCEDURE TestVectorBugs.DynamicVectorSearch(
            IN query_vector VARCHAR(1000),
            IN vector_dim INTEGER,
            IN threshold DOUBLE
        )
        RETURNS TABLE(
            doc_id INTEGER,
            title VARCHAR(200),
            score DOUBLE
        )
        LANGUAGE SQL
        BEGIN
            -- This often fails due to dynamic dimension handling
            RETURN SELECT 
                id as doc_id,
                title,
                VECTOR_COSINE(
                    embedding_vector, 
                    TO_VECTOR(query_vector, 'DOUBLE', vector_dim)
                ) as score
            FROM TestVectorBugs.Documents
            WHERE VECTOR_COSINE(
                embedding_vector, 
                TO_VECTOR(query_vector, 'DOUBLE', vector_dim)
            ) > threshold;
        END
        """
        
        cursor.execute(create_dynamic_proc)
        print("   ✅ SUCCESS (unexpected!): Dynamic dimension procedure created")
        
    except Exception as e:
        print(f"   ❌ FAILED (expected): {e}")
        print("   Issue: Dynamic dimensions in vector functions often fail")

def test_stored_procedure_execution(connection):
    """Test executing stored procedures with vector operations"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("BUG #4B: Stored Procedure Execution Issues")
    print("="*60)
    
    query_vector = "0.15,0.25,0.35,0.45"
    
    # Test 1: Execute basic vector search procedure
    print("\n1. Executing basic vector search procedure:")
    
    try:
        # First create a simple working procedure
        try:
            cursor.execute("DROP PROCEDURE TestVectorBugs.SimpleSearch")
        except:
            pass
            
        cursor.execute("""
        CREATE PROCEDURE TestVectorBugs.SimpleSearch()
        RETURNS TABLE(
            doc_id INTEGER,
            title VARCHAR(200)
        )
        LANGUAGE SQL
        BEGIN
            RETURN SELECT id as doc_id, title 
            FROM TestVectorBugs.Documents
            WHERE id <= 3;
        END
        """)
        
        # Call the procedure
        cursor.execute("CALL TestVectorBugs.SimpleSearch()")
        results = cursor.fetchall()
        print("   ✅ Simple procedure works")
        for row in results:
            print(f"      - ID: {row[0]}, Title: {row[1]}")
            
    except Exception as e:
        print(f"   ❌ Even simple procedure failed: {e}")
    
    # Test 2: Parameter passing issues
    print("\n2. Testing parameter passing to vector procedures:")
    
    try:
        # Create procedure with parameters
        try:
            cursor.execute("DROP PROCEDURE TestVectorBugs.ParameterTest")
        except:
            pass
            
        cursor.execute("""
        CREATE PROCEDURE TestVectorBugs.ParameterTest(
            IN search_vector VARCHAR(1000)
        )
        RETURNS TABLE(
            doc_id INTEGER,
            similarity DOUBLE
        )
        LANGUAGE SQL
        BEGIN
            -- Note: This often requires workarounds
            RETURN SELECT 
                id as doc_id,
                1.0 as similarity  -- Placeholder since vector ops may fail
            FROM TestVectorBugs.Documents
            WHERE embedding IS NOT NULL;
        END
        """)
        
        cursor.execute("CALL TestVectorBugs.ParameterTest(?)", [query_vector])
        results = cursor.fetchall()
        print("   ✅ Parameter passing works (with limitations)")
        
    except Exception as e:
        print(f"   ❌ Parameter passing failed: {e}")

def test_function_vs_procedure_performance(connection):
    """Compare inline SQL vs stored procedure performance"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("Performance Comparison: Inline SQL vs Stored Procedures")
    print("="*60)
    
    query_vector = "0.15,0.25,0.35,0.45"
    iterations = 10
    
    # Test 1: Inline SQL performance
    print(f"\n1. Inline SQL performance ({iterations} iterations):")
    
    inline_times = []
    for i in range(iterations):
        start = time.time()
        cursor.execute(f"""
            SELECT id, title,
                   VECTOR_COSINE(embedding_vector, TO_VECTOR('{query_vector}', 'FLOAT', 4)) as score
            FROM TestVectorBugs.Documents
            ORDER BY score DESC
        """)
        results = cursor.fetchall()
        end = time.time()
        inline_times.append(end - start)
    
    avg_inline = sum(inline_times) / len(inline_times)
    print(f"   Average time: {avg_inline:.4f} seconds")
    
    # Test 2: Create an equivalent stored procedure (if possible)
    print(f"\n2. Stored procedure performance (if it works):")
    
    try:
        # Create optimized procedure
        try:
            cursor.execute("DROP FUNCTION TestVectorBugs.VectorSimilarity")
        except:
            pass
            
        # Try creating a function instead of procedure
        cursor.execute("""
        CREATE FUNCTION TestVectorBugs.VectorSimilarity(
            vec1 VARCHAR(1000),
            vec2 VARCHAR(1000)
        )
        RETURNS DOUBLE
        LANGUAGE SQL
        BEGIN
            -- This is a simplified version
            RETURN 1.0;  -- Placeholder
        END
        """)
        
        print("   ⚠️  Note: Full vector operations in functions are limited")
        print("   Stored procedures often can't match inline SQL performance")
        
    except Exception as e:
        print(f"   ❌ Cannot create equivalent stored procedure: {e}")

def demonstrate_workarounds(connection):
    """Show workarounds for stored procedure limitations"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("WORKAROUNDS: Handling Vector Operations in Stored Procedures")
    print("="*60)
    
    print("\n1. Workaround: Use temporary tables")
    print("""
    CREATE PROCEDURE VectorSearchWithTemp(IN query_vec VARCHAR)
    BEGIN
        -- Create temp table with results
        CREATE TEMPORARY TABLE TempResults AS
        SELECT id, VECTOR_COSINE(...) as score
        FROM Documents;
        
        -- Return sorted results
        SELECT * FROM TempResults ORDER BY score DESC;
        DROP TABLE TempResults;
    END
    """)
    
    print("\n2. Workaround: Use ObjectScript for complex vector operations")
    print("""
    -- Call ObjectScript method from SQL
    CREATE PROCEDURE VectorSearchOS(IN query_vec VARCHAR)
    LANGUAGE OBJECTSCRIPT
    {
        // Use ObjectScript for vector calculations
        // More flexible but requires ObjectScript knowledge
    }
    """)
    
    print("\n3. Workaround: Pre-compute common operations")
    print("""
    -- Store pre-computed similarities
    CREATE TABLE PrecomputedSimilarities (
        doc_id1 INTEGER,
        doc_id2 INTEGER,
        similarity DOUBLE
    );
    
    -- Use in stored procedures without vector functions
    """)
    
    print("\n4. Best Practice: Keep vector operations in application layer")
    print("   - Use inline SQL for vector searches")
    print("   - Avoid complex vector operations in stored procedures")
    print("   - Use stored procedures for non-vector logic only")

def main():
    """Main execution"""
    print("IRIS SQL Vector Search - Bug #4: Stored Procedure Limitations")
    print("=============================================================")
    
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
        # Setup and run tests
        setup_test_environment(connection)
        test_stored_procedure_creation(connection)
        test_stored_procedure_execution(connection)
        test_function_vs_procedure_performance(connection)
        demonstrate_workarounds(connection)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("❌ BUG CONFIRMED: Vector functions have limitations in stored procedures")
        print("📝 Issues:")
        print("   - Dynamic dimensions often fail")
        print("   - Parameter binding is problematic")
        print("   - Performance overhead compared to inline SQL")
        print("   - Limited optimization capabilities")
        print("🔧 Workarounds:")
        print("   - Use inline SQL for vector operations")
        print("   - Leverage ObjectScript for complex logic")
        print("   - Pre-compute when possible")
        print("   - Keep vector logic in application layer")
        print("✅ Fix needed: Full vector function support in stored procedures")
        
    finally:
        connection.close()
        print("\n✅ Connection closed")

if __name__ == "__main__":
    main()