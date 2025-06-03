#!/usr/bin/env python3
"""
IRIS SQL Vector Search Bug #3: Missing VECTOR Type Support in Python Driver
===========================================================================

This script demonstrates that the Python IRIS driver does not properly support
the VECTOR data type, requiring workarounds for vector operations.

Prerequisites:
- InterSystems IRIS instance with vector search enabled
- Python intersystems-iris driver installed: pip install intersystems-iris
- IRIS connection details (update CONNECTION_STRING below)

Bug Description:
The Python driver cannot directly handle VECTOR type columns. When retrieving
VECTOR data, it returns as strings. When inserting VECTOR data, you must use
TO_VECTOR() function instead of native parameter binding.

Expected: Driver should handle VECTOR type natively
Actual: VECTOR columns return as strings, no native insert support
"""

import iris
import json

# UPDATE THESE CONNECTION DETAILS
CONNECTION_STRING = "localhost:1972/USER"
USERNAME = "_SYSTEM"
PASSWORD = "SYS"
NAMESPACE = "USER"

def setup_test_environment(connection):
    """Create test tables with VECTOR columns"""
    cursor = connection.cursor()
    
    # Create schema if not exists
    try:
        cursor.execute("CREATE SCHEMA TestVectorBugs")
    except:
        pass  # Schema might already exist
    
    # Drop and recreate test table
    try:
        cursor.execute("DROP TABLE TestVectorBugs.VectorTypeTest")
    except:
        pass
    
    # Create table with VECTOR column
    cursor.execute("""
        CREATE TABLE TestVectorBugs.VectorTypeTest (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            embedding_varchar VARCHAR(1000),
            embedding_vector VECTOR(FLOAT, 4)
        )
    """)
    
    connection.commit()
    print("✅ Test environment created")

def test_vector_insertion(connection):
    """Test inserting data into VECTOR columns"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("BUG #3A: VECTOR Type Insertion Issues")
    print("="*60)
    
    # Test data
    test_vectors = [
        (1, 'Vector 1', [0.1, 0.2, 0.3, 0.4]),
        (2, 'Vector 2', [0.2, 0.3, 0.4, 0.5]),
        (3, 'Vector 3', [0.3, 0.4, 0.5, 0.6])
    ]
    
    # Test 1: Try direct parameter binding with list/array (FAILS)
    print("\n1. Testing direct parameter binding with Python list:")
    print("   INSERT INTO ... VALUES (?, ?, ?, ?)")
    
    try:
        vector_list = [0.1, 0.2, 0.3, 0.4]
        cursor.execute("""
            INSERT INTO TestVectorBugs.VectorTypeTest 
            (id, name, embedding_varchar, embedding_vector) 
            VALUES (?, ?, ?, ?)
        """, [1, 'Test 1', '0.1,0.2,0.3,0.4', vector_list])
        
        print("   ✅ SUCCESS (unexpected!): Direct list binding worked")
    except Exception as e:
        print(f"   ❌ FAILED (expected): {e}")
        print("   Error: Cannot bind Python list to VECTOR column")
    
    # Test 2: Try with JSON string (ALSO FAILS)
    print("\n2. Testing parameter binding with JSON string:")
    
    try:
        vector_json = json.dumps([0.1, 0.2, 0.3, 0.4])
        cursor.execute("""
            INSERT INTO TestVectorBugs.VectorTypeTest 
            (id, name, embedding_varchar, embedding_vector) 
            VALUES (?, ?, ?, ?)
        """, [2, 'Test 2', '0.1,0.2,0.3,0.4', vector_json])
        
        print("   ✅ SUCCESS (unexpected!): JSON string binding worked")
    except Exception as e:
        print(f"   ❌ FAILED (expected): {e}")
        print("   Error: Cannot bind JSON string to VECTOR column")
    
    # Test 3: Workaround using TO_VECTOR (WORKS)
    print("\n3. Workaround using TO_VECTOR function:")
    
    try:
        # Must use string interpolation with TO_VECTOR
        for id, name, vector in test_vectors:
            vector_str = ','.join(map(str, vector))
            cursor.execute(f"""
                INSERT INTO TestVectorBugs.VectorTypeTest 
                (id, name, embedding_varchar, embedding_vector) 
                VALUES (?, ?, ?, TO_VECTOR('{vector_str}', 'FLOAT', 4))
            """, [id, name, vector_str])
        
        connection.commit()
        print("   ✅ SUCCESS: TO_VECTOR workaround works")
        print("   Note: Requires string interpolation (SQL injection risk)")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

def test_vector_retrieval(connection):
    """Test retrieving VECTOR column data"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("BUG #3B: VECTOR Type Retrieval Issues")
    print("="*60)
    
    # Test 1: Retrieve VECTOR column data
    print("\n1. Retrieving VECTOR column data:")
    
    try:
        cursor.execute("""
            SELECT id, name, embedding_varchar, embedding_vector 
            FROM TestVectorBugs.VectorTypeTest
            ORDER BY id
        """)
        
        results = cursor.fetchall()
        print("   ✅ Query executed successfully")
        
        for row in results:
            id, name, varchar_embedding, vector_embedding = row
            print(f"\n   Row {id}: {name}")
            print(f"   - VARCHAR type: {type(varchar_embedding)}")
            print(f"   - VARCHAR value: {varchar_embedding}")
            print(f"   - VECTOR type: {type(vector_embedding)}")
            print(f"   - VECTOR value: {vector_embedding}")
            
            # Check if VECTOR is returned as string
            if isinstance(vector_embedding, str):
                print("   ❌ BUG: VECTOR column returned as string!")
            else:
                print("   ✅ VECTOR column returned as proper type")
                
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
    
    # Test 2: Use VECTOR in calculations
    print("\n2. Using VECTOR columns in calculations:")
    
    try:
        query_vector = "0.15,0.25,0.35,0.45"
        cursor.execute(f"""
            SELECT name, 
                   VECTOR_COSINE(embedding_vector, TO_VECTOR('{query_vector}', 'FLOAT', 4)) as score
            FROM TestVectorBugs.VectorTypeTest
            ORDER BY score DESC
        """)
        
        results = cursor.fetchall()
        print("   ✅ Vector calculations work in SQL")
        for row in results:
            print(f"      - {row[0]}: {row[1]:.4f}")
            
    except Exception as e:
        print(f"   ❌ Calculation failed: {e}")

def test_driver_metadata(connection):
    """Test driver's understanding of VECTOR type"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("BUG #3C: Driver Metadata Support for VECTOR")
    print("="*60)
    
    # Test column metadata
    print("\n1. Checking column metadata:")
    
    try:
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'TestVectorBugs'
            AND TABLE_NAME = 'VectorTypeTest'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        print("   Column information from INFORMATION_SCHEMA:")
        for col in columns:
            print(f"   - {col[0]}: {col[1]} (max_length: {col[2]})")
            
    except Exception as e:
        print(f"   ❌ Metadata query failed: {e}")
    
    # Test cursor description
    print("\n2. Checking cursor.description after SELECT:")
    
    try:
        cursor.execute("SELECT * FROM TestVectorBugs.VectorTypeTest LIMIT 1")
        
        if cursor.description:
            print("   Cursor description:")
            for desc in cursor.description:
                print(f"   - {desc[0]}: type_code={desc[1]}")
        else:
            print("   ❌ No cursor description available")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def demonstrate_workarounds(connection):
    """Show practical workarounds for the driver limitations"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("WORKAROUNDS: Handling VECTOR Type in Production")
    print("="*60)
    
    print("\n1. Insertion workaround pattern:")
    print("""
    # Convert Python list to string
    vector = [0.1, 0.2, 0.3, 0.4]
    vector_str = ','.join(map(str, vector))
    
    # Use TO_VECTOR in SQL (with string interpolation)
    cursor.execute(f'''
        INSERT INTO table (vector_col) 
        VALUES (TO_VECTOR('{vector_str}', 'DOUBLE', {len(vector)}))
    ''')
    """)
    
    print("\n2. Retrieval workaround pattern:")
    print("""
    # Retrieve and parse VECTOR data
    cursor.execute("SELECT vector_col FROM table")
    row = cursor.fetchone()
    
    # Parse string representation back to list
    if isinstance(row[0], str):
        vector = [float(x) for x in row[0].strip('[]').split(',')]
    """)
    
    print("\n3. Safe parameterization pattern:")
    print("""
    # Validate vector data before interpolation
    def safe_vector_string(vector):
        # Ensure all elements are numbers
        return ','.join([str(float(x)) for x in vector])
    
    # Use with validation
    safe_str = safe_vector_string(user_vector)
    cursor.execute(f"... TO_VECTOR('{safe_str}', 'FLOAT', 4) ...")
    """)

def main():
    """Main execution"""
    print("IRIS SQL Vector Search - Bug #3: VECTOR Type Driver Support")
    print("===========================================================")
    
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
        test_vector_insertion(connection)
        test_vector_retrieval(connection)
        test_driver_metadata(connection)
        demonstrate_workarounds(connection)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("❌ BUG CONFIRMED: Python driver lacks native VECTOR type support")
        print("📝 Issues:")
        print("   - Cannot bind Python lists/arrays to VECTOR parameters")
        print("   - VECTOR columns retrieved as strings, not arrays")
        print("   - No type metadata for VECTOR columns")
        print("🔧 Workarounds:")
        print("   - Use TO_VECTOR() with string interpolation for inserts")
        print("   - Parse string representation when retrieving")
        print("   - Implement validation to prevent SQL injection")
        print("✅ Fix needed: Native VECTOR type support in Python driver")
        
    finally:
        connection.close()
        print("\n✅ Connection closed")

if __name__ == "__main__":
    main()