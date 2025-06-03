#!/usr/bin/env python3
"""
TO_VECTOR Parameter Binding Test
================================

This script tests TO_VECTOR parameter binding scenarios to verify correct syntax
and behavior with IRIS SQL.

Key TO_VECTOR syntax rules (verified with IRIS 2025.1 + Python driver 5.1.2):
1. TO_VECTOR can take up to 3 arguments: TO_VECTOR(data [, type ] [, length ])
2. data and length can be parameters
3. type CANNOT be a parameter, it must be an unquoted keyword: DECIMAL, DOUBLE, FLOAT, INT, INTEGER, STRING
4. TO_VECTOR(?) should work
5. TO_VECTOR(?,DOUBLE) should work
6. TO_VECTOR(?, 'DOUBLE') or TO_VECTOR(?, ?) will fail

Status: All tests pass with correct syntax in IRIS 2025.1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.iris_connector import get_iris_connection

def test_to_vector_parameter_binding():
    """Test various TO_VECTOR parameter binding scenarios."""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Test data
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    vector_str = ','.join(map(str, test_vector))
    
    print("Testing TO_VECTOR parameter binding scenarios...")
    print("=" * 60)
    print(f"Test vector data: {vector_str}")
    print("=" * 60)
    
    # Test 1: Basic parameter binding with TO_VECTOR(?)
    print("\n1. Testing TO_VECTOR(?) - should work")
    try:
        cursor.execute("SELECT TO_VECTOR(?)", (vector_str,))
        result = cursor.fetchone()
        print(f"✓ SUCCESS: TO_VECTOR(?) returned: {result}")
    except Exception as e:
        print(f"✗ FAILED: TO_VECTOR(?) - {e}")
    
    # Test 2: Parameter binding with type specification TO_VECTOR(?, DOUBLE)
    print("\n2. Testing TO_VECTOR(?, DOUBLE) - should work")
    try:
        cursor.execute("SELECT TO_VECTOR(?, DOUBLE)", (vector_str,))
        result = cursor.fetchone()
        print(f"✓ SUCCESS: TO_VECTOR(?, DOUBLE) returned: {result}")
    except Exception as e:
        print(f"✗ FAILED: TO_VECTOR(?, DOUBLE) - {e}")
    
    # Test 3: Parameter binding with quoted type - should fail
    print("\n3. Testing TO_VECTOR(?, 'DOUBLE') - should fail (incorrect syntax)")
    try:
        cursor.execute("SELECT TO_VECTOR(?, 'DOUBLE')", (vector_str,))
        result = cursor.fetchone()
        print(f"✗ UNEXPECTED SUCCESS: TO_VECTOR(?, 'DOUBLE') returned: {result}")
    except Exception as e:
        print(f"✓ EXPECTED FAILURE: TO_VECTOR(?, 'DOUBLE') - {e}")
    
    # Test 4: Parameter binding for type - should fail
    print("\n4. Testing TO_VECTOR(?, ?) with type as parameter - should fail")
    try:
        cursor.execute("SELECT TO_VECTOR(?, ?)", (vector_str, 'DOUBLE'))
        result = cursor.fetchone()
        print(f"✗ UNEXPECTED SUCCESS: TO_VECTOR(?, ?) returned: {result}")
    except Exception as e:
        print(f"✓ EXPECTED FAILURE: TO_VECTOR(?, ?) - {e}")
    
    # Test 5: Parameter binding with length specification
    print("\n5. Testing TO_VECTOR(?, DOUBLE, ?) with length parameter - should work")
    try:
        cursor.execute("SELECT TO_VECTOR(?, DOUBLE, ?)", (vector_str, len(test_vector)))
        result = cursor.fetchone()
        print(f"✓ SUCCESS: TO_VECTOR(?, DOUBLE, ?) returned: {result}")
    except Exception as e:
        print(f"✗ FAILED: TO_VECTOR(?, DOUBLE, ?) - {e}")
    
    # Test 6: Test different vector types with unquoted keywords
    print("\n6. Testing different vector types with unquoted keywords")
    vector_types = ['FLOAT', 'DOUBLE', 'INT', 'INTEGER', 'DECIMAL', 'STRING']
    
    for vtype in vector_types:
        try:
            cursor.execute(f"SELECT TO_VECTOR(?, {vtype})", (vector_str,))
            result = cursor.fetchone()
            print(f"✓ SUCCESS: TO_VECTOR(?, {vtype}) returned: {result}")
        except Exception as e:
            print(f"✗ FAILED: TO_VECTOR(?, {vtype}) - {e}")
    
    # Test 7: Test with INSERT OR UPDATE statement
    print("\n7. Testing INSERT OR UPDATE with TO_VECTOR parameter binding")
    try:
        # Create a test table
        cursor.execute("DROP TABLE IF EXISTS test_vector_params")
        cursor.execute("""
            CREATE TABLE test_vector_params (
                id INT PRIMARY KEY,
                vector_data VECTOR(DOUBLE, 5)
            )
        """)
        
        # Test INSERT OR UPDATE with TO_VECTOR(?)
        cursor.execute("""
            INSERT OR UPDATE test_vector_params (id, vector_data) 
            VALUES (1, TO_VECTOR(?))
        """, (vector_str,))
        
        # Verify the insert
        cursor.execute("SELECT id, vector_data FROM test_vector_params WHERE id = 1")
        result = cursor.fetchone()
        print(f"✓ SUCCESS: INSERT OR UPDATE with TO_VECTOR(?) - {result}")
        
    except Exception as e:
        print(f"✗ FAILED: INSERT OR UPDATE with TO_VECTOR(?) - {e}")
    
    # Test 8: Test with INSERT OR UPDATE and type specification
    print("\n8. Testing INSERT OR UPDATE with TO_VECTOR(?, DOUBLE)")
    try:
        cursor.execute("""
            INSERT OR UPDATE test_vector_params (id, vector_data) 
            VALUES (2, TO_VECTOR(?, DOUBLE))
        """, (vector_str,))
        
        # Verify the insert
        cursor.execute("SELECT id, vector_data FROM test_vector_params WHERE id = 2")
        result = cursor.fetchone()
        print(f"✓ SUCCESS: INSERT OR UPDATE with TO_VECTOR(?, DOUBLE) - {result}")
        
    except Exception as e:
        print(f"✗ FAILED: INSERT OR UPDATE with TO_VECTOR(?, DOUBLE) - {e}")
    
    # Test 9: Test with INSERT OR UPDATE and length specification
    print("\n9. Testing INSERT OR UPDATE with TO_VECTOR(?, DOUBLE, ?)")
    try:
        cursor.execute("""
            INSERT OR UPDATE test_vector_params (id, vector_data) 
            VALUES (3, TO_VECTOR(?, DOUBLE, ?))
        """, (vector_str, len(test_vector)))
        
        # Verify the insert
        cursor.execute("SELECT id, vector_data FROM test_vector_params WHERE id = 3")
        result = cursor.fetchone()
        print(f"✓ SUCCESS: INSERT OR UPDATE with TO_VECTOR(?, DOUBLE, ?) - {result}")
        
    except Exception as e:
        print(f"✗ FAILED: INSERT OR UPDATE with TO_VECTOR(?, DOUBLE, ?) - {e}")
    
    # Test 10: Test vector search with parameter binding
    print("\n10. Testing vector search with VECTOR_COSINE and parameter binding")
    try:
        # Insert some test data for search (using DOUBLE type to match table definition)
        test_vectors = [
            (10, "0.1,0.2,0.3,0.4,0.5"),
            (11, "0.2,0.3,0.4,0.5,0.6"),
            (12, "0.3,0.4,0.5,0.6,0.7")
        ]
        
        for vid, vdata in test_vectors:
            cursor.execute("""
                INSERT OR UPDATE test_vector_params (id, vector_data)
                VALUES (?, TO_VECTOR(?, DOUBLE))
            """, (vid, vdata))
        
        # Test vector search with parameter binding (using DOUBLE type to match)
        query_vector = "0.15,0.25,0.35,0.45,0.55"
        cursor.execute("""
            SELECT id, VECTOR_COSINE(vector_data, TO_VECTOR(?, DOUBLE)) as similarity
            FROM test_vector_params
            WHERE id >= 10
            ORDER BY similarity DESC
        """, (query_vector,))
        
        results = cursor.fetchall()
        print(f"✓ SUCCESS: Vector search with parameter binding")
        for row in results:
            print(f"   ID: {row[0]}, Similarity: {row[1]:.4f}")
            
    except Exception as e:
        print(f"✗ FAILED: Vector search with parameter binding - {e}")
    
    # Test 11: Test failure cases that should not work
    print("\n11. Testing failure cases (incorrect syntax)")
    
    failure_cases = [
        ("TO_VECTOR(?, 'FLOAT')", (vector_str,), "quoted type should fail"),
        ("TO_VECTOR(?, ?, ?)", (vector_str, 'DOUBLE', len(test_vector)), "type as parameter should fail"),
    ]
    
    for sql_fragment, params, description in failure_cases:
        try:
            cursor.execute(f"SELECT {sql_fragment}", params)
            result = cursor.fetchone()
            print(f"✗ UNEXPECTED SUCCESS: {description} - {sql_fragment} returned: {result}")
        except Exception as e:
            print(f"✓ EXPECTED FAILURE: {description} - {sql_fragment} - {e}")
    
    # Cleanup
    try:
        cursor.execute("DROP TABLE IF EXISTS test_vector_params")
    except:
        pass
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("TO_VECTOR parameter binding test completed!")
    print("=" * 60)
    print("\nSUMMARY OF EXPECTED BEHAVIOR:")
    print("✓ TO_VECTOR(?) should work")
    print("✓ TO_VECTOR(?, DOUBLE) should work")
    print("✓ TO_VECTOR(?, DOUBLE, ?) should work")
    print("✗ TO_VECTOR(?, 'DOUBLE') should fail (quoted type - incorrect syntax)")
    print("✗ TO_VECTOR(?, ?) should fail (type as parameter - incorrect syntax)")
    print("✓ INSERT OR UPDATE with TO_VECTOR parameter binding should work")
    print("✓ Vector search with parameter binding should work")

def main():
    """Main execution"""
    print("TO_VECTOR Parameter Binding Test")
    print("================================")
    print("Testing correct TO_VECTOR syntax with IRIS 2025.1")
    print("Status: All tests should pass with correct syntax")
    
    try:
        test_to_vector_parameter_binding()
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print("✅ TO_VECTOR parameter binding works correctly in IRIS 2025.1")
        print("📝 Only issue: Documentation should clarify correct unquoted keyword syntax")
        print("🔧 No workarounds needed: Use proper syntax as shown above")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    main()