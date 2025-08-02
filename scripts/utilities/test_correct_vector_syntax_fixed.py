#!/usr/bin/env python3
"""
Test correct TO_VECTOR syntax for IRIS 2025.1 Vector Search.
Based on working syntax: TO_VECTOR('0.1, 0.2, 0.3', double)
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_corrected_vector_syntax():
    """Test TO_VECTOR with corrected syntax - no brackets, no quotes around data type"""
    print('=== TESTING CORRECTED TO_VECTOR SYNTAX ===')
    
    try:
        import iris
        
        # Connection parameters for licensed container
        conn_params = {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "IRIS",
            "username": "SuperUser",
            "password": "SYS"
        }
        
        conn = iris.connect(**conn_params)
        cursor = conn.cursor()
        
        # Test different VECTOR column definitions
        vector_column_types = [
            'VECTOR(3, DOUBLE)',
            'VECTOR(768, DOUBLE)', 
            'VECTOR(3, FLOAT)',
            'VECTOR(768, FLOAT)'
        ]
        
        working_combinations = []
        
        for col_type in vector_column_types:
            print(f"\n--- Testing column type: {col_type} ---")
            
            # Create test table
            table_name = f"VectorTest_{col_type.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '_')}"
            
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                cursor.execute(f"CREATE TABLE {table_name} (id INT, test_vector {col_type})")
                print(f"✅ Table created with {col_type}")
                
                # Test TO_VECTOR syntaxes (corrected - no brackets, no quotes around data type)
                test_cases = [
                    ("3D vector with double", "1.0, 2.0, 3.0", "double"),
                    ("3D vector with float", "0.1, 0.2, 0.3", "float"),
                    ("Negative values", "-1.0, 0.0, 1.0", "double"),
                    ("Scientific notation", "1e-3, 2e-2, 3e-1", "double")
                ]
                
                for desc, vector_str, data_type in test_cases:
                    try:
                        sql = f"INSERT INTO {table_name} (id, test_vector) VALUES (1, TO_VECTOR('{vector_str}', {data_type}))"
                        print(f"Testing: {desc}")
                        print(f"SQL: {sql}")
                        
                        cursor.execute(sql)
                        print(f"✅ {desc} - SUCCESS")
                        
                        # Verify retrieval
                        cursor.execute(f"SELECT test_vector FROM {table_name} WHERE id = 1")
                        result = cursor.fetchone()[0]
                        print(f"   Retrieved: {str(result)[:50]}...")
                        
                        # Clear for next test
                        cursor.execute(f"DELETE FROM {table_name}")
                        
                        working_combinations.append({
                            'column_type': col_type,
                            'description': desc,
                            'vector_string': vector_str,
                            'data_type': data_type,
                            'sql': sql
                        })
                        
                    except Exception as e:
                        print(f"❌ {desc} - FAILED: {e}")
                
                # Test vector operations
                print(f"\n--- Testing vector operations with {col_type} ---")
                try:
                    # Insert test vectors
                    cursor.execute(f"INSERT INTO {table_name} (id, test_vector) VALUES (1, TO_VECTOR('1.0, 0.0, 0.0', double))")
                    cursor.execute(f"INSERT INTO {table_name} (id, test_vector) VALUES (2, TO_VECTOR('0.0, 1.0, 0.0', double))")
                    cursor.execute(f"INSERT INTO {table_name} (id, test_vector) VALUES (3, TO_VECTOR('0.0, 0.0, 1.0', double))")
                    
                    # Test VECTOR_DOT_PRODUCT
                    cursor.execute(f"""
                        SELECT id, VECTOR_DOT_PRODUCT(test_vector, TO_VECTOR('1.0, 1.0, 1.0', double)) as similarity
                        FROM {table_name} 
                        ORDER BY similarity DESC
                    """)
                    results = cursor.fetchall()
                    print("✅ VECTOR_DOT_PRODUCT results:")
                    for row in results:
                        print(f"   ID={row[0]}, Similarity={row[1]}")
                        
                except Exception as e:
                    print(f"❌ Vector operations failed: {e}")
                
                # Cleanup
                cursor.execute(f"DROP TABLE {table_name}")
                
            except Exception as e:
                print(f"❌ Column type {col_type} failed: {e}")
        
        cursor.close()
        conn.close()
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Working combinations found: {len(working_combinations)}")
        
        if working_combinations:
            print("\n✅ SUCCESSFUL TO_VECTOR SYNTAXES:")
            for combo in working_combinations:
                print(f"   Column: {combo['column_type']}")
                print(f"   SQL: {combo['sql']}")
                print()
        
        return len(working_combinations) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_vector_search_functions():
    """Test various vector search functions with correct syntax"""
    print('\n=== TESTING VECTOR SEARCH FUNCTIONS ===')
    
    try:
        import iris
        
        conn_params = {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "IRIS",
            "username": "SuperUser",
            "password": "SYS"
        }
        
        conn = iris.connect(**conn_params)
        cursor = conn.cursor()
        
        # Create test table
        table_name = "VectorSearchTest"
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} (id INT, doc_vector VECTOR(3, DOUBLE), name VARCHAR(100))")
        
        # Insert test data
        test_vectors = [
            (1, "1.0, 0.0, 0.0", "Unit X"),
            (2, "0.0, 1.0, 0.0", "Unit Y"), 
            (3, "0.0, 0.0, 1.0", "Unit Z"),
            (4, "0.707, 0.707, 0.0", "Diagonal XY"),
            (5, "0.577, 0.577, 0.577", "Diagonal XYZ")
        ]
        
        for vec_id, vector_str, name in test_vectors:
            cursor.execute(f"""
                INSERT INTO {table_name} (id, doc_vector, name) 
                VALUES ({vec_id}, TO_VECTOR('{vector_str}', double), '{name}')
            """)
        
        print(f"✅ Inserted {len(test_vectors)} test vectors")
        
        # Test different vector functions
        vector_functions = [
            ("VECTOR_DOT_PRODUCT", "VECTOR_DOT_PRODUCT(doc_vector, TO_VECTOR('1.0, 1.0, 1.0', double))"),
            ("VECTOR_COSINE", "VECTOR_COSINE(doc_vector, TO_VECTOR('1.0, 1.0, 1.0', double))"),
            ("VECTOR_EUCLIDEAN", "VECTOR_EUCLIDEAN(doc_vector, TO_VECTOR('1.0, 1.0, 1.0', double))")
        ]
        
        for func_name, func_sql in vector_functions:
            try:
                print(f"\n--- Testing {func_name} ---")
                cursor.execute(f"""
                    SELECT name, {func_sql} as score
                    FROM {table_name} 
                    ORDER BY score DESC
                """)
                results = cursor.fetchall()
                print(f"✅ {func_name} results:")
                for row in results:
                    print(f"   {row[0]}: {row[1]:.4f}")
                    
            except Exception as e:
                print(f"❌ {func_name} failed: {e}")
        
        # Test VECTOR_TOP_K if available
        try:
            print(f"\n--- Testing VECTOR_TOP_K ---")
            cursor.execute(f"""
                SELECT TOP 3 name, VECTOR_COSINE(doc_vector, TO_VECTOR('1.0, 1.0, 1.0', double)) as similarity
                FROM {table_name} 
                ORDER BY similarity DESC
            """)
            results = cursor.fetchall()
            print("✅ TOP 3 most similar vectors:")
            for row in results:
                print(f"   {row[0]}: {row[1]:.4f}")
                
        except Exception as e:
            print(f"❌ VECTOR_TOP_K test failed: {e}")
        
        # Cleanup
        cursor.execute(f"DROP TABLE {table_name}")
        cursor.close()
        conn.close()
        
        print("✅ Vector search functions test completed")
        return True
        
    except Exception as e:
        print(f"❌ Vector search functions test failed: {e}")
        return False

def main():
    """Run all corrected syntax tests"""
    print("CORRECTED TO_VECTOR SYNTAX TEST FOR IRIS 2025.1")
    print("=" * 60)
    print("Key corrections:")
    print("- NO brackets around vector values: '1.0, 2.0, 3.0' NOT '[1.0, 2.0, 3.0]'")
    print("- NO quotes around data type: double NOT 'double'")
    print("- Correct format: TO_VECTOR('1.0, 2.0, 3.0', double)")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Corrected TO_VECTOR syntax
    if test_corrected_vector_syntax():
        success_count += 1
    
    # Test 2: Vector search functions
    if test_vector_search_functions():
        success_count += 1
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED - Corrected TO_VECTOR syntax working!")
        print("\nRecommended syntax for production:")
        print("  TO_VECTOR('x1, x2, x3, ...', double)")
        print("  TO_VECTOR('x1, x2, x3, ...', float)")
    else:
        print("❌ Some tests failed - Check IRIS setup and Vector Search configuration")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)