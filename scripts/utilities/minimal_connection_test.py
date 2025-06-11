#!/usr/bin/env python3
"""
Minimal Python test to debug IRIS connection and Vector Search syntax.
Focus on licensed container: iris_db_rag_licensed_simple
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_basic_connection():
    """Test basic connection to licensed IRIS container"""
    print("=== TESTING BASIC CONNECTION TO LICENSED IRIS ===")
    
    try:
        import intersystems_iris
        print("✅ intersystems_iris module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import intersystems_iris: {e}")
        return False
    
    # Connection parameters for licensed container
    conn_params = {
        "hostname": "localhost",
        "port": 1972,
        "namespace": "IRIS",  # Using IRIS namespace, not IRISHEALTH
        "username": "SuperUser",
        "password": "SYS"
    }
    
    print(f"Attempting connection to: {conn_params['hostname']}:{conn_params['port']}/{conn_params['namespace']}")
    
    try:
        conn = intersystems_iris.connect(**conn_params)
        print("✅ Connection established successfully")
        
        # Test basic SQL
        cursor = conn.cursor()
        cursor.execute("SELECT $ZVERSION")
        version = cursor.fetchone()[0]
        print(f"✅ IRIS Version: {version}")
        
        # Test namespace
        cursor.execute("SELECT $NAMESPACE")
        namespace = cursor.fetchone()[0]
        print(f"✅ Current namespace: {namespace}")
        
        cursor.close()
        conn.close()
        print("✅ Connection closed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_vector_operations():
    """Test Vector Search operations with correct syntax"""
    print("\n=== TESTING VECTOR OPERATIONS ===")
    
    try:
        import intersystems_iris
        
        # Connection parameters
        conn_params = {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "IRIS",
            "username": "SuperUser", 
            "password": "SYS"
        }
        
        conn = intersystems_iris.connect(**conn_params)
        cursor = conn.cursor()
        
        # Create test table with VECTOR column
        test_table = "VectorTest"
        
        print(f"Creating test table: {test_table}")
        cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
        cursor.execute(f"CREATE TABLE {test_table} (id INT, test_vector VECTOR(3, DOUBLE))")
        print("✅ Table created successfully")
        
        # Test TO_VECTOR with correct syntax (no brackets, no quotes around data type)
        print("Testing TO_VECTOR syntax...")
        test_vectors = [
            "0.1, 0.2, 0.3",
            "1.0, 2.0, 3.0", 
            "-0.5, 0.0, 0.5"
        ]
        
        for i, vector_str in enumerate(test_vectors, 1):
            try:
                sql = f"INSERT INTO {test_table} (id, test_vector) VALUES ({i}, TO_VECTOR('{vector_str}', double))"
                print(f"Executing: {sql}")
                cursor.execute(sql)
                print(f"✅ Vector {i} inserted successfully")
            except Exception as e:
                print(f"❌ Vector {i} failed: {e}")
        
        # Test retrieval
        print("Testing vector retrieval...")
        cursor.execute(f"SELECT id, test_vector FROM {test_table} ORDER BY id")
        results = cursor.fetchall()
        
        for row in results:
            print(f"✅ Retrieved: ID={row[0]}, Vector={str(row[1])[:50]}...")
        
        # Test VECTOR_DOT_PRODUCT function
        print("Testing VECTOR_DOT_PRODUCT...")
        try:
            cursor.execute(f"""
                SELECT id, VECTOR_DOT_PRODUCT(test_vector, TO_VECTOR('1.0, 1.0, 1.0', double)) as similarity
                FROM {test_table} 
                ORDER BY similarity DESC
            """)
            results = cursor.fetchall()
            print("✅ VECTOR_DOT_PRODUCT results:")
            for row in results:
                print(f"   ID={row[0]}, Similarity={row[1]}")
        except Exception as e:
            print(f"❌ VECTOR_DOT_PRODUCT failed: {e}")
        
        # Cleanup
        cursor.execute(f"DROP TABLE {test_table}")
        print("✅ Test table dropped")
        
        cursor.close()
        conn.close()
        print("✅ Vector operations test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        return False

def test_with_iris_connector():
    """Test using the project's iris_connector module"""
    print("\n=== TESTING WITH PROJECT IRIS_CONNECTOR ===")
    
    try:
        from common.iris_connector import get_iris_connection
        
        # Set environment variables for licensed container
        os.environ["IRIS_HOST"] = "localhost"
        os.environ["IRIS_PORT"] = "1972"
        os.environ["IRIS_NAMESPACE"] = "IRIS"
        os.environ["IRIS_USERNAME"] = "SuperUser"
        os.environ["IRIS_PASSWORD"] = "SYS"
        
        print("Getting connection via iris_connector...")
        conn = get_iris_connection()
        print("✅ Connection obtained via iris_connector")
        
        cursor = conn.cursor()
        cursor.execute("SELECT $ZVERSION")
        version = cursor.fetchone()[0]
        print(f"✅ IRIS Version via iris_connector: {version}")
        
        cursor.close()
        conn.close()
        print("✅ iris_connector test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ iris_connector test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MINIMAL IRIS CONNECTION AND VECTOR SEARCH TEST")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Basic connection
    if test_basic_connection():
        success_count += 1
    
    # Test 2: Vector operations
    if test_vector_operations():
        success_count += 1
    
    # Test 3: Project iris_connector
    if test_with_iris_connector():
        success_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED - Connection and Vector Search working!")
    else:
        print("❌ Some tests failed - Check connection parameters and IRIS setup")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)