#!/usr/bin/env python3
"""
Test the CORRECT TO_VECTOR syntax based on official IRIS documentation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection

def test_correct_vector_functions():
    """Test vector functions with correct syntax from official docs"""
    
    try:
        connection = get_iris_connection()
        print("✅ Connected to IRIS successfully")
        
        # Test 1: TO_VECTOR with correct syntax (comma-separated string)
        print("\n🧪 Test 1: TO_VECTOR with comma-separated string")
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT TO_VECTOR('0.1,0.2,0.3,0.4,0.5') AS vector_result")
                result = cursor.fetchone()
                if result:
                    print(f"   ✅ SUCCESS: {result[0]}")
                else:
                    print(f"   ❌ No result returned")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        # Test 2: TO_VECTOR with type specification
        print("\n🧪 Test 2: TO_VECTOR with type specification")
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT TO_VECTOR('0.1,0.2,0.3,0.4,0.5', 'DOUBLE') AS vector_result")
                result = cursor.fetchone()
                if result:
                    print(f"   ✅ SUCCESS: {result[0]}")
                else:
                    print(f"   ❌ No result returned")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        # Test 3: TO_VECTOR with length specification
        print("\n🧪 Test 3: TO_VECTOR with length specification")
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT TO_VECTOR('0.1,0.2,0.3,0.4,0.5', 'DOUBLE', 5) AS vector_result")
                result = cursor.fetchone()
                if result:
                    print(f"   ✅ SUCCESS: {result[0]}")
                else:
                    print(f"   ❌ No result returned")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        # Test 4: TO_VECTOR with square brackets (optional format)
        print("\n🧪 Test 4: TO_VECTOR with square brackets")
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT TO_VECTOR('[0.1,0.2,0.3,0.4,0.5]') AS vector_result")
                result = cursor.fetchone()
                if result:
                    print(f"   ✅ SUCCESS: {result[0]}")
                else:
                    print(f"   ❌ No result returned")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        # Test 5: VECTOR_COSINE function
        print("\n🧪 Test 5: VECTOR_COSINE function")
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT VECTOR_COSINE(
                        TO_VECTOR('0.1,0.2,0.3,0.4,0.5'),
                        TO_VECTOR('0.2,0.3,0.4,0.5,0.6')
                    ) AS cosine_similarity
                """)
                result = cursor.fetchone()
                if result:
                    similarity = float(result[0])
                    print(f"   ✅ SUCCESS: Cosine similarity = {similarity:.4f}")
                else:
                    print(f"   ❌ No result returned")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        # Test 6: VECTOR data type with TO_VECTOR
        print("\n🧪 Test 6: VECTOR data type with TO_VECTOR")
        try:
            with connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS test_vector_correct")
                cursor.execute("""
                    CREATE TABLE test_vector_correct (
                        id INTEGER PRIMARY KEY,
                        embedding VECTOR(DOUBLE, 5)
                    )
                """)
                
                # Insert using TO_VECTOR
                cursor.execute("""
                    INSERT INTO test_vector_correct (id, embedding) 
                    VALUES (1, TO_VECTOR('0.1,0.2,0.3,0.4,0.5'))
                """)
                
                # Query back
                cursor.execute("SELECT id, embedding FROM test_vector_correct WHERE id = 1")
                result = cursor.fetchone()
                if result:
                    print(f"   ✅ SUCCESS: ID={result[0]}, EMBEDDING={result[1]}")
                else:
                    print(f"   ❌ No result returned")
                    
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        finally:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("DROP TABLE IF EXISTS test_vector_correct")
            except:
                pass
        
        # Test 7: HNSW index creation
        print("\n🧪 Test 7: HNSW index creation")
        try:
            with connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS test_hnsw_correct")
                cursor.execute("""
                    CREATE TABLE test_hnsw_correct (
                        id INTEGER PRIMARY KEY,
                        embedding VECTOR(DOUBLE, 5)
                    )
                """)
                
                # Try to create HNSW index
                cursor.execute("""
                    CREATE INDEX idx_test_hnsw_correct
                    ON test_hnsw_correct (embedding)
                    AS HNSW(M=16, efConstruction=200, Distance='COSINE')
                """)
                print(f"   ✅ SUCCESS: HNSW index created")
                    
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        finally:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("DROP TABLE IF EXISTS test_hnsw_correct")
            except:
                pass
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_correct_vector_functions()