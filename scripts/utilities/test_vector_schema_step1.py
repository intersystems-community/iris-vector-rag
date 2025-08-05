#!/usr/bin/env python3
"""
STEP 1: Test Vector Schema with Correct HNSW Syntax

This script methodically tests:
1. VECTOR column creation and data types
2. HNSW index creation with correct syntax
3. Vector search functionality
4. Performance comparison

The goal is to determine exactly what vector capabilities are available
before proceeding with data conversion.
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

def test_vector_schema():
    """
    Comprehensive test of VECTOR schema capabilities
    """
    print("=" * 80)
    print("STEP 1: VECTOR SCHEMA TESTING")
    print("=" * 80)
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "summary": {},
        "recommendations": []
    }
    
    try:
        # Get database connection
        print("\n1. Connecting to IRIS database...")
        conn = get_iris_connection()
        cursor = conn.cursor()
        print("✅ Database connection successful")
        
        # Test 1: Basic VECTOR column creation
        print("\n2. Testing VECTOR column creation...")
        test_results["tests"]["vector_column_creation"] = test_vector_column_creation(cursor)
        
        # Test 2: VECTOR data insertion and retrieval
        print("\n3. Testing VECTOR data operations...")
        test_results["tests"]["vector_data_operations"] = test_vector_data_operations(cursor)
        
        # Test 3: TO_VECTOR function availability
        print("\n4. Testing TO_VECTOR function...")
        test_results["tests"]["to_vector_function"] = test_to_vector_function(cursor)
        
        # Test 4: VECTOR_COSINE function availability
        print("\n5. Testing VECTOR_COSINE function...")
        test_results["tests"]["vector_cosine_function"] = test_vector_cosine_function(cursor)
        
        # Test 5: HNSW index creation with correct syntax
        print("\n6. Testing HNSW index creation...")
        test_results["tests"]["hnsw_index_creation"] = test_hnsw_index_creation(cursor)
        
        # Test 6: Vector search performance
        print("\n7. Testing vector search performance...")
        test_results["tests"]["vector_search_performance"] = test_vector_search_performance(cursor)
        
        # Test 7: Alternative approaches
        print("\n8. Testing alternative approaches...")
        test_results["tests"]["alternative_approaches"] = test_alternative_approaches(cursor)
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Critical error during testing: {e}")
        test_results["critical_error"] = str(e)
        test_results["traceback"] = traceback.format_exc()
    
    # Generate summary and recommendations
    generate_summary_and_recommendations(test_results)
    
    # Save results
    results_file = f"vector_schema_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📊 Test results saved to: {results_file}")
    
    return test_results

def test_vector_column_creation(cursor):
    """Test if VECTOR columns can be created"""
    test_result = {"success": False, "details": [], "error": None}
    
    try:
        # Drop test table if exists
        cursor.execute("DROP TABLE IF EXISTS VectorTest CASCADE")
        
        # Test different VECTOR column syntaxes
        vector_syntaxes = [
            "VECTOR(FLOAT, 384)",
            "VECTOR(FLOAT, 384)", 
            "VECTOR(384)",
            "VECTOR",
        ]
        
        for syntax in vector_syntaxes:
            try:
                print(f"   Testing VECTOR syntax: {syntax}")
                
                create_sql = f"""
                CREATE TABLE VectorTest (
                    id INTEGER PRIMARY KEY,
                    test_vector {syntax},
                    test_name VARCHAR(100)
                )
                """
                
                cursor.execute(create_sql)
                
                # Check actual column type
                cursor.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = 'VectorTest' AND COLUMN_NAME = 'test_vector'
                """)
                
                result = cursor.fetchone()
                if result:
                    actual_type = result[1]
                    max_length = result[2]
                    
                    test_result["details"].append({
                        "syntax": syntax,
                        "actual_type": actual_type,
                        "max_length": max_length,
                        "success": True
                    })
                    
                    print(f"   ✅ {syntax} -> {actual_type} (max_length: {max_length})")
                    
                    if actual_type.upper() == 'VECTOR':
                        test_result["success"] = True
                        print(f"   🎉 TRUE VECTOR TYPE DETECTED!")
                    else:
                        print(f"   ⚠️  Falls back to {actual_type}")
                
                cursor.execute("DROP TABLE VectorTest")
                
            except Exception as e:
                test_result["details"].append({
                    "syntax": syntax,
                    "error": str(e),
                    "success": False
                })
                print(f"   ❌ {syntax} failed: {e}")
        
    except Exception as e:
        test_result["error"] = str(e)
        print(f"❌ Vector column creation test failed: {e}")
    
    return test_result

def test_vector_data_operations(cursor):
    """Test VECTOR data insertion and retrieval"""
    test_result = {"success": False, "details": [], "error": None}
    
    try:
        # Create test table with best available VECTOR syntax
        cursor.execute("DROP TABLE IF EXISTS VectorDataTest CASCADE")
        
        # Try VECTOR(FLOAT, 384) first, fall back to VARCHAR if needed
        try:
            cursor.execute("""
                CREATE TABLE VectorDataTest (
                    id INTEGER PRIMARY KEY,
                    embedding VECTOR(FLOAT, 384),
                    description VARCHAR(100)
                )
            """)
            vector_type = "VECTOR(FLOAT, 384)"
        except:
            cursor.execute("""
                CREATE TABLE VectorDataTest (
                    id INTEGER PRIMARY KEY,
                    embedding VARCHAR(30000),
                    description VARCHAR(100)
                )
            """)
            vector_type = "VARCHAR(30000)"
        
        print(f"   Using column type: {vector_type}")
        
        # Test data insertion
        test_vectors = [
            ([0.1, 0.2, 0.3, 0.4], "Simple 4D vector"),
            ([0.5, -0.3, 0.8, -0.1, 0.2], "5D vector with negatives"),
            ([1.0] * 10, "10D vector of ones"),
        ]
        
        for i, (vector_data, description) in enumerate(test_vectors, 1):
            try:
                if vector_type.startswith("VECTOR"):
                    # Try native VECTOR insertion
                    vector_str = ','.join(map(str, vector_data))
                    cursor.execute("""
                        INSERT INTO VectorDataTest (id, embedding, description)
                        VALUES (?, TO_VECTOR(?, 'FLOAT'), ?)
                    """, (i, vector_str, description))
                else:
                    # Use VARCHAR storage
                    vector_str = ','.join(map(str, vector_data))
                    cursor.execute("""
                        INSERT INTO VectorDataTest (id, embedding, description)
                        VALUES (?, ?, ?)
                    """, (i, vector_str, description))
                
                test_result["details"].append({
                    "vector_data": vector_data,
                    "description": description,
                    "insertion_success": True
                })
                print(f"   ✅ Inserted {description}")
                
            except Exception as e:
                test_result["details"].append({
                    "vector_data": vector_data,
                    "description": description,
                    "insertion_success": False,
                    "error": str(e)
                })
                print(f"   ❌ Failed to insert {description}: {e}")
        
        # Test data retrieval
        cursor.execute("SELECT id, embedding, description FROM VectorDataTest ORDER BY id")
        results = cursor.fetchall()
        
        print(f"   Retrieved {len(results)} records:")
        for row in results:
            print(f"     ID {row[0]}: {row[2]} -> {str(row[1])[:50]}...")
        
        if len(results) > 0:
            test_result["success"] = True
        
        cursor.execute("DROP TABLE VectorDataTest")
        
    except Exception as e:
        test_result["error"] = str(e)
        print(f"❌ Vector data operations test failed: {e}")
    
    return test_result

def test_to_vector_function(cursor):
    """Test TO_VECTOR function availability"""
    test_result = {"success": False, "details": [], "error": None}
    
    test_cases = [
        ("'0.1,0.2,0.3,0.4'", "DOUBLE"),
        ("'0.1,0.2,0.3,0.4'", "FLOAT"),
        ("'1,2,3,4'", "INTEGER"),
    ]
    
    for vector_str, data_type in test_cases:
        try:
            sql = f"SELECT TO_VECTOR({vector_str}, '{data_type}') AS test_vector"
            cursor.execute(sql)
            result = cursor.fetchone()
            
            test_result["details"].append({
                "input": vector_str,
                "data_type": data_type,
                "success": True,
                "result": str(result[0]) if result else None
            })
            print(f"   ✅ TO_VECTOR({vector_str}, '{data_type}') works")
            test_result["success"] = True
            
        except Exception as e:
            test_result["details"].append({
                "input": vector_str,
                "data_type": data_type,
                "success": False,
                "error": str(e)
            })
            print(f"   ❌ TO_VECTOR({vector_str}, '{data_type}') failed: {e}")
    
    return test_result

def test_vector_cosine_function(cursor):
    """Test VECTOR_COSINE function availability"""
    test_result = {"success": False, "details": [], "error": None}
    
    try:
        # Test VECTOR_COSINE with different approaches
        test_approaches = [
            {
                "name": "Direct string vectors",
                "sql": "SELECT VECTOR_COSINE('0.1,0.2,0.3', '0.4,0.5,0.6') AS similarity"
            },
            {
                "name": "TO_VECTOR conversion",
                "sql": "SELECT VECTOR_COSINE(TO_VECTOR('0.1,0.2,0.3', 'DOUBLE'), TO_VECTOR('0.4,0.5,0.6', 'DOUBLE')) AS similarity"
            },
        ]
        
        for approach in test_approaches:
            try:
                cursor.execute(approach["sql"])
                result = cursor.fetchone()
                
                test_result["details"].append({
                    "approach": approach["name"],
                    "sql": approach["sql"],
                    "success": True,
                    "result": float(result[0]) if result else None
                })
                print(f"   ✅ {approach['name']}: {result[0]}")
                test_result["success"] = True
                
            except Exception as e:
                test_result["details"].append({
                    "approach": approach["name"],
                    "sql": approach["sql"],
                    "success": False,
                    "error": str(e)
                })
                print(f"   ❌ {approach['name']} failed: {e}")
    
    except Exception as e:
        test_result["error"] = str(e)
        print(f"❌ VECTOR_COSINE function test failed: {e}")
    
    return test_result

def test_hnsw_index_creation(cursor):
    """Test HNSW index creation with correct syntax"""
    test_result = {"success": False, "details": [], "error": None}
    
    try:
        # Create test table
        cursor.execute("DROP TABLE IF EXISTS HNSWTest CASCADE")
        
        # Try different table creation approaches
        table_approaches = [
            {
                "name": "VECTOR column",
                "sql": """
                CREATE TABLE HNSWTest (
                    id INTEGER PRIMARY KEY,
                    embedding VECTOR(FLOAT, 384),
                    title VARCHAR(100)
                )
                """
            },
            {
                "name": "VARCHAR column",
                "sql": """
                CREATE TABLE HNSWTest (
                    id INTEGER PRIMARY KEY,
                    embedding VARCHAR(30000),
                    title VARCHAR(100)
                )
                """
            }
        ]
        
        table_created = False
        table_type = None
        
        for approach in table_approaches:
            try:
                cursor.execute(approach["sql"])
                table_created = True
                table_type = approach["name"]
                print(f"   ✅ Table created with {approach['name']}")
                break
            except Exception as e:
                print(f"   ❌ {approach['name']} table creation failed: {e}")
        
        if not table_created:
            test_result["error"] = "Could not create test table"
            return test_result
        
        # Insert test data
        if table_type == "VECTOR column":
            try:
                cursor.execute("""
                    INSERT INTO HNSWTest (id, embedding, title)
                    VALUES (1, TO_VECTOR('0.1,0.2,0.3,0.4', 'DOUBLE'), 'Test Vector 1')
                """)
            except:
                cursor.execute("""
                    INSERT INTO HNSWTest (id, embedding, title)
                    VALUES (1, '0.1,0.2,0.3,0.4', 'Test Vector 1')
                """)
        else:
            cursor.execute("""
                INSERT INTO HNSWTest (id, embedding, title)
                VALUES (1, '0.1,0.2,0.3,0.4', 'Test Vector 1')
            """)
        
        # Test HNSW index creation with different syntaxes
        hnsw_syntaxes = [
            "AS HNSW(Distance='Cosine')",
            "AS HNSW(M=16, efConstruction=200, Distance='Cosine')",
            "AS HNSW(Distance='COSINE')",
            "AS HNSW(Distance=COSINE)",
            "AS HNSW",
        ]
        
        for syntax in hnsw_syntaxes:
            try:
                index_name = f"idx_test_hnsw_{len(test_result['details'])}"
                sql = f"CREATE INDEX {index_name} ON HNSWTest (embedding) {syntax}"
                
                print(f"   Testing HNSW syntax: {syntax}")
                cursor.execute(sql)
                
                test_result["details"].append({
                    "syntax": syntax,
                    "success": True,
                    "index_name": index_name
                })
                print(f"   ✅ HNSW index created successfully!")
                test_result["success"] = True
                
                # Try to drop the index
                cursor.execute(f"DROP INDEX {index_name}")
                
            except Exception as e:
                test_result["details"].append({
                    "syntax": syntax,
                    "success": False,
                    "error": str(e)
                })
                print(f"   ❌ HNSW syntax failed: {e}")
        
        cursor.execute("DROP TABLE HNSWTest")
        
    except Exception as e:
        test_result["error"] = str(e)
        print(f"❌ HNSW index creation test failed: {e}")
    
    return test_result

def test_vector_search_performance(cursor):
    """Test vector search performance"""
    test_result = {"success": False, "details": [], "error": None}
    
    try:
        # Create test table with sample data
        cursor.execute("DROP TABLE IF EXISTS VectorPerfTest CASCADE")
        
        # Use VARCHAR for compatibility
        cursor.execute("""
            CREATE TABLE VectorPerfTest (
                id INTEGER PRIMARY KEY,
                embedding VARCHAR(30000),
                title VARCHAR(100)
            )
        """)
        
        # Insert test vectors
        print("   Inserting test vectors...")
        test_vectors = []
        for i in range(100):
            # Generate random-ish vector
            vector = [0.1 * (i % 10), 0.2 * ((i + 1) % 10), 0.3 * ((i + 2) % 10), 0.4 * ((i + 3) % 10)]
            vector_str = ','.join(map(str, vector))
            test_vectors.append((i + 1, vector_str, f"Test Document {i + 1}"))
        
        cursor.executemany("""
            INSERT INTO VectorPerfTest (id, embedding, title)
            VALUES (?, ?, ?)
        """, test_vectors)
        
        print(f"   Inserted {len(test_vectors)} test vectors")
        
        # Test different search approaches
        query_vector = "0.5,0.5,0.5,0.5"
        
        search_approaches = [
            {
                "name": "Application-level search (retrieve all)",
                "sql": "SELECT id, embedding, title FROM VectorPerfTest"
            }
        ]
        
        # Try VECTOR_COSINE if available
        try:
            cursor.execute(f"SELECT VECTOR_COSINE('{query_vector}', '{query_vector}') AS test")
            search_approaches.append({
                "name": "VECTOR_COSINE search",
                "sql": f"""
                SELECT id, title, VECTOR_COSINE(embedding, '{query_vector}') AS similarity
                FROM VectorPerfTest
                ORDER BY similarity DESC
                LIMIT 10
                """
            })
        except:
            print("   VECTOR_COSINE not available, skipping native search test")
        
        for approach in search_approaches:
            try:
                start_time = time.time()
                cursor.execute(approach["sql"])
                results = cursor.fetchall()
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                test_result["details"].append({
                    "approach": approach["name"],
                    "execution_time": execution_time,
                    "result_count": len(results),
                    "success": True
                })
                
                print(f"   ✅ {approach['name']}: {execution_time:.4f}s, {len(results)} results")
                test_result["success"] = True
                
            except Exception as e:
                test_result["details"].append({
                    "approach": approach["name"],
                    "success": False,
                    "error": str(e)
                })
                print(f"   ❌ {approach['name']} failed: {e}")
        
        cursor.execute("DROP TABLE VectorPerfTest")
        
    except Exception as e:
        test_result["error"] = str(e)
        print(f"❌ Vector search performance test failed: {e}")
    
    return test_result

def test_alternative_approaches(cursor):
    """Test alternative vector storage and search approaches"""
    test_result = {"success": False, "details": [], "error": None}
    
    try:
        # Test computed columns approach
        print("   Testing computed columns approach...")
        
        cursor.execute("DROP TABLE IF EXISTS ComputedVectorTest CASCADE")
        
        try:
            cursor.execute("""
                CREATE TABLE ComputedVectorTest (
                    id INTEGER PRIMARY KEY,
                    embedding_str VARCHAR(30000),
                    embedding_vector VECTOR(FLOAT, 384) COMPUTECODE {
                        if ({embedding_str} '= "") {
                            set {embedding_vector} = TO_VECTOR({embedding_str}, 'FLOAT')
                        } else {
                            set {embedding_vector} = ""
                        }
                    } CALCULATED,
                    title VARCHAR(100)
                )
            """)
            
            # Insert test data
            cursor.execute("""
                INSERT INTO ComputedVectorTest (id, embedding_str, title)
                VALUES (1, '0.1,0.2,0.3,0.4', 'Computed Vector Test')
            """)
            
            # Query computed column
            cursor.execute("SELECT id, embedding_vector, title FROM ComputedVectorTest")
            result = cursor.fetchone()
            
            test_result["details"].append({
                "approach": "Computed columns",
                "success": True,
                "result": str(result[1]) if result else None
            })
            print("   ✅ Computed columns approach works")
            
        except Exception as e:
            test_result["details"].append({
                "approach": "Computed columns",
                "success": False,
                "error": str(e)
            })
            print(f"   ❌ Computed columns approach failed: {e}")
        
        # Test view-based approach
        print("   Testing view-based approach...")
        
        try:
            cursor.execute("DROP TABLE IF EXISTS ViewVectorTest CASCADE")
            cursor.execute("""
                CREATE TABLE ViewVectorTest (
                    id INTEGER PRIMARY KEY,
                    embedding_str VARCHAR(30000),
                    title VARCHAR(100)
                )
            """)
            
            cursor.execute("""
                CREATE VIEW ViewVectorTestVector AS
                SELECT 
                    id,
                    title,
                    embedding_str,
                    TO_VECTOR(embedding_str, 'FLOAT') AS embedding
                FROM ViewVectorTest
                WHERE embedding_str IS NOT NULL AND embedding_str <> ''
            """)
            
            cursor.execute("""
                INSERT INTO ViewVectorTest (id, embedding_str, title)
                VALUES (1, '0.1,0.2,0.3,0.4', 'View Vector Test')
            """)
            
            cursor.execute("SELECT id, embedding, title FROM ViewVectorTestVector")
            result = cursor.fetchone()
            
            test_result["details"].append({
                "approach": "View-based conversion",
                "success": True,
                "result": str(result[1]) if result else None
            })
            print("   ✅ View-based approach works")
            
        except Exception as e:
            test_result["details"].append({
                "approach": "View-based conversion",
                "success": False,
                "error": str(e)
            })
            print(f"   ❌ View-based approach failed: {e}")
        
        if len([d for d in test_result["details"] if d["success"]]) > 0:
            test_result["success"] = True
        
        # Cleanup
        try:
            cursor.execute("DROP VIEW IF EXISTS ViewVectorTestVector")
            cursor.execute("DROP TABLE IF EXISTS ViewVectorTest CASCADE")
            cursor.execute("DROP TABLE IF EXISTS ComputedVectorTest CASCADE")
        except:
            pass
        
    except Exception as e:
        test_result["error"] = str(e)
        print(f"❌ Alternative approaches test failed: {e}")
    
    return test_result

def generate_summary_and_recommendations(test_results):
    """Generate summary and recommendations based on test results"""
    
    print("\n" + "=" * 80)
    print("STEP 1 TEST SUMMARY")
    print("=" * 80)
    
    # Count successes
    successful_tests = 0
    total_tests = len(test_results.get("tests", {}))
    
    capabilities = {
        "vector_columns": False,
        "to_vector_function": False,
        "vector_cosine_function": False,
        "hnsw_indexes": False,
        "native_vector_search": False,
        "alternative_approaches": False
    }
    
    for test_name, test_result in test_results.get("tests", {}).items():
        if test_result.get("success", False):
            successful_tests += 1
            
            if test_name == "vector_column_creation":
                # Check if any true VECTOR types were detected
                for detail in test_result.get("details", []):
                    if detail.get("actual_type", "").upper() == "VECTOR":
                        capabilities["vector_columns"] = True
            
            elif test_name == "to_vector_function":
                capabilities["to_vector_function"] = True
            
            elif test_name == "vector_cosine_function":
                capabilities["vector_cosine_function"] = True
            
            elif test_name == "hnsw_index_creation":
                capabilities["hnsw_indexes"] = True
            
            elif test_name == "vector_search_performance":
                # Check if VECTOR_COSINE search worked
                for detail in test_result.get("details", []):
                    if "VECTOR_COSINE" in detail.get("approach", ""):
                        capabilities["native_vector_search"] = True
            
            elif test_name == "alternative_approaches":
                capabilities["alternative_approaches"] = True
    
    print(f"Tests passed: {successful_tests}/{total_tests}")
    print("\nCapability Assessment:")
    
    for capability, available in capabilities.items():
        status = "✅ AVAILABLE" if available else "❌ NOT AVAILABLE"
        print(f"  {capability.replace('_', ' ').title()}: {status}")
    
    # Generate recommendations
    recommendations = []
    
    if capabilities["vector_columns"] and capabilities["hnsw_indexes"]:
        recommendations.append("🎉 FULL VECTOR SUPPORT: Proceed with native VECTOR columns and HNSW indexes")
        approach = "native_vector"
    elif capabilities["to_vector_function"] and capabilities["vector_cosine_function"]:
        recommendations.append("⚠️  PARTIAL SUPPORT: Use VARCHAR storage with TO_VECTOR conversion")
        approach = "varchar_with_conversion"
    elif capabilities["alternative_approaches"]:
        recommendations.append("🔄 ALTERNATIVE APPROACH: Use computed columns or views")
        approach = "alternative_methods"
    else:
        recommendations.append("❌ LIMITED SUPPORT: Use application-level vector operations")
        approach = "application_level"
    
    # Specific recommendations for data conversion
    if approach == "native_vector":
        recommendations.extend([
            "✅ Convert existing VARCHAR embeddings to VECTOR columns",
            "✅ Create HNSW indexes with AS HNSW(Distance='Cosine') syntax",
            "✅ Use native VECTOR_COSINE for similarity search"
        ])
    elif approach == "varchar_with_conversion":
        recommendations.extend([
            "⚠️  Keep VARCHAR storage, use TO_VECTOR in queries",
            "⚠️  HNSW indexes may not be available",
            "✅ Use VECTOR_COSINE with TO_VECTOR conversion"
        ])
    elif approach == "alternative_methods":
        recommendations.extend([
            "🔄 Use computed columns for VECTOR conversion",
            "🔄 Create views with TO_VECTOR conversion",
            "⚠️  Test HNSW index creation on computed columns"
        ])
    else:
        recommendations.extend([
            "❌ Keep VARCHAR storage",
            "❌ No HNSW indexes available",
            "❌ Use application-level similarity computation (numpy, faiss)"
        ])
    
    test_results["summary"] = {
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "capabilities": capabilities,
        "recommended_approach": approach
    }
    test_results["recommendations"] = recommendations
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\n📋 Recommended approach for STEP 2: {approach}")

if __name__ == "__main__":
    test_results = test_vector_schema()
    
    # Determine if we can proceed to STEP 2
    approach = test_results.get("summary", {}).get("recommended_approach", "application_level")
    
    print("\n" + "=" * 80)
    print("STEP 1 COMPLETION STATUS")
    print("=" * 80)
    
    if approach in ["native_vector", "varchar_with_conversion", "alternative_methods"]:
        print("✅ STEP 1 COMPLETE - Ready to proceed to STEP 2 (Data Conversion)")
        print(f"   Recommended approach: {approach}")
    else:
        print("⚠️  STEP 1 COMPLETE - Limited vector support detected")
        print("   Consider using application-level vector operations")
    
    print("\nNext steps:")
    print("1. Review the test results file")
    print("2. Proceed to STEP 2 based on recommended approach")
    print("3. Implement data conversion strategy")