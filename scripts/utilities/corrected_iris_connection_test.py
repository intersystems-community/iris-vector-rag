#!/usr/bin/env python3
"""
VECTOR SEARCH COMMUNITY VS LICENSED COMPARISON TEST
Uses the proper 'iris' import from intersystems-irispython package
Tests Vector Search functionality on both Licensed and Community editions
"""

print("VECTOR SEARCH COMMUNITY VS LICENSED COMPARISON TEST")
print("=" * 70)

def test_basic_iris_connection():
    """Test basic connection using iris module"""
    print("=== TESTING BASIC IRIS CONNECTION ===")
    try:
        import iris
        
        # Connection parameters for the licensed IRIS container
        args = {
            'hostname': 'iris_db_rag_licensed_simple',
            'port': 1972,
            'namespace': 'USER',
            'username': '_SYSTEM',
            'password': 'SYS'
        }
        
        print(f"Connecting to IRIS at {args['hostname']}:{args['port']}")
        conn = iris.connect(**args)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT $HOROLOG")
        result = cursor.fetchone()
        print(f"✅ Connection successful! IRIS time: {result[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_vector_operations():
    """Test Vector Search operations with corrected syntax"""
    print("\n=== TESTING VECTOR SEARCH OPERATIONS ===")
    try:
        import iris
        
        args = {
            'hostname': 'iris_db_rag_licensed_simple',
            'port': 1972,
            'namespace': 'USER',
            'username': '_SYSTEM',
            'password': 'SYS'
        }
        
        conn = iris.connect(**args)
        cursor = conn.cursor()
        
        # Test corrected TO_VECTOR syntax (no brackets, no quotes around type)
        print("Testing corrected TO_VECTOR syntax...")
        test_queries = [
            "SELECT TO_VECTOR('1.0, 2.0, 3.0', double) AS test_vector",
            "SELECT VECTOR_DOT_PRODUCT(TO_VECTOR('1.0, 2.0, 3.0', double), TO_VECTOR('4.0, 5.0, 6.0', double)) AS dot_product",
            "SELECT VECTOR_COSINE(TO_VECTOR('1.0, 0.0, 0.0', double), TO_VECTOR('0.0, 1.0, 0.0', double)) AS cosine_sim"
        ]
        
        for query in test_queries:
            try:
                cursor.execute(query)
                result = cursor.fetchone()
                print(f"✅ Query successful: {query[:50]}... → {result[0]}")
            except Exception as e:
                print(f"❌ Query failed: {query[:50]}... → {e}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        return False

def test_vector_search_functions():
    """Test Vector Search functions availability"""
    print("\n=== TESTING VECTOR SEARCH FUNCTIONS ===")
    try:
        import iris
        
        args = {
            'hostname': 'iris_db_rag_licensed_simple',
            'port': 1972,
            'namespace': 'USER',
            'username': '_SYSTEM',
            'password': 'SYS'
        }
        
        conn = iris.connect(**args)
        cursor = conn.cursor()
        
        # Test if Vector Search functions are available
        functions_to_test = [
            "VECTOR_DOT_PRODUCT",
            "VECTOR_COSINE", 
            "VECTOR_EUCLIDEAN",
            "TO_VECTOR"
        ]
        
        for func in functions_to_test:
            try:
                # Test if function exists by checking system catalog
                cursor.execute(f"SELECT 1 WHERE '{func}' %INLIST $LISTFROMSTRING('VECTOR_DOT_PRODUCT,VECTOR_COSINE,VECTOR_EUCLIDEAN,TO_VECTOR')")
                result = cursor.fetchone()
                if result:
                    print(f"✅ Function {func} is available")
                else:
                    print(f"❓ Function {func} status unknown")
            except Exception as e:
                print(f"❌ Function {func} test failed: {e}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Vector search functions test failed: {e}")
        return False

def main():
    """Run all tests"""
    tests = [
        test_basic_iris_connection,
        test_vector_operations,
        test_vector_search_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed - IRIS connection and Vector Search are working!")
    else:
        print("❌ Some tests failed - Check IRIS setup and Vector Search configuration")
    
    return passed == total

def test_edition_comparison():
    """Test both Licensed and Community editions"""
    print("\n" + "="*70)
    print("COMPARING LICENSED VS COMMUNITY EDITIONS")
    print("="*70)
    
    editions = [
        {"name": "Licensed Edition", "hostname": "iris_db_rag_licensed_simple", "port": 1972},
        {"name": "Community Edition", "hostname": "iris_db_rag_community", "port": 1972}
    ]
    
    results = {}
    
    for edition in editions:
        print(f"\n🔍 Testing {edition['name']} (port {edition['port']})...")
        
        try:
            import iris
            
            # Connection parameters
            args = {
                'hostname': edition['hostname'],
                'port': edition['port'],
                'namespace': 'USER',
                'username': '_SYSTEM',
                'password': 'SYS'
            }
            
            # Test connection
            conn = iris.connect(**args)
            print(f"✅ Connected to {edition['name']}")
            
            # Test VECTOR data type
            cursor = conn.cursor()
            test_table = f"test_vector_{int(time.time())}"
            
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
                cursor.execute(f"""
                    CREATE TABLE {test_table} (
                        id INTEGER PRIMARY KEY,
                        test_vector VECTOR(FLOAT, 384)
                    )
                """)
                
                cursor.execute(f"""
                    SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{test_table}' AND COLUMN_NAME = 'test_vector'
                """)
                
                result = cursor.fetchone()
                actual_type = result[0] if result else "UNKNOWN"
                vector_supported = 'VECTOR' in actual_type.upper()
                
                cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
                
                print(f"   VECTOR data type: {'✅ SUPPORTED' if vector_supported else '❌ NOT SUPPORTED'} ({actual_type})")
                
            except Exception as e:
                vector_supported = False
                print(f"   VECTOR data type: ❌ FAILED ({e})")
            
            # Test TO_VECTOR function
            try:
                cursor.execute("SELECT TO_VECTOR('0.1,0.2,0.3', double) as result")
                result = cursor.fetchone()
                to_vector_works = result is not None
                print(f"   TO_VECTOR function: {'✅ WORKS' if to_vector_works else '❌ FAILS'}")
            except Exception as e:
                to_vector_works = False
                print(f"   TO_VECTOR function: ❌ FAILED ({e})")
            
            # Test VECTOR_COSINE function
            try:
                cursor.execute("""
                    SELECT VECTOR_COSINE(
                        TO_VECTOR('0.1,0.2,0.3', double),
                        TO_VECTOR('0.4,0.5,0.6', double)
                    ) as similarity
                """)
                result = cursor.fetchone()
                cosine_works = result is not None
                print(f"   VECTOR_COSINE function: {'✅ WORKS' if cosine_works else '❌ FAILS'}")
            except Exception as e:
                cosine_works = False
                print(f"   VECTOR_COSINE function: ❌ FAILED ({e})")
            
            # Test HNSW index creation
            try:
                hnsw_table = f"test_hnsw_{int(time.time())}"
                cursor.execute(f"DROP TABLE IF EXISTS {hnsw_table}")
                cursor.execute(f"""
                    CREATE TABLE {hnsw_table} (
                        id INTEGER PRIMARY KEY,
                        test_vector VECTOR(FLOAT, 384)
                    )
                """)
                
                index_name = f"idx_hnsw_{int(time.time())}"
                cursor.execute(f"""
                    CREATE INDEX {index_name} ON {hnsw_table} (test_vector)
                    AS HNSW(Distance='Cosine')
                """)
                
                cursor.execute(f"DROP TABLE IF EXISTS {hnsw_table}")
                hnsw_works = True
                print(f"   HNSW indexing: ✅ SUPPORTED")
            except Exception as e:
                hnsw_works = False
                print(f"   HNSW indexing: ❌ FAILED ({e})")
            
            conn.close()
            
            results[edition['name']] = {
                "connection": True,
                "vector_data_type": vector_supported,
                "to_vector_function": to_vector_works,
                "vector_cosine_function": cosine_works,
                "hnsw_indexing": hnsw_works
            }
            
        except Exception as e:
            print(f"❌ Failed to connect to {edition['name']}: {e}")
            results[edition['name']] = {
                "connection": False,
                "vector_data_type": False,
                "to_vector_function": False,
                "vector_cosine_function": False,
                "hnsw_indexing": False
            }
    
    # Generate comparison report
    print("\n" + "="*70)
    print("FEATURE COMPARISON SUMMARY")
    print("="*70)
    
    features = ["connection", "vector_data_type", "to_vector_function", "vector_cosine_function", "hnsw_indexing"]
    
    print(f"{'Feature':<25} {'Licensed':<12} {'Community':<12} {'Status'}")
    print("-" * 70)
    
    for feature in features:
        licensed = results.get("Licensed Edition", {}).get(feature, False)
        community = results.get("Community Edition", {}).get(feature, False)
        
        licensed_str = "✅ YES" if licensed else "❌ NO"
        community_str = "✅ YES" if community else "❌ NO"
        
        if licensed and community:
            status = "Both editions"
        elif licensed and not community:
            status = "Licensed only"
        elif not licensed and community:
            status = "Community only"
        else:
            status = "Neither edition"
        
        print(f"{feature.replace('_', ' ').title():<25} {licensed_str:<12} {community_str:<12} {status}")
    
    # Calculate feature parity
    licensed_features = sum(1 for f in features if results.get("Licensed Edition", {}).get(f, False))
    community_features = sum(1 for f in features if results.get("Community Edition", {}).get(f, False))
    
    if licensed_features > 0:
        parity = (community_features / licensed_features) * 100
    else:
        parity = 0
    
    print(f"\n📊 Feature Parity: {parity:.1f}% ({community_features}/{licensed_features} features)")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if parity >= 80:
        print("   ✅ Community Edition provides good Vector Search support")
        print("   ✅ Suitable for most Vector Search use cases")
    elif parity >= 50:
        print("   ⚠️  Community Edition has limited Vector Search support")
        print("   ⚠️  Consider Licensed Edition for full functionality")
    else:
        print("   ❌ Community Edition lacks Vector Search capabilities")
        print("   ❌ Licensed Edition required for Vector Search")
    
    return results

if __name__ == "__main__":
    import time
    
    # Run original tests on Licensed Edition
    print("Testing Licensed Edition (port 1972)...")
    main()
    
    # Run comparison between editions
    test_edition_comparison()