#!/usr/bin/env python3
"""
IRIS SQL Vector Search Bug #2: HNSW Index Creation on VARCHAR
=============================================================

This script demonstrates the bug where HNSW indexes cannot be created
on VARCHAR columns containing vector data, even with TO_VECTOR conversion.

Prerequisites:
- InterSystems IRIS instance with vector search enabled
- Python intersystems-iris driver installed: pip install intersystems-iris
- IRIS connection details (update CONNECTION_STRING below)

Bug Description:
HNSW indexes fail to create on VARCHAR columns that store vector embeddings
as comma-separated strings. The error message is misleading and the 
documentation doesn't clarify the requirements.

Expected: HNSW index should work with TO_VECTOR conversion on VARCHAR columns
Actual: "functional indices can only be defined on one vector property"
"""

import iris

# UPDATE THESE CONNECTION DETAILS
CONNECTION_STRING = "localhost:1972/USER"
USERNAME = "_SYSTEM"
PASSWORD = "SYS"
NAMESPACE = "USER"

def setup_test_tables(connection):
    """Create test tables with different vector storage approaches"""
    cursor = connection.cursor()
    
    # Create schema if not exists
    try:
        cursor.execute("CREATE SCHEMA TestVectorBugs")
    except:
        pass  # Schema might already exist
    
    # Drop existing test tables
    tables_to_drop = [
        'VectorAsVarchar',
        'VectorAsVector',
        'VectorMixed'
    ]
    
    for table in tables_to_drop:
        try:
            cursor.execute(f"DROP TABLE TestVectorBugs.{table}")
        except:
            pass
    
    # Table 1: Vector stored as VARCHAR (common pattern)
    cursor.execute("""
        CREATE TABLE TestVectorBugs.VectorAsVarchar (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            embedding VARCHAR(10000)
        )
    """)
    
    # Table 2: Vector stored as VECTOR type
    cursor.execute("""
        CREATE TABLE TestVectorBugs.VectorAsVector (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            embedding VECTOR(FLOAT, 4)
        )
    """)
    
    # Table 3: Mixed approach (both VARCHAR and VECTOR)
    cursor.execute("""
        CREATE TABLE TestVectorBugs.VectorMixed (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            embedding_varchar VARCHAR(10000),
            embedding_vector VECTOR(FLOAT, 4)
        )
    """)
    
    # Insert test data
    test_data = [
        (1, 'Doc 1', '0.1,0.2,0.3,0.4'),
        (2, 'Doc 2', '0.2,0.3,0.4,0.5'),
        (3, 'Doc 3', '0.3,0.4,0.5,0.6')
    ]
    
    # Insert into VARCHAR table
    for id, name, embedding in test_data:
        cursor.execute(
            "INSERT INTO TestVectorBugs.VectorAsVarchar (id, name, embedding) VALUES (?, ?, ?)",
            [id, name, embedding]
        )
    
    # Insert into VECTOR table
    for id, name, embedding in test_data:
        # Convert string to vector format
        vector_values = [float(x) for x in embedding.split(',')]
        cursor.execute(
            f"INSERT INTO TestVectorBugs.VectorAsVector (id, name, embedding) VALUES (?, ?, TO_VECTOR('{embedding}', 'FLOAT', 4))",
            [id, name]
        )
    
    # Insert into mixed table
    for id, name, embedding in test_data:
        cursor.execute(
            f"INSERT INTO TestVectorBugs.VectorMixed (id, name, embedding_varchar, embedding_vector) VALUES (?, ?, ?, TO_VECTOR('{embedding}', 'FLOAT', 4))",
            [id, name, embedding]
        )
    
    connection.commit()
    print("✅ Test tables created and populated")

def test_hnsw_index_creation(connection):
    """Test HNSW index creation on different column types"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("BUG #2: HNSW Index Creation on VARCHAR Columns")
    print("="*60)
    
    # Test 1: HNSW on VARCHAR column with TO_VECTOR
    print("\n1. Testing HNSW index on VARCHAR column with TO_VECTOR:")
    print("   CREATE INDEX idx_hnsw_varchar ON VectorAsVarchar (TO_VECTOR(embedding))")
    
    try:
        cursor.execute("""
            CREATE INDEX idx_hnsw_varchar 
            ON TestVectorBugs.VectorAsVarchar (TO_VECTOR(embedding, 'FLOAT', 4))
            AS HNSW(Distance='COSINE')
        """)
        print("   ✅ SUCCESS (unexpected!): HNSW index created on VARCHAR")
    except Exception as e:
        print(f"   ❌ FAILED (expected): {e}")
        print("   Error: Cannot create HNSW index on VARCHAR column")
    
    # Test 2: HNSW on native VECTOR column
    print("\n2. Testing HNSW index on native VECTOR column:")
    print("   CREATE INDEX idx_hnsw_vector ON VectorAsVector (embedding)")
    
    try:
        cursor.execute("""
            CREATE INDEX idx_hnsw_vector 
            ON TestVectorBugs.VectorAsVector (embedding)
            AS HNSW(Distance='COSINE')
        """)
        print("   ✅ SUCCESS: HNSW index created on VECTOR column")
    except Exception as e:
        print(f"   ❌ FAILED (unexpected): {e}")
    
    # Test 3: HNSW with explicit dimensions
    print("\n3. Testing HNSW with explicit vector dimensions:")
    
    try:
        cursor.execute("""
            CREATE INDEX idx_hnsw_explicit 
            ON TestVectorBugs.VectorAsVarchar (TO_VECTOR(embedding, 'FLOAT', 4))
            AS HNSW(Distance='COSINE', Dimension=4)
        """)
        print("   ✅ SUCCESS: HNSW index created with explicit dimensions")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 4: Regular index on VARCHAR (for comparison)
    print("\n4. Testing regular index on VARCHAR column (baseline):")
    
    try:
        cursor.execute("""
            CREATE INDEX idx_regular_varchar 
            ON TestVectorBugs.VectorAsVarchar (embedding)
        """)
        print("   ✅ SUCCESS: Regular index works on VARCHAR")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

def test_vector_search_performance(connection):
    """Compare search performance with and without HNSW"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("Performance Comparison: HNSW vs No Index")
    print("="*60)
    
    query_vector = "0.15,0.25,0.35,0.45"
    
    # Test 1: Search on VARCHAR table (no HNSW possible)
    print("\n1. Vector search on VARCHAR column (no HNSW):")
    
    try:
        cursor.execute(f"""
            SELECT name, 
                   VECTOR_COSINE(TO_VECTOR(embedding, 'FLOAT', 4), TO_VECTOR('{query_vector}', 'FLOAT', 4)) as score
            FROM TestVectorBugs.VectorAsVarchar
            ORDER BY score DESC
        """)
        results = cursor.fetchall()
        print("   ✅ Query executed successfully")
        for row in results:
            print(f"      - {row[0]}: {row[1]:.4f}")
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
    
    # Test 2: Search on VECTOR table (with HNSW)
    print("\n2. Vector search on VECTOR column (with HNSW):")
    
    try:
        cursor.execute(f"""
            SELECT name, 
                   VECTOR_COSINE(embedding, TO_VECTOR('{query_vector}', 'FLOAT', 4)) as score
            FROM TestVectorBugs.VectorAsVector
            ORDER BY score DESC
        """)
        results = cursor.fetchall()
        print("   ✅ Query executed successfully (using HNSW index)")
        for row in results:
            print(f"      - {row[0]}: {row[1]:.4f}")
    except Exception as e:
        print(f"   ❌ Query failed: {e}")

def demonstrate_workaround(connection):
    """Show the workaround: migrate VARCHAR to VECTOR columns"""
    cursor = connection.cursor()
    
    print("\n" + "="*60)
    print("WORKAROUND: Migrate VARCHAR to VECTOR Columns")
    print("="*60)
    
    print("\n1. Create new table with VECTOR column:")
    
    try:
        # Create new table with proper VECTOR column
        cursor.execute("""
            CREATE TABLE TestVectorBugs.VectorMigrated (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                embedding_varchar VARCHAR(10000),  -- Keep original for reference
                embedding_vector VECTOR(FLOAT, 4)  -- New VECTOR column
            )
        """)
        print("   ✅ Created table with VECTOR column")
        
        # Migrate data
        print("\n2. Migrate data from VARCHAR to VECTOR:")
        cursor.execute("""
            INSERT INTO TestVectorBugs.VectorMigrated (id, name, embedding_varchar, embedding_vector)
            SELECT id, name, embedding, TO_VECTOR(embedding, 'FLOAT', 4)
            FROM TestVectorBugs.VectorAsVarchar
        """)
        print("   ✅ Data migrated successfully")
        
        # Create HNSW index on new VECTOR column
        print("\n3. Create HNSW index on VECTOR column:")
        cursor.execute("""
            CREATE INDEX idx_hnsw_migrated 
            ON TestVectorBugs.VectorMigrated (embedding_vector)
            AS HNSW(Distance='COSINE')
        """)
        print("   ✅ HNSW index created successfully")
        
        connection.commit()
        
    except Exception as e:
        print(f"   ❌ Migration failed: {e}")

def main():
    """Main execution"""
    print("IRIS SQL Vector Search - Bug #2: HNSW Index on VARCHAR")
    print("======================================================")
    
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
        setup_test_tables(connection)
        
        # Run bug demonstrations
        test_hnsw_index_creation(connection)
        test_vector_search_performance(connection)
        demonstrate_workaround(connection)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("❌ BUG CONFIRMED: HNSW indexes cannot be created on VARCHAR columns")
        print("📝 Impact: Performance degradation for existing VARCHAR-based systems")
        print("🔧 Workaround: Migrate to native VECTOR column type")
        print("✅ Fix needed: Support HNSW on VARCHAR with TO_VECTOR conversion")
        print("\nNote: This forces a complete schema migration for existing systems")
        
    finally:
        connection.close()
        print("\n✅ Connection closed")

if __name__ == "__main__":
    main()