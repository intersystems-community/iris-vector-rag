#!/usr/bin/env python3
"""
Quick JDBC test with correct IRIS credentials
"""

import os
import sys

# Check if JDBC driver exists
jdbc_paths = [
    "./intersystems-jdbc-3.8.4.jar",
    "../intersystems-jdbc-3.8.4.jar",
    "./jdbc_exploration/intersystems-jdbc-3.8.4.jar",
    os.path.expanduser("~/intersystems-jdbc-3.8.4.jar"),
    "/opt/iris/jdbc/intersystems-jdbc-3.8.4.jar"
]

jdbc_driver_path = None
for path in jdbc_paths:
    if os.path.exists(path):
        jdbc_driver_path = path
        print(f"✅ Found JDBC driver at: {path}")
        break

if not jdbc_driver_path:
    print("❌ IRIS JDBC driver not found!")
    sys.exit(1)

# Try to import required libraries
try:
    import jaydebeapi
    import jpype
    print("✅ JayDeBeAPI and JPype installed")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# Test connection
print("\n🔍 Testing JDBC Connection to IRIS...")
print("=" * 50)

# Connection parameters - using correct defaults from iris_connector.py
host = os.getenv('IRIS_HOST', 'localhost')
port = os.getenv('IRIS_PORT', '1972')
namespace = os.getenv('IRIS_NAMESPACE', 'USER')  # Changed from RAG to USER
username = os.getenv('IRIS_USERNAME', 'SuperUser')  # Changed from demo
password = os.getenv('IRIS_PASSWORD', 'SYS')  # Changed from demo

jdbc_url = f"jdbc:IRIS://{host}:{port}/{namespace}"
jdbc_driver_class = "com.intersystems.jdbc.IRISDriver"

print(f"📊 Connection URL: {jdbc_url}")
print(f"👤 Username: {username}")
print(f"🔑 Namespace: {namespace}")

try:
    # Start JVM
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), 
                      f"-Djava.class.path={jdbc_driver_path}")
    
    # Connect
    conn = jaydebeapi.connect(
        jdbc_driver_class,
        jdbc_url,
        [username, password],
        jdbc_driver_path
    )
    
    print("\n✅ Successfully connected to IRIS via JDBC!")
    
    # Test basic query
    cursor = conn.cursor()
    
    # First check what namespace we're in
    cursor.execute("SELECT $NAMESPACE")
    current_ns = cursor.fetchone()[0]
    print(f"📊 Current namespace: {current_ns}")
    
    # Check if RAG schema exists
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.SCHEMATA 
        WHERE SCHEMA_NAME = 'RAG'
    """)
    schema_exists = cursor.fetchone()[0] > 0
    
    if schema_exists:
        print("✅ RAG schema exists")
        
        # Check if SourceDocuments table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            count = cursor.fetchone()[0]
            print(f"📊 Found {count:,} documents in RAG.SourceDocuments table")
        else:
            print("⚠️  RAG.SourceDocuments table not found")
    else:
        print("⚠️  RAG schema not found - you may need to switch namespace or create schema")
    
    # Test vector function
    print("\n🔍 Testing vector functions...")
    cursor.execute("""
        SELECT VECTOR_COSINE(
            TO_VECTOR('1.0,2.0,3.0', 'DOUBLE', 3),
            TO_VECTOR('4.0,5.0,6.0', 'DOUBLE', 3)
        ) as similarity
    """)
    similarity = cursor.fetchone()[0]
    print(f"✅ VECTOR_COSINE works! Result: {similarity}")
    
    # Test parameter binding
    print("\n🔍 Testing parameter binding with vectors...")
    test_vector = "0.1,0.2,0.3"
    cursor.execute("""
        SELECT VECTOR_COSINE(
            TO_VECTOR(?, 'DOUBLE', 3),
            TO_VECTOR('1.0,2.0,3.0', 'DOUBLE', 3)
        ) as similarity
    """, [test_vector])
    similarity = cursor.fetchone()[0]
    print(f"✅ Parameter binding works! Result: {similarity}")
    
    # If RAG schema exists, test real vector search
    if schema_exists and table_exists:
        print("\n🔍 Testing real vector search with parameter binding...")
        # Create a test embedding (384 dimensions)
        test_embedding = ','.join([str(i * 0.001) for i in range(384)])
        
        try:
            cursor.execute("""
                SELECT TOP 3 
                    doc_id, 
                    title,
                    VECTOR_COSINE(
                        TO_VECTOR(embedding, 'DOUBLE', 384),
                        TO_VECTOR(?, 'DOUBLE', 384)
                    ) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """, [test_embedding])
            
            results = cursor.fetchall()
            print(f"✅ Vector search with parameter binding works! Found {len(results)} results")
            for doc_id, title, score in results:
                print(f"   - {doc_id}: {title[:50]}... (score: {score:.4f})")
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
    
    cursor.close()
    conn.close()
    
    print("\n🎉 All tests passed! JDBC is ready to use.")
    print("\nNext steps:")
    print("1. Update iris_jdbc_connector.py to use correct namespace")
    print("2. Test BasicRAG JDBC: python basic_rag/pipeline_jdbc.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")
    
    if "No suitable driver found" in str(e):
        print("\n💡 This usually means the JDBC URL format is incorrect")
    elif "Connection refused" in str(e):
        print("\n💡 Check that IRIS is running and accessible")
    elif "Access Denied" in str(e):
        print("\n💡 Check credentials - default should be SuperUser/SYS")
    
    import traceback
    traceback.print_exc()

finally:
    # Shutdown JVM
    if jpype.isJVMStarted():
        jpype.shutdownJVM()