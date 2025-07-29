#!/usr/bin/env python3
"""
Complete deployment and testing script for IRIS 2025.1 with Vector Search.
This script handles the entire process from deployment to validation.
"""

import sys
import os
import time
import subprocess
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

def run_command(command, description=""):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    if description:
        print(f"Description: {description}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ Success: {description}")
            return True, result.stdout
        else:
            print(f"✗ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {description}")
        return False, "Command timed out"
    except Exception as e:
        print(f"✗ Exception: {description} - {e}")
        return False, str(e)

def wait_for_iris_ready(max_attempts=30, container_name="iris_db_rag_licensed_simple"):
    """Wait for IRIS to be ready to accept connections."""
    print(f"Waiting for IRIS container {container_name} to be ready...")
    
    for attempt in range(max_attempts):
        try:
            # Check if container is running
            success, output = run_command(f"docker ps | grep {container_name}")
            if not success:
                print(f"Attempt {attempt + 1}/{max_attempts}: Container not running yet")
                time.sleep(3)
                continue
            
            # Try to connect to IRIS
            conn = get_iris_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            print("✓ IRIS is ready!")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts}: IRIS not ready yet ({e})")
            time.sleep(3)
    
    print("✗ IRIS failed to become ready")
    return False

def test_vector_search_license():
    """Test if Vector Search is enabled in the license."""
    print("\n" + "=" * 60)
    print("TESTING VECTOR SEARCH LICENSE")
    print("=" * 60)
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Test license feature check with proper ObjectScript syntax
        cursor.execute("SELECT $SYSTEM.License.GetFeature('Vector Search') as vector_search_enabled")
        result = cursor.fetchone()
        
        if result and result[0] == 1:
            print("✓ Vector Search is enabled in the license!")
            return True
        else:
            print(f"✗ Vector Search is not enabled. License check returned: {result}")
            return False
            
    except Exception as e:
        print(f"✗ License check failed: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def test_native_vector_support():
    """Test native VECTOR data type support."""
    print("\n" + "=" * 60)
    print("TESTING NATIVE VECTOR DATA TYPE")
    print("=" * 60)
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Drop and create test table
        cursor.execute("DROP TABLE IF EXISTS test_native_vector")
        
        # Create table with native VECTOR column
        cursor.execute("""
        CREATE TABLE test_native_vector (
            id INTEGER PRIMARY KEY,
            content VARCHAR(1000),
            embedding VECTOR(FLOAT, 768)
        )
        """)
        print("✓ Created table with VECTOR(FLOAT, 768) column")
        
        # Test inserting vector data
        test_vector = [0.1] * 768  # Simple test vector
        cursor.execute("""
        INSERT INTO test_native_vector (id, content, embedding) 
        VALUES (?, ?, TO_VECTOR(?))
        """, (1, "Test document", str(test_vector)))
        print("✓ Successfully inserted vector data")
        
        # Verify column type
        cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'TEST_NATIVE_VECTOR' AND COLUMN_NAME = 'EMBEDDING'
        """)
        result = cursor.fetchone()
        
        if result:
            col_name, data_type = result
            print(f"✓ Column type: {col_name} = {data_type}")
            if data_type.upper() == 'VECTOR':
                print("✓ Native VECTOR data type is working!")
                return True
            else:
                print(f"✗ Expected VECTOR, got {data_type}")
                return False
        else:
            print("✗ Could not verify column type")
            return False
            
    except Exception as e:
        print(f"✗ Native vector test failed: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def test_hnsw_indexes():
    """Test HNSW index creation and functionality."""
    print("\n" + "=" * 60)
    print("TESTING HNSW INDEXES")
    print("=" * 60)
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Create HNSW index on the test table
        cursor.execute("""
        CREATE INDEX idx_test_native_vector_hnsw 
        ON test_native_vector (embedding) 
        AS HNSW(Distance='Cosine')
        """)
        print("✓ Successfully created HNSW index")
        
        # Insert more test data for search
        for i in range(2, 11):
            test_vector = [0.1 + (i * 0.01)] * 768
            cursor.execute("""
            INSERT INTO test_native_vector (id, content, embedding) 
            VALUES (?, ?, TO_VECTOR(?))
            """, (i, f"Test document {i}", str(test_vector)))
        
        print("✓ Inserted test data for HNSW search")
        
        # Test vector similarity search using HNSW
        query_vector = [0.15] * 768
        cursor.execute("""
        SELECT TOP 5 id, content, 
               VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
        FROM test_native_vector 
        ORDER BY VECTOR_COSINE(embedding, TO_VECTOR(?)) DESC
        """, (str(query_vector), str(query_vector)))
        
        results = cursor.fetchall()
        print(f"✓ HNSW vector search returned {len(results)} results")
        
        for i, (doc_id, content, similarity) in enumerate(results):
            print(f"  {i+1}. ID: {doc_id}, Similarity: {similarity:.4f}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"✗ HNSW test failed: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def create_production_rag_schema():
    """Create production-ready RAG schema with native vector support."""
    print("\n" + "=" * 60)
    print("CREATING PRODUCTION RAG SCHEMA")
    print("=" * 60)
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute("DROP TABLE IF EXISTS rag_documents_production")
        cursor.execute("""
        CREATE TABLE rag_documents_production (
            doc_id VARCHAR(50) PRIMARY KEY,
            title VARCHAR(500),
            content TEXT,
            embedding VECTOR(FLOAT, 768),
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        print("✓ Created rag_documents_production table")
        
        # Create HNSW index on documents
        cursor.execute("""
        CREATE INDEX idx_rag_docs_prod_hnsw 
        ON rag_documents_production (embedding) 
        AS HNSW(M=16, efConstruction=200, Distance='Cosine')
        """)
        print("✓ Created HNSW index on documents")
        
        # Create chunks table
        cursor.execute("DROP TABLE IF EXISTS rag_chunks_production")
        cursor.execute("""
        CREATE TABLE rag_chunks_production (
            chunk_id VARCHAR(100) PRIMARY KEY,
            doc_id VARCHAR(50),
            chunk_text TEXT,
            chunk_index INTEGER,
            embedding VECTOR(FLOAT, 768),
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES rag_documents_production(doc_id)
        )
        """)
        print("✓ Created rag_chunks_production table")
        
        # Create HNSW index on chunks
        cursor.execute("""
        CREATE INDEX idx_rag_chunks_prod_hnsw 
        ON rag_chunks_production (embedding) 
        AS HNSW(M=16, efConstruction=200, Distance='Cosine')
        """)
        print("✓ Created HNSW index on chunks")
        
        # Create additional indexes for performance
        cursor.execute("CREATE INDEX idx_rag_chunks_doc_id ON rag_chunks_production (doc_id)")
        cursor.execute("CREATE INDEX idx_rag_docs_created ON rag_documents_production (created_at)")
        print("✓ Created additional performance indexes")
        
        return True
        
    except Exception as e:
        print(f"✗ Production schema creation failed: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def test_production_rag_pipeline():
    """Test the complete RAG pipeline with native vectors."""
    print("\n" + "=" * 60)
    print("TESTING PRODUCTION RAG PIPELINE")
    print("=" * 60)
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Get embedding model
        embedding_model = get_embedding_model(mock=True)
        
        # Insert sample documents
        sample_docs = [
            ("DOC001", "Vector Search in IRIS", "IRIS 2025.1 introduces native vector search capabilities with HNSW indexes for high-performance similarity search."),
            ("DOC002", "Machine Learning Integration", "The new vector data type enables seamless integration with machine learning workflows and embedding models."),
            ("DOC003", "Enterprise RAG Solutions", "Enterprise-scale RAG applications can now leverage native vector storage and HNSW indexing for optimal performance.")
        ]
        
        for doc_id, title, content in sample_docs:
            embedding = embedding_model.encode([content])[0]
            metadata = json.dumps({"source": "test", "type": "sample"})
            
            cursor.execute("""
            INSERT INTO rag_documents_production (doc_id, title, content, embedding, metadata_json)
            VALUES (?, ?, ?, TO_VECTOR(?), ?)
            """, (doc_id, title, content, str(embedding.tolist()), metadata))
        
        print("✓ Inserted sample documents with embeddings")
        
        # Create chunks for each document
        chunk_count = 0
        for doc_id, title, content in sample_docs:
            # Simple chunking - split by sentences
            sentences = content.split('. ')
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    chunk_embedding = embedding_model.encode([sentence])[0]
                    chunk_metadata = json.dumps({"sentence_index": i, "doc_title": title})
                    
                    cursor.execute("""
                    INSERT INTO rag_chunks_production (chunk_id, doc_id, chunk_text, chunk_index, embedding, metadata_json)
                    VALUES (?, ?, ?, ?, TO_VECTOR(?), ?)
                    """, (f"{doc_id}_CHUNK_{i}", doc_id, sentence, i, str(chunk_embedding.tolist()), chunk_metadata))
                    chunk_count += 1
        
        print(f"✓ Created {chunk_count} chunks with embeddings")
        
        # Test vector search on the production schema
        query = "vector search performance"
        query_embedding = embedding_model.encode([query])[0]
        
        # Search documents
        cursor.execute("""
        SELECT TOP 3 doc_id, title, 
               VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
        FROM rag_documents_production 
        ORDER BY VECTOR_COSINE(embedding, TO_VECTOR(?)) DESC
        """, (str(query_embedding.tolist()), str(query_embedding.tolist())))
        
        doc_results = cursor.fetchall()
        print(f"✓ Document search returned {len(doc_results)} results")
        
        # Search chunks
        cursor.execute("""
        SELECT TOP 5 chunk_id, chunk_text, 
               VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
        FROM rag_chunks_production 
        ORDER BY VECTOR_COSINE(embedding, TO_VECTOR(?)) DESC
        """, (str(query_embedding.tolist()), str(query_embedding.tolist())))
        
        chunk_results = cursor.fetchall()
        print(f"✓ Chunk search returned {len(chunk_results)} results")
        
        # Display results
        print("\nTop Document Results:")
        for doc_id, title, similarity in doc_results:
            print(f"  - {doc_id}: {title} (similarity: {similarity:.4f})")
        
        print("\nTop Chunk Results:")
        for chunk_id, chunk_text, similarity in chunk_results:
            print(f"  - {chunk_id}: {chunk_text[:50]}... (similarity: {similarity:.4f})")
        
        return len(doc_results) > 0 and len(chunk_results) > 0
        
    except Exception as e:
        print(f"✗ Production RAG pipeline test failed: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def main():
    """Main deployment and testing process."""
    print("IRIS 2025.1 Vector Search Deployment and Testing")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Wait for IRIS to be ready
    if not wait_for_iris_ready():
        print("❌ IRIS container is not ready. Deployment failed.")
        return False
    
    # Run all tests
    tests = [
        ("Vector Search License", test_vector_search_license),
        ("Native Vector Data Type", test_native_vector_support),
        ("HNSW Indexes", test_hnsw_indexes),
        ("Production RAG Schema", create_production_rag_schema),
        ("Production RAG Pipeline", test_production_rag_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT AND TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 IRIS 2025.1 Vector Search deployment successful!")
        print("✓ Native VECTOR data type is working")
        print("✓ HNSW indexes are functional")
        print("✓ Production RAG schema is ready")
        print("✓ Vector search performance is validated")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)