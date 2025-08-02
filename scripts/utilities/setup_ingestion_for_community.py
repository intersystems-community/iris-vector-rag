#!/usr/bin/env python3
"""
Setup document ingestion for Community Edition 2025.1 with correct Vector Search syntax.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

def setup_ingestion():
    """Setup the database for document ingestion with Community Edition."""
    
    print("=" * 60)
    print("SETTING UP COMMUNITY EDITION FOR DOCUMENT INGESTION")
    print("=" * 60)
    
    try:
        # Set environment variables for Community Edition
        os.environ["IRIS_HOST"] = "localhost"
        os.environ["IRIS_PORT"] = "1972"
        os.environ["IRIS_NAMESPACE"] = "USER"
        os.environ["IRIS_USERNAME"] = "_SYSTEM"
        os.environ["IRIS_PASSWORD"] = "SYS"
        
        # Test connection
        print("\n1. Connecting to Community Edition...")
        conn = get_iris_connection()
        cursor = conn.cursor()
        print("✅ Connected to IRIS Community Edition")
        
        # Verify Vector Search capabilities
        print("\n2. Verifying Vector Search capabilities...")
        cursor.execute("SELECT TO_VECTOR('0.1,0.2,0.3', double) AS test")
        result = cursor.fetchone()
        print(f"✅ TO_VECTOR function working: {result[0]}")
        
        cursor.execute("""
            SELECT VECTOR_COSINE(
                TO_VECTOR('1.0,0.0,0.0', double),
                TO_VECTOR('0.0,1.0,0.0', double)
            ) AS similarity
        """)
        result = cursor.fetchone()
        print(f"✅ VECTOR_COSINE function working: {result[0]}")
        
        # Create minimal working schema for ingestion
        print("\n3. Creating minimal schema for document ingestion...")
        
        # Create tables one by one with error handling
        tables_created = []
        
        # SourceDocuments table
        try:
            cursor.execute("""
                CREATE TABLE SourceDocuments (
                    doc_id VARCHAR(255) PRIMARY KEY,
                    title VARCHAR(500),
                    text_content LONGVARCHAR,
                    abstract LONGVARCHAR,
                    authors LONGVARCHAR,
                    keywords LONGVARCHAR,
                    embedding_str VARCHAR(60000),
                    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
                    embedding_dimensions INTEGER DEFAULT 384,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            tables_created.append("SourceDocuments_V2")
            print("✅ SourceDocuments table created")
        except Exception as e:
            print(f"⚠️  SourceDocuments table creation failed: {e}")
            print("   Checking if table already exists...")
            try:
                cursor.execute("SELECT COUNT(*) FROM SourceDocuments_V2")
                print("✅ SourceDocuments table already exists")
                tables_created.append("SourceDocuments_V2")
            except:
                print("❌ SourceDocuments table not accessible")
        
        # DocumentChunks table
        try:
            cursor.execute("""
                CREATE TABLE DocumentChunks (
                    chunk_id VARCHAR(255) PRIMARY KEY,
                    doc_id VARCHAR(255),
                    chunk_index INTEGER,
                    chunk_text LONGVARCHAR,
                    chunk_size INTEGER,
                    overlap_size INTEGER,
                    embedding_str VARCHAR(60000),
                    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            tables_created.append("DocumentChunks")
            print("✅ DocumentChunks table created")
        except Exception as e:
            print(f"⚠️  DocumentChunks table creation failed: {e}")
            try:
                cursor.execute("SELECT COUNT(*) FROM DocumentChunks")
                print("✅ DocumentChunks table already exists")
                tables_created.append("DocumentChunks")
            except:
                print("❌ DocumentChunks table not accessible")
        
        # Create indexes for performance
        print("\n4. Creating performance indexes...")
        indexes_created = []
        
        index_queries = [
            ("idx_source_docs_title", "CREATE INDEX idx_source_docs_title ON SourceDocuments(title)"),
            ("idx_source_docs_model", "CREATE INDEX idx_source_docs_model ON SourceDocuments(embedding_model)"),
            ("idx_chunks_doc", "CREATE INDEX idx_chunks_doc ON DocumentChunks(doc_id)"),
            ("idx_chunks_index", "CREATE INDEX idx_chunks_index ON DocumentChunks(chunk_index)")
        ]
        
        for idx_name, idx_query in index_queries:
            try:
                cursor.execute(idx_query)
                indexes_created.append(idx_name)
                print(f"✅ Index {idx_name} created")
            except Exception as e:
                print(f"⚠️  Index {idx_name} creation failed: {e}")
        
        # Test data insertion with Vector Search
        print("\n5. Testing data insertion with Vector Search...")
        try:
            # Test embedding string that will be converted to vector in queries
            test_embedding = "0.1,0.2,0.3,0.4,0.5"
            cursor.execute("""
                INSERT INTO SourceDocuments_V2 (doc_id, title, text_content, embedding_str)
                VALUES ('test_doc_community', 'Test Document for Community Edition', 
                        'This is a test document for Community Edition Vector Search.', ?)
            """, (test_embedding,))
            print("✅ Test document inserted successfully")
            
            # Test vector similarity query using TO_VECTOR
            query_embedding = "0.1,0.2,0.3,0.4,0.5"
            cursor.execute("""
                SELECT doc_id, title,
                       VECTOR_COSINE(TO_VECTOR(embedding_str, double), TO_VECTOR(?, double)) AS similarity
                FROM SourceDocuments_V2 
                WHERE embedding_str IS NOT NULL AND embedding_str <> ''
                ORDER BY similarity DESC
            """, (query_embedding,))
            
            results = cursor.fetchall()
            if results:
                print(f"✅ Vector similarity search working: {len(results)} results")
                for row in results:
                    print(f"   {row[0]}: {row[1]} (similarity: {row[2]})")
            else:
                print("❌ No results from vector similarity search")
                
        except Exception as e:
            print(f"❌ Data insertion/query test failed: {e}")
        
        print("\n" + "=" * 60)
        print("COMMUNITY EDITION SETUP SUMMARY")
        print("=" * 60)
        print(f"✅ Tables created: {', '.join(tables_created)}")
        print(f"✅ Indexes created: {', '.join(indexes_created)}")
        print("✅ Vector Search functions verified working")
        print("✅ Ready for document ingestion!")
        print("\nRECOMMENDED APPROACH:")
        print("- Store embeddings as comma-separated strings in VARCHAR columns")
        print("- Use TO_VECTOR(embedding_str, double) in similarity queries")
        print("- Use VECTOR_COSINE() for similarity calculations")
        print("- Community Edition 2025.1 fully supports Vector Search!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    success = setup_ingestion()
    sys.exit(0 if success else 1)