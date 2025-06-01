#!/usr/bin/env python3
"""
Test to reproduce and fix the BasicRAG issue.
Based on evidence from logs, BasicRAG was working earlier but broke due to schema changes.
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) # Assuming script is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from src.common.iris_connector_jdbc import get_iris_connection # Updated import
from src.common.utils import get_embedding_func, get_llm_func # Updated import

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_current_basic_rag():
    """Test current BasicRAG implementation"""
    print("=== Testing Current BasicRAG Implementation ===")
    
    try:
        from src.deprecated.basic_rag.pipeline import BasicRAGPipeline as DeprecatedBasicRAGPipeline # Updated import
        
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")
        
        pipeline = DeprecatedBasicRAGPipeline( # Updated class name
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )
        
        # Test with a simple query
        result = pipeline.run("What is diabetes?", top_k=5)
        
        print(f"✅ Current BasicRAG Result:")
        print(f"   Documents retrieved: {result['document_count']}")
        print(f"   Query: {result['query']}")
        print(f"   Answer length: {len(result['answer'])} chars")
        
        if result['document_count'] > 0:
            print("   ✅ SUCCESS: BasicRAG is retrieving documents!")
            return True
        else:
            print("   ❌ PROBLEM: BasicRAG retrieved 0 documents")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'db_conn' in locals():
            db_conn.close()

def test_pipeline_final():
    """Test pipeline_final.py implementation"""
    print("\n=== Testing BasicRAG pipeline_final.py Implementation ===")
    
    try:
        from src.experimental.basic_rag.pipeline_final import BasicRAGPipeline # Updated import
        
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")
        
        pipeline = BasicRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )
        
        # Test with a simple query
        result = pipeline.run("What is diabetes?", top_k=5)
        
        print(f"✅ pipeline_final.py Result:")
        print(f"   Documents retrieved: {result['document_count']}")
        print(f"   Query: {result['query']}")
        print(f"   Answer length: {len(result['answer'])} chars")
        
        if result['document_count'] > 0:
            print("   ✅ SUCCESS: pipeline_final.py is retrieving documents!")
            return True
        else:
            print("   ❌ PROBLEM: pipeline_final.py retrieved 0 documents")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'db_conn' in locals():
            db_conn.close()

def check_table_status():
    """Check what tables exist and their record counts"""
    print("\n=== Checking Table Status ===")
    
    try:
        db_conn = get_iris_connection()
        cursor = db_conn.cursor()
        
        tables_to_check = [
            "RAG.SourceDocuments",
            "RAG.SourceDocuments_V2", 
            "RAG.SourceDocuments_OLD"
        ]
        
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   ✅ {table}: {count} records")
                
                # Check if it has vector column
                cursor.execute(f"SELECT TOP 1 embedding FROM {table} WHERE embedding IS NOT NULL")
                has_embeddings = cursor.fetchone() is not None
                print(f"      Has embeddings: {has_embeddings}")
                
            except Exception as e:
                print(f"   ❌ {table}: ERROR - {e}")
        
        cursor.close()
        db_conn.close()
        
    except Exception as e:
        print(f"   ❌ Database connection error: {e}")

def test_v2_table_directly():
    """Test using SourceDocuments_V2 table directly"""
    print("\n=== Testing SourceDocuments_V2 Table Directly ===")
    
    try:
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        
        # Generate a test embedding
        query_embedding = embed_fn(["What is diabetes?"])[0]
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = db_conn.cursor()
        
        # Test direct SQL on V2 table
        sql = f"""
            SELECT TOP 5
                doc_id,
                title,
                text_content,
                VECTOR_COSINE(embedding, TO_VECTOR('{query_embedding_str}')) as similarity_score
            FROM RAG.SourceDocuments_V2
            WHERE embedding IS NOT NULL
              AND VECTOR_COSINE(embedding, TO_VECTOR('{query_embedding_str}')) > 0.1
            ORDER BY similarity_score DESC
        """
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        print(f"   ✅ Direct V2 table query result: {len(results)} documents")
        
        for i, row in enumerate(results[:3]):
            doc_id = row[0]
            title = row[1]
            score = row[3]
            print(f"      Doc {i+1}: ID={doc_id}, Score={score:.4f}, Title={title[:50]}...")
        
        cursor.close()
        db_conn.close()
        
        return len(results) > 0
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 INVESTIGATING BASICRAG ISSUE")
    print("=" * 50)
    
    # Check table status first
    check_table_status()
    
    # Test current implementation
    current_works = test_current_basic_rag()
    
    # Test pipeline_final implementation  
    final_works = test_pipeline_final()
    
    # Test V2 table directly
    v2_works = test_v2_table_directly()
    
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS SUMMARY:")
    print(f"   Current BasicRAG works: {current_works}")
    print(f"   pipeline_final.py works: {final_works}")
    print(f"   SourceDocuments_V2 has data: {v2_works}")
    
    if not current_works and v2_works:
        print("\n💡 SOLUTION IDENTIFIED:")
        print("   The issue is that BasicRAG is using the old SourceDocuments table")
        print("   but the V2 migration moved the working data to SourceDocuments_V2.")
        print("   BasicRAG needs to be updated to use SourceDocuments_V2.")
    
    print("\n🔧 NEXT STEPS:")
    print("   1. Update BasicRAG to use SourceDocuments_V2 table")
    print("   2. Test that it retrieves 5 documents as it did before")
    print("   3. Restore the working state from earlier today")