#!/usr/bin/env python3
"""
Debug script to investigate BasicRAG context extraction issues
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import logging
from basic_rag.pipeline import BasicRAGPipeline
from common.utils import get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_basicrag_context():
    """Debug BasicRAG context extraction"""
    print("🔍 Debugging BasicRAG Context Extraction...")
    
    try:
        # Initialize components
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="openai")
        
        # Create pipeline
        pipeline = BasicRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )
        
        # Test query
        test_query = "What are the effects of metformin on type 2 diabetes?"
        print(f"\n📝 Test Query: {test_query}")
        
        # Step 1: Test document retrieval
        print("\n🔍 Step 1: Testing document retrieval...")
        retrieved_docs = pipeline.retrieve_documents(test_query, top_k=3)
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1}:")
            print(f"  ID: {doc.id}")
            print(f"  Score: {doc.score}")
            print(f"  Content type: {type(doc.content)}")
            print(f"  Content length: {len(str(doc.content)) if doc.content else 0}")
            print(f"  Content preview: {str(doc.content)[:100]}...")
            if hasattr(doc, '_title'):
                print(f"  Title: {doc._title}")
        
        # Step 2: Test full pipeline
        print("\n🔍 Step 2: Testing full pipeline...")
        result = pipeline.run(test_query, top_k=3)
        
        print(f"\nPipeline Result:")
        print(f"  Query: {result['query']}")
        print(f"  Answer length: {len(result['answer'])}")
        print(f"  Answer preview: {result['answer'][:200]}...")
        print(f"  Retrieved documents count: {len(result['retrieved_documents'])}")
        print(f"  Contexts count: {len(result['contexts'])}")
        
        print(f"\nContexts for RAGAS:")
        for i, context in enumerate(result['contexts']):
            print(f"  Context {i+1}: {type(context)} - {context[:100]}...")
        
        # Step 3: Check raw database query
        print("\n🔍 Step 3: Testing raw database query...")
        cursor = db_conn.cursor()
        
        # Test the exact query used by BasicRAG
        query_embedding = embed_fn([test_query])[0]
        query_vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        sql = """
            SELECT TOP 3
                doc_id,
                title,
                text_content,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY similarity_score DESC
        """
        
        cursor.execute(sql, [query_vector_str])
        raw_results = cursor.fetchall()
        
        print(f"Raw database results: {len(raw_results)} rows")
        for i, row in enumerate(raw_results):
            print(f"\nRaw Row {i+1}:")
            print(f"  doc_id type: {type(row[0])}, value: {row[0]}")
            print(f"  title type: {type(row[1])}, value: {str(row[1])[:50]}...")
            print(f"  text_content type: {type(row[2])}, value: {str(row[2])[:50]}...")
            print(f"  similarity_score: {row[3]}")
        
        cursor.close()
        db_conn.close()
        
    except Exception as e:
        logger.error(f"Debug failed: {e}", exc_info=True)
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_basicrag_context()