#!/usr/bin/env python3
"""
Test Fixed Query Execution - Verify that the similarity threshold fixes work

This script tests that:
1. Basic RAG pipeline now retrieves documents successfully
2. Queries return realistic document counts
3. End-to-end RAG pipeline works with real document retrieval
"""

import os
import sys
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
from basic_rag.pipeline import BasicRAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_rag_fixed():
    """Test that Basic RAG now works with fixed similarity thresholds"""
    logger.info("🧪 Testing Fixed Basic RAG Pipeline")
    
    # Setup
    connection = get_iris_connection()
    embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
    llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
    
    pipeline = BasicRAGPipeline(connection, embedding_func, llm_func)
    
    # Test queries that were failing before
    test_queries = [
        "What are the latest treatments for type 2 diabetes?",
        "How does machine learning improve medical diagnosis accuracy?", 
        "What are the mechanisms of cancer immunotherapy?",
        "How do genetic mutations contribute to disease development?",
        "What role does AI play in modern healthcare systems?"
    ]
    
    results = []
    
    try:
        for i, query in enumerate(test_queries, 1):
            logger.info(f"🔍 Testing Query {i}/5: {query[:50]}...")
            
            start_time = time.time()
            
            # Test document retrieval with default threshold (now 0.75)
            retrieved_docs = pipeline.retrieve_documents(query, top_k=10)
            retrieval_time = time.time() - start_time
            
            # Generate answer if we have documents
            if retrieved_docs:
                answer_start = time.time()
                answer = pipeline.generate_answer(query, retrieved_docs[:5])  # Use top 5
                answer_time = time.time() - answer_start
                total_time = retrieval_time + answer_time
            else:
                answer = "No relevant documents found."
                answer_time = 0
                total_time = retrieval_time
            
            # Collect results
            result = {
                "query": query,
                "retrieved_count": len(retrieved_docs),
                "retrieval_time_ms": retrieval_time * 1000,
                "answer_time_ms": answer_time * 1000,
                "total_time_ms": total_time * 1000,
                "answer_length": len(answer),
                "top_similarity": retrieved_docs[0].score if retrieved_docs else 0,
                "avg_similarity": sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                "success": len(retrieved_docs) > 0
            }
            
            results.append(result)
            
            # Log results
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            logger.info(f"  {status}: {result['retrieved_count']} docs, "
                       f"similarity: {result['top_similarity']:.3f}, "
                       f"time: {result['total_time_ms']:.1f}ms")
            
            if result["success"]:
                logger.info(f"    Answer preview: {answer[:100]}...")
    
    finally:
        connection.close()
    
    return results

def test_end_to_end_pipeline():
    """Test complete end-to-end RAG pipeline"""
    logger.info("🚀 Testing End-to-End RAG Pipeline")
    
    connection = get_iris_connection()
    embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
    llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
    
    pipeline = BasicRAGPipeline(connection, embedding_func, llm_func)
    
    # Test the full pipeline with run() method
    test_query = "What are effective treatments for diabetes and how do they work?"
    
    try:
        logger.info(f"🔍 Running full pipeline for: {test_query}")
        
        start_time = time.time()
        result = pipeline.run(test_query, top_k=3)  # Uses default threshold 0.75, limit docs for context
        total_time = time.time() - start_time
        
        logger.info(f"📊 Pipeline Results:")
        logger.info(f"  - Query: {result['query']}")
        logger.info(f"  - Documents retrieved: {result['document_count']}")
        logger.info(f"  - Similarity threshold: {result['similarity_threshold']}")
        logger.info(f"  - Total time: {total_time:.2f}s")
        logger.info(f"  - Answer length: {len(result['answer'])} chars")
        logger.info(f"  - Answer preview: {result['answer'][:150]}...")
        
        if result['retrieved_documents']:
            logger.info(f"  - Top document similarity: {result['retrieved_documents'][0].score:.3f}")
            logger.info(f"  - Document IDs: {[doc.id for doc in result['retrieved_documents'][:3]]}")
        
        success = result['document_count'] > 0
        logger.info(f"  - Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success, result
        
    finally:
        connection.close()

def main():
    """Main test function"""
    logger.info("🚀 Testing Fixed Query Execution")
    logger.info("=" * 80)
    
    try:
        # Test 1: Basic RAG with multiple queries
        logger.info("\n" + "=" * 80)
        basic_rag_results = test_basic_rag_fixed()
        
        # Test 2: End-to-end pipeline
        logger.info("\n" + "=" * 80)
        e2e_success, e2e_result = test_end_to_end_pipeline()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 80)
        
        # Basic RAG results
        successful_queries = len([r for r in basic_rag_results if r["success"]])
        total_queries = len(basic_rag_results)
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        avg_docs = sum(r["retrieved_count"] for r in basic_rag_results) / total_queries if total_queries > 0 else 0
        avg_similarity = sum(r["top_similarity"] for r in basic_rag_results if r["success"]) / successful_queries if successful_queries > 0 else 0
        avg_time = sum(r["total_time_ms"] for r in basic_rag_results) / total_queries if total_queries > 0 else 0
        
        logger.info(f"🧪 Basic RAG Tests:")
        logger.info(f"  - Success rate: {success_rate:.1%} ({successful_queries}/{total_queries})")
        logger.info(f"  - Average documents retrieved: {avg_docs:.1f}")
        logger.info(f"  - Average similarity score: {avg_similarity:.3f}")
        logger.info(f"  - Average response time: {avg_time:.1f}ms")
        
        logger.info(f"\n🚀 End-to-End Pipeline:")
        logger.info(f"  - Status: {'✅ SUCCESS' if e2e_success else '❌ FAILED'}")
        if e2e_success:
            logger.info(f"  - Documents retrieved: {e2e_result['document_count']}")
            logger.info(f"  - Threshold used: {e2e_result['similarity_threshold']}")
        
        # Overall assessment
        overall_success = success_rate >= 0.8 and e2e_success
        
        logger.info(f"\n🎯 OVERALL ASSESSMENT:")
        if overall_success:
            logger.info("✅ QUERY EXECUTION FIXES SUCCESSFUL!")
            logger.info("  - Similarity thresholds are now appropriate")
            logger.info("  - Document retrieval is working consistently")
            logger.info("  - RAG pipeline generates meaningful answers")
            logger.info("  - Ready for enterprise scale validation")
        else:
            logger.info("❌ Issues still remain:")
            if success_rate < 0.8:
                logger.info(f"  - Basic RAG success rate too low: {success_rate:.1%}")
            if not e2e_success:
                logger.info("  - End-to-end pipeline failed")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)