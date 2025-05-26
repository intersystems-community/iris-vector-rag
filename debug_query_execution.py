#!/usr/bin/env python3
"""
Debug Query Execution Issues - Investigate "0 docs" results

This script will:
1. Check database state and document counts
2. Test vector similarity queries with different thresholds
3. Show actual similarity scores and document retrieval
4. Verify embeddings are stored correctly
5. Test the RAG pipeline with realistic thresholds
"""

import os
import sys
import logging
import time
import numpy as np
from typing import List, Dict, Any

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

def check_database_state():
    """Check the current state of the database"""
    logger.info("🔍 Checking database state...")
    
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    try:
        # Check document counts
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        docs_with_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG_HNSW.SourceDocuments WHERE embedding IS NOT NULL")
        hnsw_docs = cursor.fetchone()[0]
        
        # Check embedding dimensions
        cursor.execute("SELECT TOP 1 embedding FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        sample_embedding = cursor.fetchone()
        if sample_embedding:
            embedding_str = sample_embedding[0]
            embedding_list = [float(x) for x in embedding_str.split(',')]
            embedding_dims = len(embedding_list)
        else:
            embedding_dims = 0
        
        # Check sample documents
        cursor.execute("SELECT TOP 5 doc_id, title FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        sample_docs = cursor.fetchall()
        
        logger.info(f"📊 Database State:")
        logger.info(f"  - Total documents: {total_docs}")
        logger.info(f"  - Documents with embeddings: {docs_with_embeddings}")
        logger.info(f"  - HNSW documents: {hnsw_docs}")
        logger.info(f"  - Embedding dimensions: {embedding_dims}")
        logger.info(f"  - Sample documents:")
        for doc in sample_docs:
            logger.info(f"    * {doc[0]}: {doc[1][:50] if doc[1] else 'No title'}...")
        
        return {
            "total_docs": total_docs,
            "docs_with_embeddings": docs_with_embeddings,
            "hnsw_docs": hnsw_docs,
            "embedding_dims": embedding_dims,
            "sample_docs": len(sample_docs)
        }
        
    finally:
        cursor.close()
        connection.close()

def test_similarity_thresholds():
    """Test vector similarity queries with different thresholds"""
    logger.info("🎯 Testing similarity thresholds...")
    
    connection = get_iris_connection()
    embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
    
    # Test query
    test_query = "diabetes treatment and management strategies"
    query_embedding = embedding_func([test_query])[0]
    query_vector_str = ','.join(map(str, query_embedding))
    
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    results = {}
    
    cursor = connection.cursor()
    
    try:
        for threshold in thresholds:
            logger.info(f"  Testing threshold: {threshold}")
            
            sql = """
            SELECT TOP 10 doc_id, title,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
            FROM RAG_HNSW.SourceDocuments
            WHERE embedding IS NOT NULL
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
            ORDER BY similarity DESC
            """
            
            start_time = time.time()
            cursor.execute(sql, (query_vector_str, query_vector_str, threshold))
            docs = cursor.fetchall()
            query_time = time.time() - start_time
            
            similarities = [float(doc[2]) for doc in docs if doc[2] is not None]
            
            results[threshold] = {
                "count": len(docs),
                "query_time_ms": query_time * 1000,
                "similarities": similarities,
                "top_similarity": similarities[0] if similarities else 0,
                "avg_similarity": np.mean(similarities) if similarities else 0,
                "sample_docs": [(doc[0], doc[1][:50], float(doc[2])) for doc in docs[:3]]
            }
            
            logger.info(f"    Results: {len(docs)} docs, top similarity: {similarities[0] if similarities else 0:.4f}")
    
    finally:
        cursor.close()
        connection.close()
    
    return results

def test_basic_rag_with_thresholds():
    """Test Basic RAG pipeline with different similarity thresholds"""
    logger.info("🔧 Testing Basic RAG with different thresholds...")
    
    connection = get_iris_connection()
    embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
    llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
    
    pipeline = BasicRAGPipeline(connection, embedding_func, llm_func)
    
    test_queries = [
        "What are the latest treatments for type 2 diabetes?",
        "How does machine learning improve medical diagnosis accuracy?",
        "What are the mechanisms of cancer immunotherapy?"
    ]
    
    thresholds = [0.7, 0.75, 0.8, 0.85]
    results = {}
    
    try:
        for threshold in thresholds:
            logger.info(f"  Testing threshold: {threshold}")
            threshold_results = []
            
            for query in test_queries:
                logger.info(f"    Query: {query[:50]}...")
                
                start_time = time.time()
                retrieved_docs = pipeline.retrieve_documents(query, top_k=10, similarity_threshold=threshold)
                retrieval_time = time.time() - start_time
                
                if retrieved_docs:
                    similarities = [doc.score for doc in retrieved_docs]
                    answer = pipeline.generate_answer(query, retrieved_docs[:5])  # Use top 5 for answer
                else:
                    similarities = []
                    answer = "No relevant documents found."
                
                query_result = {
                    "query": query,
                    "retrieved_count": len(retrieved_docs),
                    "retrieval_time_ms": retrieval_time * 1000,
                    "similarities": similarities,
                    "top_similarity": similarities[0] if similarities else 0,
                    "answer_length": len(answer),
                    "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer
                }
                
                threshold_results.append(query_result)
                logger.info(f"      Retrieved: {len(retrieved_docs)} docs, top similarity: {similarities[0] if similarities else 0:.4f}")
            
            results[threshold] = threshold_results
    
    finally:
        connection.close()
    
    return results

def show_actual_similarity_distribution():
    """Show the actual distribution of similarity scores in the database"""
    logger.info("📈 Analyzing similarity score distribution...")
    
    connection = get_iris_connection()
    embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
    
    # Test with multiple queries to get a good distribution
    test_queries = [
        "diabetes treatment",
        "machine learning medical diagnosis", 
        "cancer immunotherapy",
        "genetic mutations disease",
        "artificial intelligence healthcare"
    ]
    
    all_similarities = []
    cursor = connection.cursor()
    
    try:
        for query in test_queries:
            query_embedding = embedding_func([query])[0]
            query_vector_str = ','.join(map(str, query_embedding))
            
            sql = """
            SELECT TOP 100 VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
            FROM RAG_HNSW.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            """
            
            cursor.execute(sql, (query_vector_str,))
            similarities = [float(row[0]) for row in cursor.fetchall() if row[0] is not None]
            all_similarities.extend(similarities)
        
        # Calculate distribution statistics
        all_similarities = np.array(all_similarities)
        
        logger.info(f"📊 Similarity Score Distribution (from {len(all_similarities)} samples):")
        logger.info(f"  - Min: {np.min(all_similarities):.4f}")
        logger.info(f"  - Max: {np.max(all_similarities):.4f}")
        logger.info(f"  - Mean: {np.mean(all_similarities):.4f}")
        logger.info(f"  - Median: {np.median(all_similarities):.4f}")
        logger.info(f"  - 90th percentile: {np.percentile(all_similarities, 90):.4f}")
        logger.info(f"  - 75th percentile: {np.percentile(all_similarities, 75):.4f}")
        logger.info(f"  - 50th percentile: {np.percentile(all_similarities, 50):.4f}")
        logger.info(f"  - 25th percentile: {np.percentile(all_similarities, 25):.4f}")
        
        # Recommend optimal thresholds
        logger.info(f"💡 Recommended thresholds:")
        logger.info(f"  - Conservative (top 10%): {np.percentile(all_similarities, 90):.3f}")
        logger.info(f"  - Balanced (top 25%): {np.percentile(all_similarities, 75):.3f}")
        logger.info(f"  - Liberal (top 50%): {np.percentile(all_similarities, 50):.3f}")
        
        return {
            "min": float(np.min(all_similarities)),
            "max": float(np.max(all_similarities)),
            "mean": float(np.mean(all_similarities)),
            "median": float(np.median(all_similarities)),
            "p90": float(np.percentile(all_similarities, 90)),
            "p75": float(np.percentile(all_similarities, 75)),
            "p50": float(np.percentile(all_similarities, 50)),
            "p25": float(np.percentile(all_similarities, 25))
        }
        
    finally:
        cursor.close()
        connection.close()

def main():
    """Main debugging function"""
    logger.info("🚀 Starting Query Execution Debug Analysis")
    logger.info("=" * 80)
    
    try:
        # 1. Check database state
        db_state = check_database_state()
        
        if db_state["docs_with_embeddings"] == 0:
            logger.error("❌ No documents with embeddings found! Cannot proceed with testing.")
            return
        
        # 2. Analyze similarity score distribution
        logger.info("\n" + "=" * 80)
        similarity_stats = show_actual_similarity_distribution()
        
        # 3. Test different similarity thresholds
        logger.info("\n" + "=" * 80)
        threshold_results = test_similarity_thresholds()
        
        # 4. Test Basic RAG with different thresholds
        logger.info("\n" + "=" * 80)
        rag_results = test_basic_rag_with_thresholds()
        
        # 5. Summary and recommendations
        logger.info("\n" + "=" * 80)
        logger.info("🎯 ANALYSIS SUMMARY AND RECOMMENDATIONS")
        logger.info("=" * 80)
        
        logger.info(f"📊 Database Status:")
        logger.info(f"  - {db_state['docs_with_embeddings']} documents with embeddings ready for querying")
        logger.info(f"  - {db_state['embedding_dims']} dimensional embeddings")
        
        logger.info(f"\n📈 Similarity Score Analysis:")
        logger.info(f"  - Typical similarity range: {similarity_stats['min']:.3f} - {similarity_stats['max']:.3f}")
        logger.info(f"  - Mean similarity: {similarity_stats['mean']:.3f}")
        logger.info(f"  - 75th percentile: {similarity_stats['p75']:.3f}")
        
        logger.info(f"\n🎯 Threshold Performance:")
        for threshold, result in threshold_results.items():
            logger.info(f"  - Threshold {threshold}: {result['count']} docs, avg similarity {result['avg_similarity']:.3f}")
        
        logger.info(f"\n🔧 RAG Pipeline Performance:")
        for threshold, queries in rag_results.items():
            avg_retrieved = np.mean([q['retrieved_count'] for q in queries])
            success_rate = len([q for q in queries if q['retrieved_count'] > 0]) / len(queries)
            logger.info(f"  - Threshold {threshold}: {avg_retrieved:.1f} avg docs, {success_rate:.1%} success rate")
        
        # Recommendations
        optimal_threshold = similarity_stats['p75'] - 0.05  # Slightly below 75th percentile
        logger.info(f"\n💡 RECOMMENDATIONS:")
        logger.info(f"  - Use similarity threshold: {optimal_threshold:.3f} (instead of 0.85)")
        logger.info(f"  - This should retrieve documents for most queries while maintaining quality")
        logger.info(f"  - Current 0.85 threshold is too restrictive for this dataset")
        
        logger.info(f"\n✅ NEXT STEPS:")
        logger.info(f"  1. Update Basic RAG pipeline to use threshold {optimal_threshold:.3f}")
        logger.info(f"  2. Update enterprise validation script to use realistic thresholds")
        logger.info(f"  3. Re-run validation to verify document retrieval works")
        
    except Exception as e:
        logger.error(f"❌ Debug analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()