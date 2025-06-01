#!/usr/bin/env python3
"""
Test RAG system functionality while ingestion continues in background.
Tests multiple RAG techniques with various query types.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent # Corrected path to project root
sys.path.insert(0, str(project_root))

from src.common.iris_connector import get_iris_connection, IRISConnectionError # Updated import
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.working.colbert.pipeline import ColbertRAGPipeline # Updated import
from src.deprecated.colbert.pipeline import OptimizedColbertRAGPipeline # Updated import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_database_state():
    """Check current database state and document count."""
    logger.info("🔍 Checking current database state...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Check document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        
        # Check token embedding count
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        
        # Check sample document
        cursor.execute("SELECT TOP 1 doc_id, title, text_content FROM RAG.SourceDocuments")
        sample_doc = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        logger.info(f"📊 Database State:")
        logger.info(f"   Documents: {doc_count:,}")
        logger.info(f"   Token embeddings: {token_count:,}")
        if sample_doc:
            logger.info(f"   Sample doc: {sample_doc[0]} - {sample_doc[1][:50]}...")
        
        return {
            'doc_count': doc_count,
            'token_count': token_count,
            'sample_doc': sample_doc
        }
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return None

def test_basic_rag(queries: List[str]) -> Dict[str, Any]:
    """Test Basic RAG pipeline."""
    logger.info("🧪 Testing Basic RAG Pipeline...")
    
    try:
        conn = get_iris_connection()
        pipeline = BasicRAGPipeline(conn)
        
        results = {}
        for i, query in enumerate(queries):
            logger.info(f"   Query {i+1}: {query}")
            start_time = time.time()
            
            result = pipeline.run(query, top_k=3)
            
            end_time = time.time()
            query_time = end_time - start_time
            
            results[f"query_{i+1}"] = {
                'query': query,
                'answer': result.get('answer', 'No answer generated'),
                'retrieved_docs': len(result.get('retrieved_documents', [])),
                'query_time': query_time
            }
            
            logger.info(f"      Answer: {result.get('answer', 'No answer')[:100]}...")
            logger.info(f"      Retrieved: {len(result.get('retrieved_documents', []))} docs")
            logger.info(f"      Time: {query_time:.2f}s")
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"❌ Basic RAG test failed: {e}")
        return {'error': str(e)}

def test_colbert_rag(queries: List[str]) -> Dict[str, Any]:
    """Test ColBERT RAG pipeline."""
    logger.info("🧪 Testing ColBERT RAG Pipeline...")
    
    try:
        conn = get_iris_connection()
        pipeline = ColbertRAGPipeline(conn)
        
        results = {}
        for i, query in enumerate(queries):
            logger.info(f"   Query {i+1}: {query}")
            start_time = time.time()
            
            result = pipeline.run(query, top_k=3)
            
            end_time = time.time()
            query_time = end_time - start_time
            
            results[f"query_{i+1}"] = {
                'query': query,
                'answer': result.get('answer', 'No answer generated'),
                'retrieved_docs': len(result.get('retrieved_documents', [])),
                'query_time': query_time
            }
            
            logger.info(f"      Answer: {result.get('answer', 'No answer')[:100]}...")
            logger.info(f"      Retrieved: {len(result.get('retrieved_documents', []))} docs")
            logger.info(f"      Time: {query_time:.2f}s")
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"❌ ColBERT RAG test failed: {e}")
        return {'error': str(e)}

def test_optimized_colbert_rag(queries: List[str]) -> Dict[str, Any]:
    """Test Optimized ColBERT RAG pipeline."""
    logger.info("🧪 Testing Optimized ColBERT RAG Pipeline...")
    
    try:
        conn = get_iris_connection()
        pipeline = OptimizedColbertRAGPipeline(conn)
        
        results = {}
        for i, query in enumerate(queries):
            logger.info(f"   Query {i+1}: {query}")
            start_time = time.time()
            
            result = pipeline.run(query, top_k=3)
            
            end_time = time.time()
            query_time = end_time - start_time
            
            results[f"query_{i+1}"] = {
                'query': query,
                'answer': result.get('answer', 'No answer generated'),
                'retrieved_docs': len(result.get('retrieved_documents', [])),
                'query_time': query_time
            }
            
            logger.info(f"      Answer: {result.get('answer', 'No answer')[:100]}...")
            logger.info(f"      Retrieved: {len(result.get('retrieved_documents', []))} docs")
            logger.info(f"      Time: {query_time:.2f}s")
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"❌ Optimized ColBERT RAG test failed: {e}")
        return {'error': str(e)}

def run_benchmark_queries():
    """Run comprehensive benchmark queries."""
    logger.info("🚀 Starting RAG System Benchmark While Ingestion Continues...")
    
    # Test queries covering different domains
    test_queries = [
        "What are the main causes of diabetes?",
        "How does machine learning work in healthcare?",
        "What is the role of inflammation in disease?",
        "Explain the mechanism of protein folding",
        "What are the latest treatments for cancer?"
    ]
    
    # Check database state
    db_state = check_database_state()
    if not db_state:
        logger.error("❌ Cannot proceed without database access")
        return
    
    # Initialize results
    benchmark_results = {
        'timestamp': time.time(),
        'database_state': db_state,
        'test_queries': test_queries,
        'results': {}
    }
    
    # Test Basic RAG
    logger.info("\n" + "="*60)
    basic_results = test_basic_rag(test_queries)
    benchmark_results['results']['basic_rag'] = basic_results
    
    # Test ColBERT RAG
    logger.info("\n" + "="*60)
    colbert_results = test_colbert_rag(test_queries)
    benchmark_results['results']['colbert_rag'] = colbert_results
    
    # Test Optimized ColBERT RAG
    logger.info("\n" + "="*60)
    optimized_colbert_results = test_optimized_colbert_rag(test_queries)
    benchmark_results['results']['optimized_colbert_rag'] = optimized_colbert_results
    
    # Save results
    results_file = f"rag_benchmark_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    logger.info(f"\n📊 Benchmark Results Summary:")
    logger.info(f"   Database: {db_state['doc_count']:,} docs, {db_state['token_count']:,} tokens")
    logger.info(f"   Test queries: {len(test_queries)}")
    logger.info(f"   Results saved to: {results_file}")
    
    # Performance summary
    logger.info(f"\n⚡ Performance Summary:")
    for technique, results in benchmark_results['results'].items():
        if 'error' not in results:
            avg_time = sum(r['query_time'] for r in results.values() if 'query_time' in r) / len(test_queries)
            total_docs = sum(r['retrieved_docs'] for r in results.values() if 'retrieved_docs' in r)
            logger.info(f"   {technique}: {avg_time:.2f}s avg, {total_docs} total docs retrieved")
        else:
            logger.info(f"   {technique}: ERROR - {results['error']}")
    
    return benchmark_results

if __name__ == "__main__":
    try:
        results = run_benchmark_queries()
        logger.info("✅ RAG benchmark completed successfully!")
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1)