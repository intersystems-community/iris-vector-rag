#!/usr/bin/env python3
"""
Test script to validate the RAGAS context fixes for ColBERTRAG and NodeRAG pipelines.

This script tests the specific fixes implemented to address:
1. ColBERTRAG retrieving irrelevant documents
2. NodeRAG retrieving only node names instead of full content
"""

import sys
import os
import logging
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.colbert import ColBERTRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_colbert_candidate_retrieval():
    """Test ColBERT candidate document retrieval with enhanced logging."""
    logger.info("=== Testing ColBERT Candidate Retrieval ===")
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Create ColBERT pipeline
        colbert_pipeline = ColBERTRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        # Test medical query
        test_query = "What are the symptoms of diabetes?"
        logger.info(f"Testing query: '{test_query}'")
        
        # Test candidate retrieval
        candidates = colbert_pipeline._retrieve_candidate_documents_hnsw(test_query, k=5)
        
        if candidates:
            logger.info(f"✅ ColBERT retrieved {len(candidates)} candidates")
            logger.info(f"Candidate IDs: {candidates[:3]}...")
            return True
        else:
            logger.warning("⚠️ ColBERT retrieved no candidates")
            return False
            
    except Exception as e:
        logger.error(f"❌ ColBERT test failed: {e}")
        return False

def test_noderag_content_retrieval():
    """Test NodeRAG content retrieval with full text instead of just node names."""
    logger.info("=== Testing NodeRAG Content Retrieval ===")
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Create NodeRAG pipeline
        noderag_pipeline = NodeRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        # Test medical query
        test_query = "What are the symptoms of diabetes?"
        logger.info(f"Testing query: '{test_query}'")
        
        # Test the full pipeline to see content quality
        result = noderag_pipeline.run(test_query, top_k=3)
        
        if result and result.get('retrieved_documents'):
            docs = result['retrieved_documents']
            logger.info(f"✅ NodeRAG retrieved {len(docs)} documents")
            
            # Check content quality
            for i, doc in enumerate(docs[:2]):
                content = doc.get('page_content', '') if isinstance(doc, dict) else getattr(doc, 'page_content', '')
                content_length = len(content)
                logger.info(f"Document {i+1}: {content_length} characters")
                
                if content_length > 50:
                    logger.info(f"Content preview: {content[:100]}...")
                    if content_length < 100:
                        logger.warning(f"⚠️ Short content for document {i+1}")
                else:
                    logger.warning(f"⚠️ Very short content for document {i+1}: '{content}'")
            
            return True
        else:
            logger.warning("⚠️ NodeRAG retrieved no documents")
            return False
            
    except Exception as e:
        logger.error(f"❌ NodeRAG test failed: {e}")
        return False

def test_ragas_context_quality():
    """Test that both pipelines provide good context for RAGAS evaluation."""
    logger.info("=== Testing RAGAS Context Quality ===")
    
    test_query = "What are the symptoms of diabetes?"
    results = {}
    
    # Test ColBERT
    try:
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        colbert_pipeline = ColBERTRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        colbert_result = colbert_pipeline.run(test_query, top_k=3)
        if colbert_result and colbert_result.get('contexts'):
            contexts = colbert_result['contexts']
            total_context_length = sum(len(ctx) for ctx in contexts)
            logger.info(f"✅ ColBERT contexts: {len(contexts)} items, {total_context_length} total chars")
            results['colbert'] = {'contexts': len(contexts), 'total_length': total_context_length}
        else:
            logger.warning("⚠️ ColBERT provided no contexts")
            results['colbert'] = {'contexts': 0, 'total_length': 0}
            
    except Exception as e:
        logger.error(f"❌ ColBERT context test failed: {e}")
        results['colbert'] = {'error': str(e)}
    
    # Test NodeRAG
    try:
        noderag_pipeline = NodeRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        noderag_result = noderag_pipeline.run(test_query, top_k=3)
        if noderag_result and noderag_result.get('contexts'):
            contexts = noderag_result['contexts']
            total_context_length = sum(len(ctx) for ctx in contexts)
            logger.info(f"✅ NodeRAG contexts: {len(contexts)} items, {total_context_length} total chars")
            results['noderag'] = {'contexts': len(contexts), 'total_length': total_context_length}
        else:
            logger.warning("⚠️ NodeRAG provided no contexts")
            results['noderag'] = {'contexts': 0, 'total_length': 0}
            
    except Exception as e:
        logger.error(f"❌ NodeRAG context test failed: {e}")
        results['noderag'] = {'error': str(e)}
    
    return results

def main():
    """Run all tests and report results."""
    logger.info("Starting RAGAS Context Fixes Validation")
    logger.info("=" * 50)
    
    # Run individual tests
    colbert_success = test_colbert_candidate_retrieval()
    noderag_success = test_noderag_content_retrieval()
    context_results = test_ragas_context_quality()
    
    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY:")
    logger.info(f"ColBERT candidate retrieval: {'✅ PASS' if colbert_success else '❌ FAIL'}")
    logger.info(f"NodeRAG content retrieval: {'✅ PASS' if noderag_success else '❌ FAIL'}")
    
    for pipeline, result in context_results.items():
        if 'error' in result:
            logger.info(f"{pipeline.upper()} context quality: ❌ ERROR - {result['error']}")
        else:
            contexts = result['contexts']
            total_length = result['total_length']
            status = "✅ GOOD" if contexts > 0 and total_length > 100 else "⚠️ POOR"
            logger.info(f"{pipeline.upper()} context quality: {status} ({contexts} contexts, {total_length} chars)")
    
    overall_success = colbert_success and noderag_success
    logger.info(f"\nOverall test result: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)