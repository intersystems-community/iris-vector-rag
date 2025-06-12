#!/usr/bin/env python3
"""
Validation script to test IRISInputStream handling fixes across all RAG pipelines.

This script validates that all RAG pipelines properly handle IRISInputStream objects
when retrieving document content from the database.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Import all pipeline classes
from basic_rag.pipeline import BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline import GraphRAGPipeline
from crag.pipeline import CRAGPipeline

logger = logging.getLogger(__name__)

def validate_pipeline_result(pipeline_name: str, result: Dict[str, Any]) -> bool:
    """
    Validate that a pipeline result contains proper string content (not stream objects).
    
    Args:
        pipeline_name: Name of the pipeline being tested
        result: Pipeline result dictionary
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info(f"Validating {pipeline_name} pipeline result...")
    
    # Check basic structure
    required_keys = ['query', 'answer', 'retrieved_documents']
    for key in required_keys:
        if key not in result:
            logger.error(f"{pipeline_name}: Missing required key '{key}' in result")
            return False
    
    # Validate answer is a string
    if not isinstance(result['answer'], str):
        logger.error(f"{pipeline_name}: Answer is not a string: {type(result['answer'])}")
        return False
    
    # Validate retrieved documents
    retrieved_docs = result['retrieved_documents']
    if not isinstance(retrieved_docs, list):
        logger.error(f"{pipeline_name}: retrieved_documents is not a list: {type(retrieved_docs)}")
        return False
    
    if len(retrieved_docs) == 0:
        logger.warning(f"{pipeline_name}: No documents retrieved (this may be expected)")
        return True
    
    # Check each document
    for i, doc in enumerate(retrieved_docs):
        if not isinstance(doc, dict):
            logger.error(f"{pipeline_name}: Document {i} is not a dict: {type(doc)}")
            return False
        
        # Check document content
        if 'content' in doc:
            content = doc['content']
            if not isinstance(content, str):
                logger.error(f"{pipeline_name}: Document {i} content is not a string: {type(content)}")
                return False
            
            # Check for stream object references (these indicate the fix didn't work)
            if 'IRISInputStream@' in str(content):
                logger.error(f"{pipeline_name}: Document {i} content contains stream reference: {content[:100]}")
                return False
            
            # Check for meaningful content (not just empty or None)
            if content and content.strip() and content != "None":
                logger.debug(f"{pipeline_name}: Document {i} has valid content: {content[:50]}...")
            else:
                logger.warning(f"{pipeline_name}: Document {i} has empty/None content")
    
    logger.info(f"{pipeline_name}: Validation PASSED - {len(retrieved_docs)} documents with proper string content")
    return True

def test_pipeline(pipeline_class, pipeline_name: str, iris_connector, embedding_func, llm_func) -> bool:
    """
    Test a single pipeline for proper stream handling.
    
    Args:
        pipeline_class: Pipeline class to instantiate
        pipeline_name: Name of the pipeline for logging
        iris_connector: Database connection
        embedding_func: Embedding function
        llm_func: LLM function
        
    Returns:
        bool: True if test passes, False otherwise
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing {pipeline_name} Pipeline")
    logger.info(f"{'='*50}")
    
    try:
        # Create pipeline instance
        if pipeline_name == "CRAG":
            # CRAG has additional parameters
            pipeline = pipeline_class(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func,
                web_search_func=None,  # No web search for testing
                use_chunks=False  # Use document retrieval for testing
            )
        elif pipeline_name == "NodeRAG":
            # NodeRAG has graph_lib parameter
            pipeline = pipeline_class(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func,
                graph_lib=None
            )
        else:
            # Standard pipeline initialization
            pipeline = pipeline_class(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func
            )
        
        # Test with a simple query
        test_query = "What is diabetes?"
        logger.info(f"Running {pipeline_name} with query: '{test_query}'")
        
        # Run pipeline
        result = pipeline.run(test_query, top_k=3)
        
        # Validate result
        validation_passed = validate_pipeline_result(pipeline_name, result)
        
        if validation_passed:
            logger.info(f"✅ {pipeline_name} pipeline PASSED stream handling validation")
        else:
            logger.error(f"❌ {pipeline_name} pipeline FAILED stream handling validation")
        
        return validation_passed
        
    except Exception as e:
        logger.error(f"❌ {pipeline_name} pipeline ERROR: {e}", exc_info=True)
        return False

def main():
    """Main validation function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting IRISInputStream handling validation for all RAG pipelines...")
    
    # Get database connection and functions
    try:
        iris_connector = get_iris_connection()
        embedding_func = get_embedding_func()
        llm_func = get_llm_func(provider="stub")  # Use stub for testing
        
        if iris_connector is None:
            logger.error("Failed to get IRIS connection")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize test dependencies: {e}")
        return False
    
    # Define pipelines to test
    pipelines_to_test = [
        (BasicRAGPipeline, "BasicRAG"),
        (HyDEPipeline, "HyDE"),
        (NodeRAGPipeline, "NodeRAG"),
        (GraphRAGPipeline, "GraphRAG"),
        (CRAGPipeline, "CRAG")
    ]
    
    # Test each pipeline
    results = {}
    for pipeline_class, pipeline_name in pipelines_to_test:
        try:
            success = test_pipeline(pipeline_class, pipeline_name, iris_connector, embedding_func, llm_func)
            results[pipeline_name] = success
        except Exception as e:
            logger.error(f"Unexpected error testing {pipeline_name}: {e}")
            results[pipeline_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    passed_count = 0
    total_count = len(results)
    
    for pipeline_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{pipeline_name:12}: {status}")
        if success:
            passed_count += 1
    
    logger.info(f"\nOverall: {passed_count}/{total_count} pipelines passed validation")
    
    # Close connection
    try:
        iris_connector.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.warning(f"Error closing database connection: {e}")
    
    # Return success if all pipelines passed
    all_passed = passed_count == total_count
    if all_passed:
        logger.info("🎉 ALL PIPELINES PASSED - IRISInputStream handling fixes are working correctly!")
    else:
        logger.error("⚠️  SOME PIPELINES FAILED - IRISInputStream handling fixes need attention")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)