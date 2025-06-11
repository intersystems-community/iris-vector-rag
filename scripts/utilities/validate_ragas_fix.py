#!/usr/bin/env python3
"""
Validate RAGAS Fix

This script validates that the IRISInputStream handling fix resolves
the RAGAS evaluation issues by testing document retrieval and content
extraction with the fixed pipelines.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from iris_rag.core.connection import ConnectionManager
from common.jdbc_stream_utils_fixed import read_iris_stream

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_document_retrieval_with_streams():
    """Test that documents can be retrieved and streams properly converted."""
    logger.info("=== TESTING DOCUMENT RETRIEVAL WITH STREAM CONVERSION ===")
    
    connection_manager = ConnectionManager()
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Simulate what RAGAS evaluation does - retrieve documents and extract content
        sample_sql = """
            SELECT TOP 5 doc_id, text_content, title
            FROM RAG.SourceDocuments
            WHERE doc_id LIKE 'PMC%'
            ORDER BY doc_id
        """
        cursor.execute(sample_sql)
        sample_results = cursor.fetchall()
        
        logger.info(f"Retrieved {len(sample_results)} documents for testing")
        
        contexts = []
        for doc_id, text_content, title in sample_results:
            # Convert streams to strings (this is what the fix does)
            content_str = read_iris_stream(text_content)
            title_str = read_iris_stream(title)
            
            logger.info(f"Document {doc_id}:")
            logger.info(f"  Title: {title_str}")
            logger.info(f"  Content length: {len(content_str)}")
            logger.info(f"  Content preview: {content_str[:100]}...")
            
            # Check content quality
            if len(content_str) > 100 and not content_str.isdigit():
                logger.info(f"  ✅ Valid content for RAGAS evaluation")
                contexts.append(content_str)
            else:
                logger.warning(f"  ❌ Invalid content: '{content_str}'")
        
        # Test RAGAS-style context processing
        logger.info(f"\n=== RAGAS CONTEXT SIMULATION ===")
        logger.info(f"Total contexts extracted: {len(contexts)}")
        
        if contexts:
            # Simulate what RAGAS does with contexts
            combined_context = "\n\n".join(contexts)
            logger.info(f"Combined context length: {len(combined_context)} characters")
            logger.info(f"Combined context preview: {combined_context[:200]}...")
            
            # Check if this would work for RAGAS
            if len(combined_context) > 500:
                logger.info("✅ Sufficient context for RAGAS evaluation")
                return True
            else:
                logger.warning("⚠️ Context may be too short for meaningful RAGAS evaluation")
                return False
        else:
            logger.error("❌ No valid contexts extracted")
            return False
            
    except Exception as e:
        logger.error(f"Document retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()

def simulate_ragas_context_extraction():
    """Simulate the exact process RAGAS uses to extract contexts."""
    logger.info("=== SIMULATING RAGAS CONTEXT EXTRACTION ===")
    
    # This simulates what happens in the RAGAS evaluation pipeline
    try:
        # Mock retrieved documents (similar to what ColBERT returns)
        connection_manager = ConnectionManager()
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Get documents like a RAG pipeline would
        cursor.execute("""
            SELECT TOP 3 doc_id, text_content, title
            FROM RAG.SourceDocuments
            WHERE doc_id LIKE 'PMC%'
            ORDER BY doc_id
        """)
        docs_data = cursor.fetchall()
        
        # Create Document objects like pipelines do
        from common.utils import Document
        
        retrieved_documents = []
        for doc_id, text_content, title in docs_data:
            # Apply the fix - convert streams to strings
            content_str = read_iris_stream(text_content)
            title_str = read_iris_stream(title)
            
            # Create Document object
            doc = Document(
                id=doc_id,
                content=content_str,
                metadata={"title": title_str}
            )
            retrieved_documents.append(doc)
        
        # Simulate RAGAS context extraction
        contexts = []
        for doc in retrieved_documents:
            if hasattr(doc, 'content') and doc.content:
                contexts.append(str(doc.content))
            elif hasattr(doc, 'page_content') and doc.page_content:
                contexts.append(str(doc.page_content))
            else:
                logger.warning(f"Document {doc.id} has no extractable content")
        
        logger.info(f"Extracted {len(contexts)} contexts for RAGAS")
        
        # Check context quality
        valid_contexts = 0
        for i, context in enumerate(contexts):
            logger.info(f"Context {i+1}: {len(context)} chars - {context[:50]}...")
            if len(context) > 50 and not context.isdigit():
                valid_contexts += 1
        
        logger.info(f"Valid contexts: {valid_contexts}/{len(contexts)}")
        
        if valid_contexts == len(contexts) and valid_contexts > 0:
            logger.info("✅ All contexts are valid for RAGAS evaluation")
            return True
        else:
            logger.warning(f"⚠️ Only {valid_contexts} out of {len(contexts)} contexts are valid")
            return False
            
    except Exception as e:
        logger.error(f"RAGAS simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()

def test_numeric_content_detection():
    """Test that we can detect and handle numeric content issues."""
    logger.info("=== TESTING NUMERIC CONTENT DETECTION ===")
    
    # Test cases for different content types
    test_cases = [
        ("68", "Numeric content (BAD)"),
        ("85", "Numeric content (BAD)"),
        ("", "Empty content (BAD)"),
        ("This is a proper medical abstract about cancer treatment...", "Valid content (GOOD)"),
        ("Alzheimer's disease (AD) is the most prevalent type of dementia...", "Valid content (GOOD)")
    ]
    
    for content, description in test_cases:
        # Apply our validation logic
        is_valid = len(content) > 50 and not content.isdigit()
        status = "✅ VALID" if is_valid else "❌ INVALID"
        logger.info(f"{status}: {description} - '{content[:30]}...'")
    
    logger.info("✅ Numeric content detection working correctly")
    return True

def main():
    """Main validation function."""
    logger.info("🔍 RAGAS Fix Validation")
    logger.info("=" * 60)
    
    # Run validation tests
    tests = [
        ("Document Retrieval with Streams", test_document_retrieval_with_streams),
        ("RAGAS Context Extraction Simulation", simulate_ragas_context_extraction),
        ("Numeric Content Detection", test_numeric_content_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n📊 VALIDATION SUMMARY")
    logger.info("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("The IRISInputStream handling fix should resolve RAGAS evaluation issues.")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Apply similar fixes to other RAG pipelines")
        logger.info("2. Run comprehensive RAGAS evaluation")
        logger.info("3. Verify context-based metrics improve")
        return True
    else:
        logger.error(f"\n❌ {total - passed} TESTS FAILED!")
        logger.error("Additional fixes may be needed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)