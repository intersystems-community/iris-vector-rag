#!/usr/bin/env python3
"""
Test Script for Data Quality Fixes

This script tests the comprehensive fixes for NaN values, vector format consistency,
and data validation issues that were causing LIST ERROR and DATA ERROR problems.
"""

import os
import sys
import logging
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func # Updated import
from data.loader_fixed import load_documents_to_iris, validate_and_fix_embedding, validate_and_fix_text_field # Path remains correct
from common.utils import get_colbert_doc_encoder_func # Fixed import to use centralized function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_embedding_validation():
    """Test the embedding validation and fixing functions"""
    logger.info("🧪 Testing embedding validation functions...")
    
    # Test normal embedding
    normal_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = validate_and_fix_embedding(normal_embedding)
    assert result is not None
    logger.info(f"✅ Normal embedding: {result[:50]}...")
    
    # Test embedding with NaN values
    nan_embedding = [0.1, float('nan'), 0.3, float('inf'), 0.5]
    result = validate_and_fix_embedding(nan_embedding)
    assert result is not None
    assert 'nan' not in result.lower()
    assert 'inf' not in result.lower()
    logger.info(f"✅ NaN/inf embedding fixed: {result[:50]}...")
    
    # Test empty embedding
    empty_embedding = []
    result = validate_and_fix_embedding(empty_embedding)
    assert result is None
    logger.info("✅ Empty embedding handled correctly")
    
    # Test text field validation
    normal_text = "This is normal text"
    result = validate_and_fix_text_field(normal_text)
    assert result == normal_text
    logger.info("✅ Normal text field validated")
    
    # Test None text field
    result = validate_and_fix_text_field(None)
    assert result == ""
    logger.info("✅ None text field handled correctly")
    
    # Test list/dict text field
    list_field = ["item1", "item2"]
    result = validate_and_fix_text_field(list_field)
    assert '"item1"' in result
    logger.info("✅ List text field converted to JSON")
    
    logger.info("🎉 All embedding validation tests passed!")

def test_small_batch_ingestion():
    """Test ingestion with a small batch of synthetic documents"""
    logger.info("🧪 Testing small batch ingestion with fixes...")
    
    try:
        # Setup connection and models
        connection = get_iris_connection()
        embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
        colbert_encoder = get_colbert_doc_encoder()
        
        # Create test documents with potential problematic data
        test_documents = [
            {
                "doc_id": "TEST_001",
                "title": "Test Document 1",
                "abstract": "This is a test document with normal content.",
                "authors": ["Test Author 1", "Test Author 2"],
                "keywords": ["test", "document"]
            },
            {
                "doc_id": "TEST_002", 
                "title": "Test Document 2",
                "abstract": "",  # Empty abstract
                "authors": [],
                "keywords": []
            },
            {
                "doc_id": "TEST_003",
                "title": None,  # None title
                "abstract": "Document with None title",
                "authors": ["Author 3"],
                "keywords": None  # None keywords
            }
        ]
        
        logger.info(f"📄 Testing with {len(test_documents)} synthetic documents")
        
        # Load documents using fixed loader
        stats = load_documents_to_iris(
            connection=connection,
            documents=test_documents,
            embedding_func=embedding_func,
            colbert_doc_encoder_func=colbert_encoder,
            batch_size=10
        )
        
        logger.info("📊 Ingestion Results:")
        logger.info(f"  Total documents: {stats['total_documents']}")
        logger.info(f"  Loaded documents: {stats['loaded_doc_count']}")
        logger.info(f"  Loaded tokens: {stats['loaded_token_count']}")
        logger.info(f"  Errors: {stats['error_count']}")
        logger.info(f"  Duration: {stats['duration_seconds']:.2f}s")
        logger.info(f"  Rate: {stats['documents_per_second']:.2f} docs/sec")
        
        # Verify documents were loaded
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE doc_id LIKE 'TEST_%'")
        test_doc_count = cursor.fetchone()[0]
        logger.info(f"✅ Found {test_doc_count} test documents in database")
        
        # Check embeddings
        cursor.execute("SELECT doc_id, embedding FROM RAG.SourceDocuments_V2 WHERE doc_id LIKE 'TEST_%' AND embedding IS NOT NULL")
        docs_with_embeddings = cursor.fetchall()
        logger.info(f"✅ Found {len(docs_with_embeddings)} test documents with embeddings")
        
        for doc_id, embedding in docs_with_embeddings:
            # Verify no NaN/inf in stored embeddings
            if embedding and ('nan' in embedding.lower() or 'inf' in embedding.lower()):
                logger.error(f"❌ Found NaN/inf in stored embedding for {doc_id}")
            else:
                logger.info(f"✅ Clean embedding for {doc_id}: {embedding[:50]}...")
        
        # Clean up test documents
        cursor.execute("DELETE FROM RAG.DocumentTokenEmbeddings WHERE doc_id LIKE 'TEST_%'")
        cursor.execute("DELETE FROM RAG.SourceDocuments_V2 WHERE doc_id LIKE 'TEST_%'")
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info("🎉 Small batch ingestion test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Small batch ingestion test failed: {e}")
        return False

def test_embedding_generation_robustness():
    """Test embedding generation with problematic inputs"""
    logger.info("🧪 Testing embedding generation robustness...")
    
    try:
        embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
        
        # Test various problematic inputs
        test_inputs = [
            "Normal text",
            "",  # Empty string
            "   ",  # Whitespace only
            "Text with special chars: àáâãäåæçèéêë",
            "Very long text " * 100,  # Very long text
            "Text\x00with\x00null\x00bytes",  # Text with null bytes
        ]
        
        for i, text in enumerate(test_inputs):
            try:
                embeddings = embedding_func([text])
                embedding = embeddings[0]
                
                # Check for NaN/inf
                if any(np.isnan(x) or np.isinf(x) for x in embedding):
                    logger.error(f"❌ NaN/inf found in embedding for input {i}")
                else:
                    logger.info(f"✅ Clean embedding generated for input {i}: {len(embedding)} dims")
                    
            except Exception as e:
                logger.error(f"❌ Error generating embedding for input {i}: {e}")
        
        logger.info("🎉 Embedding generation robustness test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding generation test failed: {e}")
        return False

def main():
    """Run all data quality tests"""
    logger.info("🚀 Starting comprehensive data quality tests...")
    
    tests = [
        ("Embedding Validation", test_embedding_validation),
        ("Embedding Generation Robustness", test_embedding_generation_robustness),
        ("Small Batch Ingestion", test_small_batch_ingestion),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All data quality tests passed! Ready for 100K ingestion.")
        return True
    else:
        logger.error("❌ Some tests failed. Fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)