#!/usr/bin/env python3
"""
Test Script for Optimized Ingestion Performance

This script tests the optimized loader with a small batch to verify
performance improvements before running the full ingestion.
"""

import logging
import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.loader_optimized_performance import process_and_load_documents_optimized
from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

# Import ColBERT encoder from centralized utils
from common.utils import get_colbert_doc_encoder_func

def setup_test_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def test_optimized_performance():
    """Test the optimized loader with a small batch."""
    logger = setup_test_logging()
    
    logger.info("🧪 TESTING OPTIMIZED INGESTION PERFORMANCE")
    logger.info("=" * 60)
    
    # Test parameters
    test_limit = 100  # Small test batch
    test_batch_size = 10  # Very small batches for testing
    test_token_batch_size = 100
    
    logger.info(f"📊 Test Parameters:")
    logger.info(f"   Document limit: {test_limit}")
    logger.info(f"   Document batch size: {test_batch_size}")
    logger.info(f"   Token batch size: {test_token_batch_size}")
    
    try:
        # Check database connection
        conn = get_iris_connection()
        if not conn:
            logger.error("❌ Failed to establish database connection")
            return False
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        initial_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        logger.info(f"📊 Initial document count: {initial_count}")
        
        # Set up data directory
        pmc_directory = "data/pmc_100k_downloaded"
        if not os.path.exists(pmc_directory):
            logger.error(f"❌ PMC data directory not found: {pmc_directory}")
            return False
        
        # Initialize models
        logger.info("🔧 Initializing models...")
        
        # Get embedding model
        embedding_model = get_embedding_model()
        logger.info("✅ Embedding model initialized")
        
        def embedding_func(texts):
            return embedding_model.encode(texts).tolist()
        
        # Initialize ColBERT encoder
        try:
            colbert_encoder = ColBERTDocEncoder(mock=False)
            logger.info("✅ ColBERT document encoder initialized")
            
            def colbert_doc_encoder_func(document_text):
                return colbert_encoder.encode(document_text)
                
        except Exception as e:
            logger.warning(f"⚠️  ColBERT encoder failed, using mock: {e}")
            colbert_doc_encoder_func = None
        
        # Run the test
        logger.info("🚀 Starting optimized ingestion test...")
        start_time = time.time()
        
        result = process_and_load_documents_optimized(
            pmc_directory=pmc_directory,
            embedding_func=embedding_func,
            colbert_doc_encoder_func=colbert_doc_encoder_func,
            limit=test_limit,
            batch_size=test_batch_size,
            token_batch_size=test_token_batch_size,
            use_mock=False
        )
        
        test_duration = time.time() - start_time
        
        # Analyze results
        logger.info("📊 TEST RESULTS:")
        logger.info("=" * 40)
        
        if result.get("success"):
            processed = result.get('processed_count', 0)
            loaded_docs = result.get('loaded_doc_count', 0)
            loaded_tokens = result.get('loaded_token_count', 0)
            errors = result.get('error_count', 0)
            rate = result.get('documents_per_second', 0)
            
            logger.info(f"✅ SUCCESS!")
            logger.info(f"   Processed: {processed} documents")
            logger.info(f"   Loaded: {loaded_docs} documents")
            logger.info(f"   Token embeddings: {loaded_tokens}")
            logger.info(f"   Errors: {errors}")
            logger.info(f"   Duration: {test_duration:.2f} seconds")
            logger.info(f"   Rate: {rate:.2f} docs/sec")
            
            # Performance assessment
            if rate >= 10.0:
                logger.info("🎉 EXCELLENT PERFORMANCE: Rate >= 10 docs/sec")
                performance_status = "EXCELLENT"
            elif rate >= 5.0:
                logger.info("✅ GOOD PERFORMANCE: Rate >= 5 docs/sec")
                performance_status = "GOOD"
            elif rate >= 2.0:
                logger.info("⚠️  ACCEPTABLE PERFORMANCE: Rate >= 2 docs/sec")
                performance_status = "ACCEPTABLE"
            else:
                logger.warning("❌ POOR PERFORMANCE: Rate < 2 docs/sec")
                performance_status = "POOR"
            
            # Batch time analysis
            batch_times = result.get('batch_times', [])
            if batch_times:
                avg_batch_time = sum(batch_times) / len(batch_times)
                max_batch_time = max(batch_times)
                
                logger.info(f"📈 Batch Performance:")
                logger.info(f"   Average batch time: {avg_batch_time:.1f}s")
                logger.info(f"   Maximum batch time: {max_batch_time:.1f}s")
                
                if max_batch_time > 30.0:
                    logger.warning("⚠️  Some batches exceeded 30s threshold")
                else:
                    logger.info("✅ All batches completed within acceptable time")
            
            # Performance degradation check
            if result.get('performance_degraded', False):
                logger.warning("⚠️  PERFORMANCE DEGRADATION detected during test")
                performance_status = "DEGRADED"
            
            # Final recommendation
            logger.info("🎯 RECOMMENDATION:")
            if performance_status in ["EXCELLENT", "GOOD"]:
                logger.info("✅ PROCEED with full optimized ingestion")
                logger.info(f"   Recommended batch size: {test_batch_size}")
                logger.info(f"   Recommended token batch size: {test_token_batch_size}")
                return True
            elif performance_status == "ACCEPTABLE":
                logger.info("⚠️  PROCEED with CAUTION - consider smaller batches")
                logger.info(f"   Recommended batch size: {max(5, test_batch_size // 2)}")
                logger.info(f"   Recommended token batch size: {test_token_batch_size // 2}")
                return True
            else:
                logger.warning("❌ DO NOT PROCEED - investigate performance issues")
                logger.warning("   Consider further optimization or database tuning")
                return False
                
        else:
            logger.error(f"❌ TEST FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        return False

def main():
    """Main test function."""
    logger = setup_test_logging()
    
    logger.info(f"🧪 Starting optimized ingestion performance test at {datetime.now()}")
    
    success = test_optimized_performance()
    
    if success:
        logger.info("🎉 Test completed successfully - ready for full ingestion")
        sys.exit(0)
    else:
        logger.error("💥 Test failed - do not proceed with full ingestion")
        sys.exit(1)

if __name__ == "__main__":
    main()