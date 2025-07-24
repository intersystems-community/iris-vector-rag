#!/usr/bin/env python3
"""
Test Ingestion Performance Optimizations

Quick validation script to test performance improvements:
- Increased batch sizes (1000 vs 500)
- Better memory management
- Token embedding fixes
- Performance metrics comparison

Usage:
    python scripts/test_ingestion_optimizations.py --target-docs 15000
"""

import os
import sys
import logging
import time
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utilities.ingest_100k_documents import MassiveScaleIngestionPipeline
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_ingestion_optimizations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_current_document_count(schema_type: str = "RAG") -> int:
    """Get current document count from database"""
    try:
        connection = get_iris_connection()
        table_name = f"{schema_type}.SourceDocuments"
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        # Also check token embeddings
        cursor.execute(f"SELECT COUNT(*) FROM {schema_type}.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        logger.info(f"📊 Current status: {count:,} documents, {token_count:,} token embeddings")
        return count, token_count
    except Exception as e:
        logger.error(f"❌ Error getting document count: {e}")
        return 0, 0

def test_optimized_ingestion(target_docs: int = 15000, batch_size: int = 1000) -> dict:
    """Test optimized ingestion performance"""
    logger.info("=" * 80)
    logger.info("🧪 TESTING INGESTION OPTIMIZATIONS")
    logger.info("=" * 80)
    logger.info(f"🎯 Target: {target_docs:,} documents")
    logger.info(f"📦 Batch size: {batch_size}")
    
    # Get baseline counts
    start_docs, start_tokens = get_current_document_count()
    logger.info(f"📊 Starting with: {start_docs:,} docs, {start_tokens:,} tokens")
    
    # Run optimized ingestion
    pipeline = MassiveScaleIngestionPipeline(
        data_dir="data/pmc_100k_downloaded",
        checkpoint_interval=300  # 5 minutes
    )
    
    start_time = time.time()
    
    try:
        final_count = pipeline.ingest_to_target(
            target_docs=target_docs,
            batch_size=batch_size,
            resume=True,  # Resume from existing checkpoint
            schema_type="RAG"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get final counts
        final_docs, final_tokens = get_current_document_count()
        
        # Calculate metrics
        docs_processed = final_docs - start_docs
        tokens_processed = final_tokens - start_tokens
        docs_per_second = docs_processed / duration if duration > 0 else 0
        
        results = {
            "success": True,
            "target_docs": target_docs,
            "batch_size": batch_size,
            "start_docs": start_docs,
            "final_docs": final_docs,
            "docs_processed": docs_processed,
            "start_tokens": start_tokens,
            "final_tokens": final_tokens,
            "tokens_processed": tokens_processed,
            "duration_seconds": duration,
            "docs_per_second": docs_per_second,
            "tokens_per_doc": tokens_processed / docs_processed if docs_processed > 0 else 0,
            "timestamp": time.time()
        }
        
        logger.info("=" * 80)
        logger.info("📊 OPTIMIZATION TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"✅ Documents processed: {docs_processed:,}")
        logger.info(f"✅ Token embeddings: {tokens_processed:,}")
        logger.info(f"⏱️ Duration: {duration:.1f} seconds")
        logger.info(f"🚀 Rate: {docs_per_second:.2f} docs/second")
        logger.info(f"🔢 Tokens per doc: {results['tokens_per_doc']:.1f}")
        
        # Performance assessment
        if docs_per_second >= 3.0:
            logger.info("🎉 EXCELLENT: Performance target exceeded!")
        elif docs_per_second >= 2.5:
            logger.info("✅ GOOD: Performance improved from baseline")
        else:
            logger.info("⚠️ NEEDS WORK: Performance still below target")
            
        if tokens_processed > 0:
            logger.info("✅ FIXED: Token embeddings are being generated!")
        else:
            logger.error("❌ PROBLEM: Still no token embeddings generated")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": time.time() - start_time
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Ingestion Performance Optimizations")
    parser.add_argument("--target-docs", type=int, default=15000,
                       help="Target number of documents to test with")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size to test")
    
    args = parser.parse_args()
    
    # Run test
    results = test_optimized_ingestion(args.target_docs, args.batch_size)
    
    # Save results
    results_file = f"ingestion_optimization_test_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📄 Results saved: {results_file}")
    
    if results.get("success"):
        logger.info("🎉 Optimization test completed successfully!")
        return True
    else:
        logger.error("❌ Optimization test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)