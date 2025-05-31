#!/usr/bin/env python3
"""
Scale RAG System from 50k to 100k Documents
Downloads additional PMC documents and loads them into the database
"""

import sys
import os
import time
import logging
from datetime import datetime
sys.path.append('.')

from common.iris_connector import get_iris_connection
from scripts.load_50k_pmc_direct import download_and_load_pmc_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'scale_to_100k_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_current_documents():
    """Check how many documents are currently in the database"""
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    try:
        # Check SourceDocuments
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        
        # Check unique PMC IDs
        cursor.execute("SELECT COUNT(DISTINCT pmc_id) FROM RAG.SourceDocuments")
        unique_count = cursor.fetchone()[0]
        
        # Check GraphRAG data
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
        entity_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
        rel_count = cursor.fetchone()[0]
        
        logger.info(f"Current database state:")
        logger.info(f"  Total documents: {doc_count:,}")
        logger.info(f"  Unique PMC IDs: {unique_count:,}")
        logger.info(f"  GraphRAG entities: {entity_count:,}")
        logger.info(f"  GraphRAG relationships: {rel_count:,}")
        
        return doc_count, unique_count
        
    finally:
        cursor.close()
        iris.close()

def scale_to_target(target_count=100000):
    """Scale the database to target document count"""
    
    logger.info(f"🚀 Starting scale to {target_count:,} documents")
    
    # Check current state
    total_docs, unique_docs = check_current_documents()
    
    if unique_docs >= target_count:
        logger.info(f"✅ Already have {unique_docs:,} unique documents. Target reached!")
        return
    
    # Calculate how many more we need
    needed = target_count - unique_docs
    logger.info(f"📊 Need to download {needed:,} more unique documents")
    
    # Download and load additional documents
    logger.info("📥 Starting download of additional PMC documents...")
    
    try:
        # Use the existing loader with offset
        start_time = time.time()
        
        # We'll download in batches to avoid memory issues
        batch_size = 10000
        downloaded = 0
        
        while downloaded < needed:
            current_batch = min(batch_size, needed - downloaded)
            logger.info(f"\n📦 Processing batch: {downloaded+1} to {downloaded+current_batch}")
            
            # The loader will skip existing PMC IDs automatically
            success = download_and_load_pmc_documents(
                max_articles=current_batch,
                start_offset=unique_docs + downloaded
            )
            
            if not success:
                logger.warning("⚠️ Batch processing failed, retrying...")
                time.sleep(5)
                continue
                
            downloaded += current_batch
            
            # Check progress
            _, new_unique = check_current_documents()
            actual_added = new_unique - unique_docs
            logger.info(f"✅ Progress: {actual_added:,} new unique documents added")
            
            if new_unique >= target_count:
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final check
        final_total, final_unique = check_current_documents()
        
        logger.info(f"\n🎉 Scaling complete!")
        logger.info(f"📊 Final statistics:")
        logger.info(f"  Total documents: {final_total:,}")
        logger.info(f"  Unique PMC IDs: {final_unique:,}")
        logger.info(f"  Documents added: {final_unique - unique_docs:,}")
        logger.info(f"  Time taken: {duration/60:.1f} minutes")
        logger.info(f"  Rate: {(final_unique - unique_docs)/(duration/60):.0f} docs/minute")
        
    except Exception as e:
        logger.error(f"❌ Error during scaling: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scale RAG system to target document count')
    parser.add_argument('--target', type=int, default=100000, 
                       help='Target number of documents (default: 100000)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - scale to 60k instead of 100k')
    
    args = parser.parse_args()
    
    target = 60000 if args.test else args.target
    
    logger.info("="*60)
    logger.info(f"RAG System Scaling to {target:,} Documents")
    logger.info("="*60)
    
    scale_to_target(target)

if __name__ == "__main__":
    main()