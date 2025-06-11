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
# Updated import to the refactored function
from scripts.load_50k_pmc_direct import load_pmc_documents_to_target

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
        
        # Check unique Document IDs
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.SourceDocuments WHERE doc_id IS NOT NULL AND doc_id <> ''")
        unique_count_result = cursor.fetchone()
        unique_count = unique_count_result[0] if unique_count_result else 0
        
        # Check GraphRAG data (handle if tables don't exist)
        entity_count = 0
        rel_count = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            entity_count_result = cursor.fetchone()
            entity_count = entity_count_result[0] if entity_count_result else 0
        except Exception:
            logger.warning("RAG.Entities table not found or error querying.")
            
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
            rel_count_result = cursor.fetchone()
            rel_count = rel_count_result[0] if rel_count_result else 0
        except Exception:
            logger.warning("RAG.Relationships table not found or error querying.")

        logger.info(f"Current database state:")
        logger.info(f"  Total rows in RAG.SourceDocuments: {doc_count:,}") # doc_count is from the first query
        logger.info(f"  Unique Document IDs: {unique_count:,}")
        logger.info(f"  GraphRAG entities: {entity_count:,}")
        logger.info(f"  GraphRAG relationships: {rel_count:,}")
        
        return doc_count, unique_count # Return total rows and unique doc_ids
        
    finally:
        if iris: # Check if iris connection was successfully established
            if cursor:
                cursor.close()
            iris.close()

def scale_to_target(target_doc_count: int, pmc_source_directory: str):
    """Scale the database to target document count using the refactored loader."""
    
    logger.info(f"🚀 Starting scale to {target_doc_count:,} documents using source: {pmc_source_directory}")
    
    # Initial check (optional here, as loader also checks, but good for pre-flight)
    _, initial_unique_docs = check_current_documents()
    logger.info(f"Initial unique document count: {initial_unique_docs:,}")

    if initial_unique_docs >= target_doc_count:
        logger.info(f"✅ Target of {target_doc_count:,} documents already met or exceeded ({initial_unique_docs:,} found).")
        return

    logger.info(f"Attempting to load documents up to {target_doc_count:,}...")
    
    try:
        start_time = time.time()
        
        # Call the refactored loading function
        # It will handle its own internal logic to reach the target
        success = load_pmc_documents_to_target(
            target_total_documents=target_doc_count,
            pmc_source_dir=pmc_source_directory
        )
        
        duration = time.time() - start_time
        
        if success:
            logger.info(f"\n🎉 Scaling process completed in {duration/60:.1f} minutes.")
        else:
            logger.warning("\n⚠️ Scaling process finished, but the loader reported an issue or did not confirm full success.")
            
        # Final check
        _, final_unique_docs = check_current_documents()
        logger.info(f"Final unique document count after scaling attempt: {final_unique_docs:,}")
        if final_unique_docs >= target_doc_count:
            logger.info(f"✅ Target of {target_doc_count:,} successfully reached.")
        else:
            logger.warning(f"⚠️ Target of {target_doc_count:,} not reached. Current count: {final_unique_docs:,}")
            
    except Exception as e:
        logger.error(f"❌ Error during scaling process: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scale RAG system to target document count')
    parser.add_argument('--target', type=int, default=100000,
                       help='Target number of documents (default: 100000)')
    parser.add_argument('--source-dir', type=str, default='data/pmc_100k_downloaded',
                       help='Directory containing the PMC XML files to process')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - scale to 60k instead of 100k')
    
    args = parser.parse_args()
    
    target_count = 60000 if args.test else args.target
    
    logger.info("="*60)
    logger.info(f"RAG System Scaling to {target_count:,} Documents from source: {args.source_dir}")
    logger.info("="*60)
    
    scale_to_target(target_doc_count=target_count, pmc_source_directory=args.source_dir)

if __name__ == "__main__":
    main()