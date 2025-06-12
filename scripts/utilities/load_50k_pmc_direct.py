#!/usr/bin/env python3
"""
Load 50k PMC documents directly, bypassing the 1000 limit
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
from data.pmc_processor import process_pmc_files
import time
import logging

logger = logging.getLogger(__name__) # Assuming logger is set up elsewhere or add basicConfig

def load_pmc_documents_to_target(target_total_documents=50000, pmc_source_dir='data/pmc_100k_downloaded'):
    """Load unique PMC documents up to a target total count"""
    print(f"=== Loading PMC Documents to Target: {target_total_documents:,} (Source: {pmc_source_dir}) ===\n")
    
    # Initialize
    iris = get_iris_connection()
    cursor = iris.cursor()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Get existing document count (using doc_id as confirmed earlier)
    cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.SourceDocuments WHERE doc_id IS NOT NULL AND doc_id <> ''")
    existing_unique_count = cursor.fetchone()[0]
    logger.info(f"Starting with {existing_unique_count:,} existing unique documents (doc_id based).")
    
    if existing_unique_count >= target_total_documents:
        logger.info(f"Already have {existing_unique_count:,} unique documents. Target of {target_total_documents:,} reached or exceeded.")
        cursor.close()
        iris.close()
        return True # Indicate success as target is met
    
    print(f"Processing PMC files from {pmc_source_dir}")
    
    start_time = time.time()
    
    docs_to_process_limit = target_total_documents - existing_unique_count
    # Add a buffer to account for potential duplicates not caught by simple count,
    # or if process_pmc_files yields already existing doc_ids that are skipped on insert.
    # The process_pmc_files limit is on files yielded, not necessarily new unique inserts.
    processing_limit = docs_to_process_limit + max(1000, int(docs_to_process_limit * 0.1)) # 10% or 1000 buffer
    
    logger.info(f"Need to load approximately {docs_to_process_limit:,} more unique documents. Will process up to {processing_limit:,} files.")
    
    new_docs_inserted_count = 0
    
    for doc in process_pmc_files(pmc_source_dir, limit=processing_limit):
        # Insert document with embedding
        doc_content = doc['content']
        doc_embedding = embedding_func([doc_content])[0]
        doc_embedding_str = ','.join([f'{x:.10f}' for x in doc_embedding])
        
        # Convert authors list to string
        authors_str = str(doc.get('authors', []))
        keywords_str = str(doc.get('keywords', []))
        
        try:
            cursor.execute("""
                INSERT INTO RAG.SourceDocuments 
                (doc_id, title, text_content, authors, keywords, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                doc['doc_id'],
                doc['title'],
                doc_content,
                authors_str,
                keywords_str,
                doc_embedding_str
            ])
            new_docs_inserted_count += 1 # Corrected variable
            
            # Commit every 100 documents
            if new_docs_inserted_count > 0 and new_docs_inserted_count % 100 == 0:
                iris.commit()
                logger.info(f"Committed batch. Total new documents inserted so far: {new_docs_inserted_count}")
                
            # Progress update
            if new_docs_inserted_count > 0 and new_docs_inserted_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = new_docs_inserted_count / elapsed if elapsed > 0 else 0
                current_total_unique = existing_unique_count + new_docs_inserted_count # Approximate
                eta_seconds = (target_total_documents - current_total_unique) / rate if rate > 0 and current_total_unique < target_total_documents else 0
                logger.info(f"\nProgress: Approx {current_total_unique:,}/{target_total_documents:,} total unique documents.")
                logger.info(f"New documents inserted in this run: {new_docs_inserted_count:,}")
                logger.info(f"Rate: {rate:.1f} docs/sec - Approx ETA: {eta_seconds/60:.1f} min")
            
            # Check if target is reached based on new inserts
            # More accurate check would be to re-query COUNT(DISTINCT doc_id) periodically, but that's slower.
            if (existing_unique_count + new_docs_inserted_count) >= target_total_documents:
                logger.info(f"Target of {target_total_documents:,} likely reached based on inserted count. Stopping.")
                break
                
        except Exception as e: # Catching DB insert errors (like unique constraint)
            if "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
                logger.debug(f"Skipped duplicate document: {doc.get('doc_id', 'N/A')}")
            else:
                logger.warning(f"Error inserting document {doc.get('doc_id', 'N/A')}: {e}")
    
    # Final commit
    iris.commit()
    logger.info("Final commit done.")
    
    # Final stats
    elapsed = time.time() - start_time
    cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.SourceDocuments WHERE doc_id IS NOT NULL AND doc_id <> ''")
    final_unique_count = cursor.fetchone()[0]
    
    logger.info(f"\n=== Loading Complete ===")
    logger.info(f"New documents effectively added in this run: {final_unique_count - existing_unique_count:,}")
    logger.info(f"Total unique documents in database: {final_unique_count:,}")
    logger.info(f"Target was: {target_total_documents:,}")
    logger.info(f"Time taken for this run: {elapsed/60:.1f} minutes")
    if new_docs_inserted_count > 0 and elapsed > 0:
        logger.info(f"Average insertion attempt rate for this run: {new_docs_inserted_count/elapsed:.1f} docs/sec")
    
    cursor.close()
    iris.close()
    return True

if __name__ == "__main__":
    # Example: Load up to 60,000 documents if run directly
    # For production scaling, scale_to_100k.py should be used.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    load_pmc_documents_to_target(target_total_documents=60000)