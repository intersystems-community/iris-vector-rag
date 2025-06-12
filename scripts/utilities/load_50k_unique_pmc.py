#!/usr/bin/env python3
"""
Load 50k unique PMC documents, skipping existing ones
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
from data.pmc_processor import process_pmc_files
import time

def load_50k_unique_documents():
    """Load 50k unique PMC documents"""
    print("=== Loading 50K Unique PMC Documents ===\n")
    
    # Initialize
    iris = get_iris_connection()
    cursor = iris.cursor()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Get existing document IDs
    cursor.execute("SELECT doc_id FROM RAG.SourceDocuments")
    existing_ids = set(row[0] for row in cursor.fetchall())
    print(f"Found {len(existing_ids):,} existing documents")
    
    # Process PMC files
    pmc_dir = 'data/pmc_100k_downloaded'
    print(f"Processing PMC files from {pmc_dir}")
    
    start_time = time.time()
    
    # Counters
    doc_count = 0
    skipped_count = 0
    target_count = 50000
    
    # Process documents
    for doc in process_pmc_files(pmc_dir):
        # Skip if already exists
        if doc['doc_id'] in existing_ids:
            skipped_count += 1
            continue
            
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
            doc_count += 1
            existing_ids.add(doc['doc_id'])
            
            # Commit every 100 documents
            if doc_count % 100 == 0:
                iris.commit()
                
            # Progress update
            if doc_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = doc_count / elapsed
                total_processed = len(existing_ids)
                eta = (target_count - total_processed) / rate if total_processed < target_count else 0
                print(f"\nProgress: {total_processed:,}/{target_count:,} total documents "
                      f"({total_processed/target_count*100:.1f}%) - "
                      f"New: {doc_count:,}, Skipped: {skipped_count:,}")
                print(f"Rate: {rate:.0f} docs/sec - ETA: {eta/60:.1f} min")
            
            # Stop when we reach target
            if len(existing_ids) >= target_count:
                break
                
        except Exception as e:
            if "UNIQUE" in str(e):
                skipped_count += 1
            else:
                print(f"Error inserting {doc['doc_id']}: {e}")
    
    # Final commit
    iris.commit()
    
    # Final stats
    elapsed = time.time() - start_time
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    final_count = cursor.fetchone()[0]
    
    print(f"\n=== Loading Complete ===")
    print(f"New documents loaded: {doc_count:,}")
    print(f"Documents skipped: {skipped_count:,}")
    print(f"Total documents in database: {final_count:,}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    if doc_count > 0:
        print(f"Average rate: {doc_count/elapsed:.0f} docs/sec")
    
    cursor.close()
    iris.close()

if __name__ == "__main__":
    load_50k_unique_documents()