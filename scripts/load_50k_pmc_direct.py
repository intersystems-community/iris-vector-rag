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

def load_50k_pmc_documents():
    """Load 50k unique PMC documents"""
    print("=== Loading 50K Unique PMC Documents (Direct) ===\n")
    
    # Initialize
    iris = get_iris_connection()
    cursor = iris.cursor()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Get existing document count
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    existing_count = cursor.fetchone()[0]
    print(f"Starting with {existing_count:,} existing documents")
    
    if existing_count >= 50000:
        print("Already have 50k+ documents!")
        cursor.close()
        iris.close()
        return
    
    # Process PMC files with higher limit
    pmc_dir = 'data/pmc_100k_downloaded'
    print(f"Processing PMC files from {pmc_dir}")
    
    start_time = time.time()
    
    # Process documents - use limit of 60000 to ensure we get to 50k total
    docs_needed = 50000 - existing_count
    print(f"Need to load {docs_needed:,} more documents")
    
    doc_count = 0
    
    # Process with explicit higher limit
    for doc in process_pmc_files(pmc_dir, limit=docs_needed + 1000):  # Extra buffer
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
            
            # Commit every 100 documents
            if doc_count % 100 == 0:
                iris.commit()
                
            # Progress update
            if doc_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = doc_count / elapsed
                total_docs = existing_count + doc_count
                eta = (50000 - total_docs) / rate if total_docs < 50000 else 0
                print(f"\nProgress: {total_docs:,}/50,000 total documents "
                      f"({total_docs/50000*100:.1f}%) - "
                      f"New: {doc_count:,}")
                print(f"Rate: {rate:.0f} docs/sec - ETA: {eta/60:.1f} min")
            
            # Stop when we reach 50k total
            if existing_count + doc_count >= 50000:
                break
                
        except Exception as e:
            print(f"Error inserting {doc['doc_id']}: {e}")
    
    # Final commit
    iris.commit()
    
    # Final stats
    elapsed = time.time() - start_time
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    final_count = cursor.fetchone()[0]
    
    print(f"\n=== Loading Complete ===")
    print(f"New documents loaded: {doc_count:,}")
    print(f"Total documents in database: {final_count:,}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    if doc_count > 0:
        print(f"Average rate: {doc_count/elapsed:.0f} docs/sec")
    
    cursor.close()
    iris.close()

if __name__ == "__main__":
    load_50k_pmc_documents()