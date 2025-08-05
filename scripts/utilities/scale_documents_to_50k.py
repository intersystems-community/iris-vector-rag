#!/usr/bin/env python3
"""
Scale documents to 50k by duplicating existing documents
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
import uuid
import time

def scale_documents_to_target(target: int = 50000):
    """Scale documents to target count by duplicating existing ones"""
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    print(f"=== Scaling Documents to {target:,} ===\n")
    
    # Get current count
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    current_count = cursor.fetchone()[0]
    print(f"Current documents: {current_count:,}")
    
    if current_count >= target:
        print(f"Already have {current_count:,} documents, no scaling needed")
        return
    
    # Get existing documents to duplicate
    cursor.execute("""
        SELECT doc_id, title, text_content, embedding 
        FROM RAG.SourceDocuments 
        WHERE text_content IS NOT NULL
    """)
    existing_docs = cursor.fetchall()
    print(f"Found {len(existing_docs)} documents to use as templates")
    
    # Calculate how many times to duplicate
    docs_needed = target - current_count
    print(f"Need to create {docs_needed:,} more documents")
    
    # Start duplicating
    batch_size = 100
    created = 0
    start_time = time.time()
    
    while created < docs_needed:
        batch_docs = []
        
        for i in range(min(batch_size, docs_needed - created)):
            # Pick a random document to duplicate
            template = existing_docs[i % len(existing_docs)]
            doc_id, title, content, embedding_str = template
            
            # Create new document with unique ID
            new_doc_id = f"DUP_{uuid.uuid4().hex[:8]}_{doc_id}"
            new_title = f"{title} (Copy {created + i + 1})"
            
            batch_docs.append((new_doc_id, new_title, content, embedding_str))
        
        # Insert batch
        for doc in batch_docs:
            cursor.execute("""
                INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, embedding)
                VALUES (?, ?, ?, ?)
            """, doc)
        
        iris.commit()
        created += len(batch_docs)
        
        # Progress update
        if created % 1000 == 0:
            elapsed = time.time() - start_time
            rate = created / elapsed
            eta = (docs_needed - created) / rate
            print(f"Progress: {created:,}/{docs_needed:,} documents created "
                  f"({created/docs_needed*100:.1f}%) - "
                  f"Rate: {rate:.0f} docs/sec - ETA: {eta/60:.1f} min")
    
    # Final count
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    final_count = cursor.fetchone()[0]
    
    elapsed = time.time() - start_time
    print(f"\n=== Scaling Complete ===")
    print(f"Final document count: {final_count:,}")
    print(f"Documents created: {created:,}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Average rate: {created/elapsed:.0f} docs/sec")
    
    cursor.close()
    iris.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Scale documents to target count')
    parser.add_argument('--target', type=int, default=50000, 
                        help='Target number of documents (default: 50000)')
    args = parser.parse_args()
    
    scale_documents_to_target(args.target)

if __name__ == "__main__":
    main()