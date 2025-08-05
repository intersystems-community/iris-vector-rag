#!/usr/bin/env python3
"""
Load 50k unique PMC documents from the already downloaded collection
"""

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
from data.pmc_processor import process_pmc_files
from data.loader_fixed import load_documents_to_iris
import os
import time

def load_50k_pmc_documents():
    """Load 50k unique PMC documents"""
    print("=== Loading 50K Unique PMC Documents ===\n")
    
    # Initialize
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Process PMC files
    pmc_dir = 'data/pmc_100k_downloaded'
    print(f"Processing PMC files from {pmc_dir}")
    
    start_time = time.time()
    
    # Process documents with limit
    documents = []
    doc_count = 0
    target_count = 50000
    
    for doc in process_pmc_files(pmc_dir):
        documents.append(doc)
        doc_count += 1
        
        # Load in batches of 1000
        if len(documents) >= 1000:
            print(f"\nLoading batch: {doc_count - 1000} to {doc_count}")
            stats = load_documents_to_iris(
                iris,
                documents,
                embedding_func=embedding_func,
                batch_size=250
            )
            # Check what keys are in stats
            if stats:
                loaded_count = stats.get('loaded_count', stats.get('loaded', len(documents)))
                print(f"Batch loaded: {loaded_count} documents")
            documents = []
        
        # Stop at target
        if doc_count >= target_count:
            break
        
        # Progress update
        if doc_count % 5000 == 0:
            elapsed = time.time() - start_time
            rate = doc_count / elapsed
            eta = (target_count - doc_count) / rate
            print(f"\nProgress: {doc_count:,}/{target_count:,} documents "
                  f"({doc_count/target_count*100:.1f}%) - "
                  f"Rate: {rate:.0f} docs/sec - ETA: {eta/60:.1f} min")
    
    # Load remaining documents
    if documents:
        print(f"\nLoading final batch: {len(documents)} documents")
        stats = load_documents_to_iris(
            iris,
            documents,
            embedding_func=embedding_func,
            batch_size=250
        )
        if stats:
            loaded_count = stats.get('loaded_count', stats.get('loaded', len(documents)))
            print(f"Final batch loaded: {loaded_count} documents")
    
    # Final stats
    elapsed = time.time() - start_time
    cursor = iris.cursor()
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    final_count = cursor.fetchone()[0]
    cursor.close()
    
    print(f"\n=== Loading Complete ===")
    print(f"Documents loaded: {doc_count:,}")
    print(f"Total documents in database: {final_count:,}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Average rate: {doc_count/elapsed:.0f} docs/sec")
    
    iris.close()

if __name__ == "__main__":
    load_50k_pmc_documents()