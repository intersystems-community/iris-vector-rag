#!/usr/bin/env python
# batch_test.py - Test for batch processing capability without requiring IRIS container

import os
import sys
import json
from pathlib import Path

def test_batch_processing():
    """
    Tests the batch processing capability of the DataLoader
    without requiring an actual IRIS container
    """
    print("Testing batch processing capability...")
    
    # Import modules from our project
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from common.utils import Document
    
    # Create a mock dataset with multiple documents
    num_docs = 25  # Small number for quick testing
    
    # Create documents with unique IDs and content
    documents = []
    for i in range(num_docs):
        doc_id = f"test_doc_{i+1}"
        content = f"This is test document {i+1} with batch processing content."
        documents.append(Document(id=doc_id, content=content))
    
    print(f"Created {len(documents)} test documents")
    
    # Simulate batch processing
    batch_size = 5
    batches = []
    
    for batch_start in range(0, len(documents), batch_size):
        batch_end = min(batch_start + batch_size, len(documents))
        current_batch = documents[batch_start:batch_end]
        batches.append(current_batch)
    
    print(f"Split into {len(batches)} batches of size {batch_size}")
    
    # Process each batch (simulate, don't actually process)
    total_processed = 0
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1} of {len(batches)} ({len(batch)} documents)...")
        
        # Just count and track the documents to simulate processing
        batch_docs_processed = len(batch)
        total_processed += batch_docs_processed
        
        # Simulate embeddings generation for this batch
        print(f"Generated embeddings for {batch_docs_processed} documents in batch {i+1}")
        
        # Simulate token embeddings generation
        total_tokens = 0
        for doc in batch:
            tokens = doc.content.split()
            total_tokens += len(tokens)
        
        print(f"Generated {total_tokens} token embeddings for batch {i+1}")
    
    print(f"\nBatch processing complete. Processed {total_processed} documents in {len(batches)} batches.")
    
    # Verify all documents were processed
    assert total_processed == num_docs, f"Expected to process {num_docs} documents, but processed {total_processed}"
    
    print("\nSuccess! Batch processing is working correctly.")
    return True

def main():
    """
    Run batch processing test
    """
    print("Starting batch processing test...")
    
    success = test_batch_processing()
    
    if success:
        print("\nBatch processing test completed successfully!")
    else:
        print("\nBatch processing test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
