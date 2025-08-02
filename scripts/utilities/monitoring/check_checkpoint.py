#!/usr/bin/env python3
"""
Check ingestion checkpoint status
"""

import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class IngestionCheckpoint:
    """Checkpoint data for resuming ingestion"""
    target_docs: int
    current_docs: int
    processed_files: List[str]
    failed_files: List[Dict[str, Any]]
    start_time: float
    last_checkpoint_time: float
    total_ingestion_time: float
    error_count: int
    batch_count: int
    schema_type: str  # 'RAG' or 'RAG_HNSW'

def main():
    checkpoint_file = "ingestion_checkpoint.pkl"
    
    if not os.path.exists(checkpoint_file):
        print("No checkpoint file found")
        return
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print("=== Ingestion Checkpoint Status ===")
        print(f"Target documents: {checkpoint.target_docs:,}")
        print(f"Current documents processed: {checkpoint.current_docs:,}")
        print(f"Progress: {(checkpoint.current_docs / checkpoint.target_docs * 100):.1f}%")
        print(f"Schema type: {checkpoint.schema_type}")
        print(f"Batch count: {checkpoint.batch_count}")
        print(f"Error count: {checkpoint.error_count}")
        print(f"Failed files: {len(checkpoint.failed_files)}")
        print(f"Total ingestion time: {checkpoint.total_ingestion_time:.2f} seconds")
        
        if checkpoint.failed_files:
            print("\nFailed files:")
            for failed in checkpoint.failed_files[:5]:  # Show first 5
                print(f"  - {failed.get('file', 'unknown')}: {failed.get('error', 'unknown error')}")
            if len(checkpoint.failed_files) > 5:
                print(f"  ... and {len(checkpoint.failed_files) - 5} more")
                
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    main()