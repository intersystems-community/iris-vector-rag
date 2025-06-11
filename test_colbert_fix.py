#!/usr/bin/env python3
"""
Simple test script to verify ColBERT dimension fix.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from common.utils import get_colbert_query_encoder_func

def test_colbert_dimension_fix():
    """Test that ColBERT query encoder uses correct dimensions from config."""
    
    print("Testing ColBERT dimension fix...")
    
    # Get the ColBERT query encoder function
    encoder_func = get_colbert_query_encoder_func()
    
    # Test with a simple query
    test_query = "What is machine learning?"
    query_embeddings = encoder_func(test_query)
    
    print(f"Query: '{test_query}'")
    print(f"Number of token embeddings: {len(query_embeddings)}")
    
    if query_embeddings:
        embedding_dimension = len(query_embeddings[0])
        print(f"Embedding dimension: {embedding_dimension}")
        
        # Check if dimension matches expected 384
        if embedding_dimension == 384:
            print("✅ SUCCESS: ColBERT query encoder is using correct 384-dimensional embeddings!")
            return True
        else:
            print(f"❌ FAILURE: Expected 384 dimensions, got {embedding_dimension}")
            return False
    else:
        print("❌ FAILURE: No embeddings generated")
        return False

if __name__ == "__main__":
    success = test_colbert_dimension_fix()
    sys.exit(0 if success else 1)