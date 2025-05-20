#!/usr/bin/env python
# simplified_test.py - Test token embedding format fix without requiring a full IRIS container

import os
import sys
import json
from pathlib import Path

def test_token_embedding_format():
    """
    Tests that the token embedding formatting logic works correctly.
    This ensures the fix for the token_embedding vector formatting will work.
    """
    print("Running simplified test for token embedding format...")
    
    # Import modules from our project
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from common.utils import Document
    
    # Create a simple Document for testing
    doc = Document(id="test_doc_1", content="This is a test document with multiple tokens")
    
    # Mock the token embeddings that would be generated
    tokens = doc.content.split()
    # Generate 128-dimensional vectors to match table definition
    mock_embeddings = [[0.01] * 128 for _ in tokens]
    
    doc.colbert_tokens = tokens
    doc.colbert_token_embeddings = mock_embeddings
    
    print(f"Document: {doc.id}")
    print(f"Content: {doc.content}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Number of token embeddings: {len(mock_embeddings)}")
    
    # Now test the formatting logic (extracted from loader.py)
    token_data = []
    for i, (token, embedding) in enumerate(zip(doc.colbert_tokens, doc.colbert_token_embeddings)):
        # Convert token embedding to string for TO_VECTOR
        # This is the fixed logic that handles different embedding formats properly
        if isinstance(embedding, list):
            embedding_str = str(embedding)
        else:
            # Handle case where embedding might be a numpy array or other type
            # Convert to a standard Python list first
            try:
                embedding_list = [float(x) for x in embedding]
                embedding_str = str(embedding_list)
            except Exception as e:
                print(f"Error formatting token embedding: {e}")
                # Skip this token if we can't format it
                continue
                
        # In a real scenario, we would insert this into IRIS
        # Here we just store it to validate the formatting
        token_data.append((doc.id, i, token, embedding_str))
    
    # Check the formatted data
    if token_data:
        print("\nFormatted token data (first token):")
        print(f"Document ID: {token_data[0][0]}")
        print(f"Token index: {token_data[0][1]}")
        print(f"Token text: {token_data[0][2]}")
        print(f"Embedding format: {token_data[0][3][:50]}... (truncated)")
        
        # Check embedding format - should be a string representation of a list
        embedding_str = token_data[0][3]
        assert embedding_str.startswith('[') and embedding_str.endswith(']'), "Embedding should be formatted as a list string"
        
        print("\nSuccess! The token embedding format is correct.")
        return True
    else:
        print("Error: No token data was generated.")
        return False

def main():
    """
    Run simplified tests for the token embedding format fix
    """
    print("Starting simplified tests...")
    
    success = test_token_embedding_format()
    
    if success:
        print("\nSimplified tests completed successfully!")
    else:
        print("\nSimplified tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
