#!/usr/bin/env python3
"""
Debug script to understand what the ColBERT encoder is actually returning.
"""

import sys
import os
sys.path.insert(0, '.')

# Try to import the real ColBERT encoder
try:
    archive_colbert_path = os.path.join(os.path.dirname(__file__), 'archive', 'colbert')
    if archive_colbert_path not in sys.path:
        sys.path.insert(0, archive_colbert_path)
    from doc_encoder import ColBERTDocEncoder
    REAL_COLBERT_AVAILABLE = True
    print("Real ColBERT encoder available")
except ImportError as e:
    print(f"Real ColBERT encoder not available: {e}")
    REAL_COLBERT_AVAILABLE = False

def test_colbert_encoding():
    """Test what the ColBERT encoder actually returns."""
    
    if not REAL_COLBERT_AVAILABLE:
        print("Cannot test - ColBERT encoder not available")
        return
    
    try:
        # Initialize the encoder
        print("Initializing ColBERT encoder...")
        encoder = ColBERTDocEncoder()
        
        # Test with a simple document
        test_doc = "This is a simple test document for ColBERT encoding."
        print(f"Test document: {test_doc}")
        
        # Encode the document
        print("Encoding document...")
        encoder_output = encoder(test_doc)
        
        print(f"Encoder output type: {type(encoder_output)}")
        
        # Handle different encoder output formats
        if isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            tokens, token_embeddings = encoder_output
            print(f"Tuple format - Tokens type: {type(tokens)}, count: {len(tokens)}")
            print(f"Tuple format - Token embeddings type: {type(token_embeddings)}, count: {len(token_embeddings)}")
            
            # Convert to list of (token, embedding) pairs for examination
            token_data = list(zip(tokens, token_embeddings))
        elif isinstance(encoder_output, list):
            print(f"List format - Length: {len(encoder_output)}")
            print(f"First element type: {type(encoder_output[0]) if encoder_output else 'Empty list'}")
            
            # Check if it's a list of embeddings or a list of (token, embedding) pairs
            if encoder_output and isinstance(encoder_output[0], (list, tuple)) and len(encoder_output[0]) == 2:
                # List of (token, embedding) pairs
                token_data = encoder_output
                print("Detected list of (token, embedding) pairs")
            else:
                # Assume it's just embeddings without tokens
                print("Detected list of embeddings without tokens")
                # Create dummy tokens
                token_data = [(f"token_{i}", emb) for i, emb in enumerate(encoder_output)]
        else:
            print(f"Unexpected encoder output format: {type(encoder_output)}")
            print(f"Output content: {encoder_output}")
            return
        
        print(f"Number of token pairs: {len(token_data)}")
        
        # Examine the first few tokens
        for i, (token_text, token_embedding) in enumerate(token_data[:3]):
            print(f"\nToken {i}:")
            print(f"  Text: '{token_text}'")
            print(f"  Embedding type: {type(token_embedding)}")
            print(f"  Embedding shape: {getattr(token_embedding, 'shape', 'No shape attribute')}")
            print(f"  Embedding dtype: {getattr(token_embedding, 'dtype', 'No dtype attribute')}")
            
            # Check if it's a tensor or array
            if hasattr(token_embedding, 'numpy'):
                print(f"  Has numpy() method: True")
                numpy_version = token_embedding.numpy()
                print(f"  Numpy version type: {type(numpy_version)}")
                print(f"  Numpy version shape: {numpy_version.shape}")
                print(f"  First 5 values: {numpy_version[:5]}")
            elif hasattr(token_embedding, '__array__'):
                print(f"  Has __array__ method: True")
                array_version = token_embedding.__array__()
                print(f"  Array version type: {type(array_version)}")
                print(f"  First 5 values: {array_version[:5]}")
            else:
                print(f"  Raw embedding (first 5): {token_embedding[:5] if hasattr(token_embedding, '__getitem__') else 'Cannot slice'}")
                
            # Try to convert using our vector utilities
            try:
                from common.vector_format_fix import format_vector_for_iris, create_iris_vector_string
                formatted = format_vector_for_iris(token_embedding)
                vector_str = create_iris_vector_string(formatted)
                print(f"  Converted to string: {vector_str[:50]}...")
                
                if '@$vector' in vector_str:
                    print(f"  ERROR: Contains '@$vector' hash!")
                else:
                    print(f"  SUCCESS: Proper numeric string")
                    
            except Exception as e:
                print(f"  ERROR in conversion: {e}")
                
    except Exception as e:
        print(f"ERROR in ColBERT encoding test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_colbert_encoding()