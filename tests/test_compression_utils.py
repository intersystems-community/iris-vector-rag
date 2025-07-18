# tests/test_compression_utils.py
# Tests for vector compression utilities used in ColBERT

import pytest
import numpy as np
import sys
import os

# Add project root to path so we can import common modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.compression_utils import (
    compress_vector, 
    decompress_vector, 
    calculate_compression_ratio,
    compress_token_embeddings,
    decompress_token_embeddings
)

def test_compression_ratio_requirements():
    """
    Test that the compression ratio meets the requirement: ratio ≤ 2×
    """
    print("\nTest: test_compression_ratio_requirements")
    
    # Create test vectors of different dimensions to simulate embeddings
    test_vectors = [
        np.random.randn(384).tolist(),  # Typical document embedding size
        np.random.randn(768).tolist(),  # BERT base size
        np.random.randn(128).tolist(),  # Smaller embedding
    ]
    
    bit_depths = [4, 8, 16]
    
    for vector in test_vectors:
        print(f"Testing vector of dimension {len(vector)}")
        
        for bits in bit_depths:
            # Compress the vector
            compressed, scale = compress_vector(vector, bits=bits)
            
            # Calculate compression ratio
            ratio = calculate_compression_ratio(vector, compressed, bits)
            
            print(f"  {bits}-bit compression: ratio = {ratio:.2f}x")
            
            # Check that we meet the requirement: ratio ≤ 2×
            # For 4-bit compression, the ratio should be at least 4x (32/4 = 8, but with overhead maybe 4x)
            # For 8-bit compression, the ratio should be at least 2x (32/8 = 4, but with overhead maybe 2x)
            if bits == 4:
                assert ratio >= 4.0, f"4-bit compression ratio ({ratio}) less than 4x"
            elif bits == 8:
                assert ratio >= 2.0, f"8-bit compression ratio ({ratio}) less than 2x"
            else:  # 16-bit
                assert ratio >= 1.0, f"16-bit compression ratio ({ratio}) less than 1x"
    
    print("Compression ratio requirements test passed")

def test_compression_basic_functionality():
    """
    Test basic compression and decompression functionality
    """
    print("\nTest: test_compression_basic_functionality")
    
    # Simple test with a predictable vector
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Just test 4-bit compression since that's what we'll use for ColBERT
    bits = 4
    compressed, scale = compress_vector(test_vector, bits=bits)
    
    # Verify non-empty result
    assert len(compressed) > 0, "Compressed vector should not be empty"
    assert scale > 0, "Scale factor should be positive"
    
    # Decompress and check length
    decompressed = decompress_vector(compressed, scale, bits=bits, vector_dim=len(test_vector))
    assert len(decompressed) == len(test_vector), "Decompressed length should match original"
    
    # Verify basic similarity - direction should be preserved even with low bit precision
    # Calculate cosine similarity between original and decompressed
    orig_norm = np.linalg.norm(test_vector)
    decomp_norm = np.linalg.norm(decompressed)
    
    # Avoid division by zero
    if orig_norm > 0 and decomp_norm > 0:
        cosine_sim = np.dot(test_vector, decompressed) / (orig_norm * decomp_norm)
        print(f"Cosine similarity: {cosine_sim:.6f}")
        assert cosine_sim > 0.9, f"Cosine similarity ({cosine_sim}) too low"
    
    print("Compression basic functionality test passed")
    
    print("Compression-decompression accuracy test passed")

def test_batch_compression():
    """
    Test compression of multiple token embeddings in a batch
    """
    print("\nTest: test_batch_compression")
    
    # Create a batch of token embeddings
    token_embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 10,  # 50-dim
        [-0.1, -0.2, -0.3, -0.4, -0.5] * 10,
        [0.5, 0.4, 0.3, 0.2, 0.1] * 10,
    ]
    
    # Test 4-bit compression
    compressed_batch = compress_token_embeddings(token_embeddings, bits=4)
    
    # Check structure
    assert len(compressed_batch) == len(token_embeddings), "Compressed batch length mismatch"
    
    # Each item should be a tuple of (compressed_vector, scale_factor)
    for comp_item in compressed_batch:
        assert isinstance(comp_item, tuple), "Compressed item should be a tuple"
        assert len(comp_item) == 2, "Compressed tuple should have 2 elements"
        
        compressed_vector, scale_factor = comp_item
        assert isinstance(compressed_vector, list), "Compressed vector should be a list"
        assert isinstance(scale_factor, float), "Scale factor should be a float"
    
    # Test decompression
    decompressed_batch = decompress_token_embeddings(
        compressed_batch, 
        bits=4, 
        vector_dim=len(token_embeddings[0])
    )
    
    # Check decompressed batch
    assert len(decompressed_batch) == len(token_embeddings), "Decompressed batch length mismatch"
    
    # Verify each decompressed vector has the right dimension
    for i, decompressed in enumerate(decompressed_batch):
        assert len(decompressed) == len(token_embeddings[i]), f"Decompressed vector {i} dimension mismatch"
    
    print("Batch compression test passed")

def test_normalized_vectors():
    """
    Test compression of normalized vectors (unit length)
    """
    print("\nTest: test_normalized_vectors")
    
    # Create a normalized vector
    vector = np.random.randn(100)
    vector = vector / np.linalg.norm(vector)
    vector = vector.tolist()
    
    # Compress with normalization
    compressed, scale = compress_vector(vector, bits=4, normalize=True)
    
    # Decompress
    decompressed = decompress_vector(compressed, scale, bits=4, vector_dim=len(vector))
    
    # Check that the decompressed vector is still approximately normalized
    decompressed_norm = np.linalg.norm(decompressed)
    print(f"Decompressed vector norm: {decompressed_norm:.6f}")
    
    # Should be close to 1.0 if normalization was preserved
    assert 0.9 <= decompressed_norm <= 1.1, f"Decompressed norm ({decompressed_norm}) far from 1.0"
    
    print("Normalized vectors test passed")

if __name__ == "__main__":
    # Run tests directly when script is executed
    test_compression_ratio_requirements()
    test_compression_basic_functionality()
    test_batch_compression()
    test_normalized_vectors()
    print("\nAll compression utility tests passed!")
