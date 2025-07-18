# common/compression_utils.py
# Utility functions for compressing and decompressing vector embeddings

import numpy as np
from typing import List, Union, Tuple

def compress_vector(vector: List[float], bits: int = 8, normalize: bool = True) -> Tuple[List[int], float]:
    """
    Compress a floating-point vector to quantized integers with given bit precision.
    Returns the quantized values and the scale factor for reconstruction.
    
    Args:
        vector: List of floating-point values to compress
        bits: Number of bits for quantization (4, 8, or 16)
        normalize: Whether to normalize the vector before quantization
        
    Returns:
        Tuple of (quantized_values, scale_factor)
    """
    if not vector:
        return [], 1.0
    
    # Convert to numpy array for efficient operations
    vec_array = np.array(vector, dtype=np.float32)
    
    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(vec_array)
        if norm > 0:
            vec_array = vec_array / norm
    
    # Determine quantization parameters based on bit depth
    if bits == 4:
        max_val = 7  # -8 to 7 (4-bit signed)
        dtype = np.int8  # Store in int8 but only use 4 bits
    elif bits == 8:
        max_val = 127  # -128 to 127 (8-bit signed)
        dtype = np.int8
    elif bits == 16:
        max_val = 32767  # -32768 to 32767 (16-bit signed)
        dtype = np.int16
    else:
        raise ValueError(f"Unsupported bit depth: {bits}. Use 4, 8, or 16.")
    
    # Calculate scale factor for maximum range utilization
    abs_max = np.max(np.abs(vec_array))
    if abs_max == 0:
        scale_factor = 1.0  # Avoid division by zero
    else:
        scale_factor = float(abs_max / max_val)
    
    # Quantize to integers
    quantized = np.round(vec_array / scale_factor).astype(dtype)
    
    # For 4-bit, we store two values per byte
    if bits == 4:
        # Ensure values are within 4-bit range
        quantized = np.clip(quantized, -8, 7)
        
        # Prepare for packing: group by pairs and pack two 4-bit values into one byte
        if len(quantized) % 2 == 1:
            # Pad with zero if odd length
            quantized = np.append(quantized, 0)
        
        packed = []
        for i in range(0, len(quantized), 2):
            # Convert two 4-bit signed values to one byte
            # First value in upper 4 bits, second value in lower 4 bits
            high = (quantized[i] & 0x0F) << 4
            low = (quantized[i+1] & 0x0F)
            packed.append(high | low)
        
        return packed, scale_factor
    else:
        # For 8-bit and 16-bit, return as is
        return quantized.tolist(), scale_factor

def decompress_vector(compressed_data: Union[List[int], np.ndarray], scale_factor: float, 
                     bits: int = 8, vector_dim: int = None) -> List[float]:
    """
    Decompress a quantized vector back to floating-point values.
    
    Args:
        compressed_data: List of quantized integer values
        scale_factor: Scale factor used during compression
        bits: Number of bits used in quantization (4, 8, or 16)
        vector_dim: Original vector dimension (required for 4-bit compression)
        
    Returns:
        List of decompressed floating-point values
    """
    if not compressed_data:
        return []
    
    if bits == 4:
        if vector_dim is None:
            raise ValueError("vector_dim is required for 4-bit decompression")
        
        # Unpack the 4-bit values from bytes
        unpacked = []
        for byte in compressed_data:
            # Extract high 4 bits (first value)
            high = (byte >> 4) & 0x0F
            # Handle negative 4-bit signed values (two's complement)
            if high & 0x08:  # Check if sign bit is set
                high = high - 16
            unpacked.append(high)
            
            # Extract low 4 bits (second value)
            low = byte & 0x0F
            # Handle negative 4-bit signed values
            if low & 0x08:  # Check if sign bit is set
                low = low - 16
            unpacked.append(low)
        
        # Truncate to original dimension if needed
        unpacked = unpacked[:vector_dim]
        
        # Convert back to float and apply scale
        decompressed = [val * scale_factor for val in unpacked]
        
    else:
        # For 8-bit and 16-bit, simply multiply by scale factor
        decompressed = [val * scale_factor for val in compressed_data]
    
    return decompressed

def calculate_compression_ratio(original_vector: List[float], compressed_vector: List[int], 
                               bits: int = 8) -> float:
    """
    Calculate the compression ratio between original and compressed vectors.
    
    Args:
        original_vector: Original floating-point vector (usually 32-bit per value)
        compressed_vector: Compressed integer vector
        bits: Number of bits used for compression
        
    Returns:
        Compression ratio (original size / compressed size)
    """
    # Original size: 32 bits (4 bytes) per float
    original_size = len(original_vector) * 4
    
    # Compressed size depends on bit depth
    if bits == 4:
        # For 4-bit, we pack two values per byte
        compressed_size = (len(compressed_vector) * 1) + 4  # Add 4 bytes for scale factor
    elif bits == 8:
        compressed_size = (len(compressed_vector) * 1) + 4  # Add 4 bytes for scale factor
    elif bits == 16:
        compressed_size = (len(compressed_vector) * 2) + 4  # Add 4 bytes for scale factor
    else:
        raise ValueError(f"Unsupported bit depth: {bits}")
    
    # Return compression ratio (higher is better)
    return original_size / compressed_size

def compress_token_embeddings(token_embeddings: List[List[float]], bits: int = 8) -> List[Tuple[List[int], float]]:
    """
    Compress a list of token embeddings.
    
    Args:
        token_embeddings: List of token embedding vectors
        bits: Number of bits for quantization
        
    Returns:
        List of compressed embeddings (quantized_values, scale_factor)
    """
    compressed_embeddings = []
    for embedding in token_embeddings:
        compressed, scale = compress_vector(embedding, bits=bits)
        compressed_embeddings.append((compressed, scale))
    return compressed_embeddings

def decompress_token_embeddings(compressed_embeddings: List[Tuple[List[int], float]], 
                              bits: int = 8, vector_dim: int = None) -> List[List[float]]:
    """
    Decompress a list of compressed token embeddings.
    
    Args:
        compressed_embeddings: List of (compressed_data, scale_factor) tuples
        bits: Number of bits used in compression
        vector_dim: Original vector dimension (required for 4-bit)
        
    Returns:
        List of decompressed token embeddings
    """
    decompressed_embeddings = []
    for compressed, scale in compressed_embeddings:
        decompressed = decompress_vector(compressed, scale, bits=bits, vector_dim=vector_dim)
        decompressed_embeddings.append(decompressed)
    return decompressed_embeddings

# Test the compression utilities if this script is run directly
if __name__ == "__main__":
    # Create a test vector
    test_vector = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0] * 10  # 100-dim vector
    
    # Test 8-bit compression
    compressed_8bit, scale_8bit = compress_vector(test_vector, bits=8)
    decompressed_8bit = decompress_vector(compressed_8bit, scale_8bit, bits=8)
    ratio_8bit = calculate_compression_ratio(test_vector, compressed_8bit, bits=8)
    
    # Test 4-bit compression
    compressed_4bit, scale_4bit = compress_vector(test_vector, bits=4)
    decompressed_4bit = decompress_vector(compressed_4bit, scale_4bit, bits=4, vector_dim=len(test_vector))
    ratio_4bit = calculate_compression_ratio(test_vector, compressed_4bit, bits=4)
    
    # Print results
    print(f"Original vector length: {len(test_vector)}")
    print(f"8-bit compressed length: {len(compressed_8bit)}, ratio: {ratio_8bit:.2f}x")
    print(f"4-bit compressed length: {len(compressed_4bit)}, ratio: {ratio_4bit:.2f}x")
    
    # Calculate mean absolute error to check decompression quality
    mae_8bit = sum(abs(o - d) for o, d in zip(test_vector, decompressed_8bit)) / len(test_vector)
    mae_4bit = sum(abs(o - d) for o, d in zip(test_vector, decompressed_4bit)) / len(test_vector)
    
    print(f"8-bit Mean Absolute Error: {mae_8bit:.6f}")
    print(f"4-bit Mean Absolute Error: {mae_4bit:.6f}")
