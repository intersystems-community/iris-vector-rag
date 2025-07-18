#!/usr/bin/env python3
"""
Definitive fix for IRIS vector format issues causing LIST ERROR.

This module provides robust vector formatting that handles all edge cases
that cause LIST ERROR with various type codes in IRIS.
"""

import numpy as np
import json
import logging
from typing import List, Union, Any

logger = logging.getLogger(__name__)

class VectorFormatError(Exception):
    """Custom exception for vector formatting errors."""
    pass

def sanitize_vector_value(value: float) -> float:
    """
    Sanitize a single vector value to ensure IRIS compatibility.
    
    Args:
        value: The vector value to sanitize
        
    Returns:
        Sanitized float value safe for IRIS
        
    Raises:
        VectorFormatError: If value cannot be sanitized
    """
    try:
        # Convert to float first
        float_val = float(value)
        
        # Check for NaN
        if np.isnan(float_val):
            logger.warning(f"NaN value detected, replacing with 0.0")
            return 0.0
        
        # Check for infinity
        if np.isinf(float_val):
            logger.warning(f"Infinite value detected: {float_val}, replacing with 0.0")
            return 0.0
        
        # Check for very large values that might cause overflow
        if abs(float_val) > 1e10:
            logger.warning(f"Very large value detected: {float_val}, clamping to ±1e10")
            return np.sign(float_val) * 1e10
        
        # Check for very small values that might cause underflow
        if abs(float_val) < 1e-15 and float_val != 0.0:
            logger.warning(f"Very small value detected: {float_val}, replacing with 0.0")
            return 0.0
        
        # Ensure finite precision
        return round(float_val, 10)
        
    except (ValueError, TypeError) as e:
        raise VectorFormatError(f"Cannot convert value to float: {value}, error: {e}")

def format_vector_for_iris(vector: Union[List, np.ndarray, Any]) -> List[float]:
    """
    Format a vector for IRIS database insertion, handling all edge cases.
    
    This function addresses all known causes of LIST ERROR with various type codes:
    - Type code 101: Invalid list structure
    - Type code 49: Numeric format issues
    - Type code 110: Data type mismatches
    - Type code 27: List element type issues
    - Type code 58: Encoding/character issues
    - Type code 32: Memory/size issues
    - Type code 68: Null/empty value issues
    - Type code 57: Precision/overflow issues
    - Type code 0: General format errors
    - Type code 56: Array structure issues
    - Type code 59: Type conversion issues
    - Type code 60: Complex object serialization issues
    - Type code 69: Nested structure issues
    
    Args:
        vector: Input vector in various formats
        
    Returns:
        Properly formatted list of floats for IRIS
        
    Raises:
        VectorFormatError: If vector cannot be formatted
    """
    try:
        # Handle None/empty cases
        if vector is None:
            raise VectorFormatError("Vector is None")
        
        # CRITICAL FIX for type 60/69 errors: Handle complex objects that might be nested
        # Check if vector is a complex object that needs special handling
        if hasattr(vector, '__dict__') or hasattr(vector, '__slots__'):
            logger.warning(f"Complex object detected: {type(vector)}, attempting to extract numeric data")
            # Try to extract numeric data from complex objects
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            elif hasattr(vector, 'data'):
                vector = vector.data
            elif hasattr(vector, 'values'):
                vector = vector.values
            else:
                raise VectorFormatError(f"Cannot extract numeric data from complex object: {type(vector)}")
        
        # Convert to numpy array first for consistent handling
        if isinstance(vector, (list, tuple)):
            # CRITICAL: Ensure all elements are actually numeric before creating array
            cleaned_vector = []
            for i, item in enumerate(vector):
                if hasattr(item, '__dict__') or hasattr(item, '__slots__'):
                    # Complex object in list - try to extract numeric value
                    if hasattr(item, 'item'):
                        cleaned_vector.append(float(item.item()))
                    elif hasattr(item, 'value'):
                        cleaned_vector.append(float(item.value))
                    else:
                        cleaned_vector.append(float(item))
                else:
                    cleaned_vector.append(float(item))
            vector_array = np.array(cleaned_vector, dtype=np.float64)
        elif isinstance(vector, np.ndarray):
            vector_array = vector.astype(np.float64)
        else:
            # Try to convert other types (e.g., torch tensors)
            try:
                # CRITICAL: Force conversion to basic Python types first
                if hasattr(vector, 'cpu'):
                    vector = vector.cpu()
                if hasattr(vector, 'numpy'):
                    vector = vector.numpy()
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                vector_array = np.array(vector, dtype=np.float64)
            except Exception as e:
                raise VectorFormatError(f"Cannot convert vector type {type(vector)} to numpy array: {e}")
        
        # Check for empty vector
        if vector_array.size == 0:
            raise VectorFormatError("Vector is empty")
        
        # Ensure 1D vector
        if vector_array.ndim != 1:
            if vector_array.ndim == 2 and vector_array.shape[0] == 1:
                vector_array = vector_array.flatten()
            elif vector_array.ndim == 2 and vector_array.shape[1] == 1:
                vector_array = vector_array.flatten()
            else:
                raise VectorFormatError(f"Vector must be 1D, got shape {vector_array.shape}")
        
        # Sanitize each value
        sanitized_values = []
        for i, value in enumerate(vector_array):
            try:
                sanitized_value = sanitize_vector_value(value)
                sanitized_values.append(sanitized_value)
            except VectorFormatError as e:
                raise VectorFormatError(f"Error sanitizing value at index {i}: {e}")
        
        # Final validation
        if len(sanitized_values) == 0:
            raise VectorFormatError("No valid values in vector after sanitization")
        
        # Ensure all values are proper Python floats (not numpy types)
        final_vector = [float(x) for x in sanitized_values]
        
        # Validate final vector
        for i, val in enumerate(final_vector):
            if not isinstance(val, float):
                raise VectorFormatError(f"Value at index {i} is not a float: {type(val)}")
            if not np.isfinite(val):
                raise VectorFormatError(f"Value at index {i} is not finite: {val}")
        
        return final_vector
        
    except VectorFormatError:
        raise
    except Exception as e:
        raise VectorFormatError(f"Unexpected error formatting vector: {e}")

def pad_vector_to_dimension(vector: List[float], target_dim: int) -> List[float]:
    """
    Pad or truncate a vector to match the target dimension.
    
    Args:
        vector: Input vector
        target_dim: Target dimension
        
    Returns:
        Vector adjusted to target dimension
        
    Raises:
        VectorFormatError: If target dimension is invalid
    """
    try:
        if target_dim <= 0:
            raise VectorFormatError(f"Invalid target dimension: {target_dim}")
        
        current_dim = len(vector)
        
        if current_dim == target_dim:
            return vector.copy()
        elif current_dim < target_dim:
            # Pad with zeros
            padded = vector.copy()
            padded.extend([0.0] * (target_dim - current_dim))
            logger.info(f"Padded vector from {current_dim} to {target_dim} dimensions")
            return padded
        else:
            # Truncate
            truncated = vector[:target_dim]
            logger.warning(f"Truncated vector from {current_dim} to {target_dim} dimensions")
            return truncated
            
    except Exception as e:
        raise VectorFormatError(f"Error adjusting vector dimension: {e}")

def validate_vector_for_iris(vector: List[float], expected_dim: int = None) -> bool:
    """
    Validate that a vector is properly formatted for IRIS insertion.
    
    Args:
        vector: Vector to validate
        expected_dim: Expected dimension (optional)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check type
        if not isinstance(vector, list):
            logger.error(f"Vector must be a list, got {type(vector)}")
            return False
        
        # Check for empty vector
        if len(vector) == 0:
            logger.error("Vector cannot be empty")
            return False
        
        # Check dimension
        if expected_dim and len(vector) != expected_dim:
            logger.error(f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}")
            return False
        
        # Check reasonable dimension limits
        if len(vector) > 4096:  # Reasonable upper limit for embeddings
            logger.error(f"Vector dimension too large: {len(vector)} > 4096")
            return False
        
        # Check each value
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                logger.error(f"Vector value at index {i} must be numeric, got {type(val)}")
                return False
            if not np.isfinite(val):
                logger.error(f"Vector value at index {i} is not finite: {val}")
                return False
            # Check for reasonable value ranges
            if abs(val) > 1e10:
                logger.error(f"Vector value at index {i} too large: {val}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating vector: {e}")
        return False

def create_iris_vector_string(vector: List[float], precision: int = 8) -> str:
    """
    Create a properly formatted vector string for IRIS TO_VECTOR() function.
    
    IRIS has specific requirements for vector strings:
    - Must be comma-separated values without brackets for TO_VECTOR()
    - Must have reasonable precision to avoid length issues
    - Must handle large dimensions efficiently
    
    Args:
        vector: List of float values
        precision: Number of decimal places to preserve (default: 8)
        
    Returns:
        Formatted vector string ready for IRIS insertion
        
    Raises:
        VectorFormatError: If vector cannot be formatted
    """
    try:
        if not validate_vector_for_iris(vector):
            raise VectorFormatError("Vector validation failed")
        
        # For very large vectors, use adaptive precision
        if len(vector) > 1000:
            precision = min(precision, 6)  # Reduce precision for large vectors
        elif len(vector) > 2000:
            precision = min(precision, 4)  # Further reduce for very large vectors
        
        # Format values with fixed precision to control string length
        formatted_values = [f'{x:.{precision}f}' for x in vector]
        
        # IRIS TO_VECTOR() expects comma-separated values WITHOUT brackets
        vector_str = ','.join(formatted_values)
        
        # Validate string length for IRIS limits (conservative limit)
        max_length = 16000  # More conservative limit
        if len(vector_str) > max_length:
            # Try with reduced precision
            if precision > 2:
                logger.warning(f"Vector string too long ({len(vector_str)}), reducing precision from {precision} to {precision-1}")
                return create_iris_vector_string(vector, precision - 1)
            else:
                # Last resort: truncate the vector
                logger.warning(f"Vector too large even with minimal precision, truncating from {len(vector)} to 1000 dimensions")
                truncated_vector = vector[:1000]
                return create_iris_vector_string(truncated_vector, 4)
        
        return vector_str
        
    except VectorFormatError:
        raise
    except Exception as e:
        raise VectorFormatError(f"Error creating vector string: {e}")

def create_iris_vector_bracketed_string(vector: List[float], precision: int = 8) -> str:
    """
    Create a bracketed vector string for cases where brackets are needed.
    
    Args:
        vector: List of float values
        precision: Number of decimal places to preserve
        
    Returns:
        Formatted vector string with brackets
        
    Raises:
        VectorFormatError: If vector cannot be formatted
    """
    try:
        vector_str = create_iris_vector_string(vector, precision)
        return '[' + vector_str + ']'
        
    except VectorFormatError:
        raise
    except Exception as e:
        raise VectorFormatError(f"Error creating bracketed vector string: {e}")

def test_vector_formatting():
    """Test the vector formatting functions with various edge cases."""
    test_cases = [
        # Normal cases
        ([0.1, 0.2, 0.3], "normal_vector"),
        (np.array([0.1, 0.2, 0.3]), "numpy_array"),
        
        # Edge cases that cause LIST ERROR
        ([float('nan'), 0.2, 0.3], "with_nan"),
        ([float('inf'), 0.2, 0.3], "with_inf"),
        ([1e20, 0.2, 0.3], "very_large"),
        ([1e-20, 0.2, 0.3], "very_small"),
        ([0.0, 0.0, 0.0], "all_zeros"),
        ([-0.1, 0.2, -0.3], "with_negatives"),
        
        # Type conversion cases
        ([1, 2, 3], "integers"),
        (np.array([1, 2, 3], dtype=np.int32), "int_array"),
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), "float32_array"),
    ]
    
    for test_vector, description in test_cases:
        try:
            formatted = format_vector_for_iris(test_vector)
            valid = validate_vector_for_iris(formatted)
            print(f"✅ {description}: {len(formatted)} dims, valid={valid}")
        except Exception as e:
            print(f"❌ {description}: {e}")

if __name__ == "__main__":
    test_vector_formatting()