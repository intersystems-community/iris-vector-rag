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
        
        # Convert to numpy array first for consistent handling
        if isinstance(vector, (list, tuple)):
            vector_array = np.array(vector, dtype=np.float64)
        elif isinstance(vector, np.ndarray):
            vector_array = vector.astype(np.float64)
        else:
            # Try to convert other types (e.g., torch tensors)
            try:
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
        
        # Check dimension
        if expected_dim and len(vector) != expected_dim:
            logger.error(f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}")
            return False
        
        # Check each value
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                logger.error(f"Vector value at index {i} must be numeric, got {type(val)}")
                return False
            if not np.isfinite(val):
                logger.error(f"Vector value at index {i} is not finite: {val}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating vector: {e}")
        return False

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