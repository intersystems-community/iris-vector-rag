#!/usr/bin/env python3
"""
Debug script to understand why vector conversion is producing hash strings.
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from common.vector_format_fix import format_vector_for_iris, create_iris_vector_string, validate_vector_for_iris

def test_vector_conversion():
    """Test the vector conversion process step by step."""
    
    # Create a simple test vector
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"Original vector: {test_vector}")
    print(f"Vector type: {type(test_vector)}")
    
    try:
        # Step 1: Format the vector
        formatted_vector = format_vector_for_iris(test_vector)
        print(f"Formatted vector: {formatted_vector}")
        print(f"Formatted vector type: {type(formatted_vector)}")
        
        # Step 2: Validate the vector
        is_valid = validate_vector_for_iris(formatted_vector)
        print(f"Vector is valid: {is_valid}")
        
        # Step 3: Create the IRIS vector string
        vector_str = create_iris_vector_string(formatted_vector)
        print(f"Vector string: {vector_str}")
        print(f"Vector string type: {type(vector_str)}")
        print(f"Vector string length: {len(vector_str)}")
        
        # Check if it looks like a hash
        if '@$vector' in vector_str:
            print("ERROR: Vector string contains '@$vector' - this is wrong!")
        else:
            print("SUCCESS: Vector string looks correct")
            
    except Exception as e:
        print(f"ERROR in vector conversion: {e}")
        import traceback
        traceback.print_exc()

def test_numpy_vector():
    """Test with numpy array."""
    print("\n" + "="*50)
    print("Testing with numpy array")
    
    test_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"Original numpy vector: {test_vector}")
    print(f"Vector type: {type(test_vector)}")
    
    try:
        formatted_vector = format_vector_for_iris(test_vector)
        print(f"Formatted vector: {formatted_vector}")
        
        vector_str = create_iris_vector_string(formatted_vector)
        print(f"Vector string: {vector_str}")
        
        if '@$vector' in vector_str:
            print("ERROR: Vector string contains '@$vector' - this is wrong!")
        else:
            print("SUCCESS: Vector string looks correct")
            
    except Exception as e:
        print(f"ERROR in numpy vector conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_conversion()
    test_numpy_vector()