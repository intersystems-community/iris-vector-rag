#!/usr/bin/env python3
"""
Test script to validate the vector format fix for LIST ERROR issues.
"""

import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.vector_format_fix import format_vector_for_iris, validate_vector_for_iris, VectorFormatError
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_formatting():
    """Test vector formatting with various edge cases that cause LIST ERROR."""
    
    print("🧪 Testing vector formatting fixes...")
    
    # Test cases that previously caused LIST ERROR
    test_cases = [
        # Normal case
        ([0.1, 0.2, 0.3, 0.4], "normal_vector"),
        
        # Edge cases that cause LIST ERROR
        ([float('nan'), 0.2, 0.3, 0.4], "with_nan"),
        ([float('inf'), 0.2, 0.3, 0.4], "with_inf"),
        ([1e20, 0.2, 0.3, 0.4], "very_large"),
        ([1e-20, 0.2, 0.3, 0.4], "very_small"),
        
        # Type issues
        (np.array([0.1, 0.2, 0.3, 0.4]), "numpy_array"),
        (np.array([1, 2, 3, 4], dtype=np.int32), "int_array"),
        (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "float32_array"),
        
        # Empty/problematic cases
        ([0.0, 0.0, 0.0, 0.0], "all_zeros"),
        ([-0.1, 0.2, -0.3, 0.4], "with_negatives"),
    ]
    
    success_count = 0
    for test_vector, description in test_cases:
        try:
            formatted = format_vector_for_iris(test_vector)
            valid = validate_vector_for_iris(formatted)
            
            if valid:
                print(f"✅ {description}: {len(formatted)} dims, all values finite")
                success_count += 1
            else:
                print(f"❌ {description}: validation failed")
                
        except VectorFormatError as e:
            print(f"❌ {description}: {e}")
        except Exception as e:
            print(f"❌ {description}: unexpected error: {e}")
    
    print(f"\n📊 Test Results: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)

def test_real_embedding_generation():
    """Test with real embedding generation to ensure compatibility."""
    
    print("\n🔬 Testing with real embedding generation...")
    
    try:
        # Get embedding function
        embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
        
        # Test texts that might cause issues
        test_texts = [
            "This is a normal test document.",
            "",  # Empty text
            "A" * 10000,  # Very long text
            "Special chars: àáâãäåæçèéêë",  # Unicode
            "Numbers: 123 456.789 -0.001",  # Numbers
        ]
        
        success_count = 0
        for i, text in enumerate(test_texts):
            try:
                if not text.strip():
                    print(f"⚠️  Test {i+1}: Skipping empty text")
                    continue
                
                # Generate embedding
                embeddings = embedding_func([text])
                embedding = embeddings[0]
                
                # Format for IRIS
                formatted = format_vector_for_iris(embedding)
                valid = validate_vector_for_iris(formatted, expected_dim=768)  # e5-base-v2 is 768-dim
                
                if valid:
                    print(f"✅ Test {i+1}: Generated {len(formatted)}-dim vector successfully")
                    success_count += 1
                else:
                    print(f"❌ Test {i+1}: Vector validation failed")
                    
            except Exception as e:
                print(f"❌ Test {i+1}: Error: {e}")
        
        print(f"\n📊 Real Embedding Results: {success_count}/{len([t for t in test_texts if t.strip()])} passed")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Real embedding test failed: {e}")
        return False

def test_database_insertion():
    """Test actual database insertion with formatted vectors."""
    
    print("\n💾 Testing database insertion...")
    
    try:
        # Get database connection
        connection = get_iris_connection()
        if not connection:
            print("❌ Could not connect to database")
            return False
        
        cursor = connection.cursor()
        
        # Test vector - use 768 dimensions to match e5-base-v2
        test_vector = [0.1, 0.2, 0.3] + [0.0] * 765  # 768-dim vector
        formatted_vector = format_vector_for_iris(test_vector)
        
        # Convert to string for VARCHAR column
        from data.loader_varchar_fixed import format_vector_for_varchar_column
        vector_string = format_vector_for_varchar_column(formatted_vector)
        
        # Test insertion
        test_sql = """
        INSERT INTO RAG.SourceDocuments_V2
        (doc_id, title, text_content, authors, keywords, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        test_params = [
            "test_vector_fix",
            "Test Document for Vector Fix",
            "This is a test document to validate vector format fixes.",
            "[]",
            "[]",
            vector_string  # This should now work without LIST ERROR
        ]
        
        cursor.execute(test_sql, test_params)
        connection.commit()
        
        print("✅ Database insertion successful - no LIST ERROR!")
        
        # Clean up
        cursor.execute("DELETE FROM RAG.SourceDocuments_V2 WHERE doc_id = ?", ["test_vector_fix"])
        connection.commit()
        cursor.close()
        connection.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database insertion failed: {e}")
        return False

def main():
    """Run all vector format tests."""
    
    print("🔧 VECTOR FORMAT FIX VALIDATION")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_vector_formatting()
    test2_passed = test_real_embedding_generation()
    test3_passed = test_database_insertion()
    
    print("\n" + "=" * 50)
    print("📋 FINAL RESULTS:")
    print(f"✅ Vector Formatting: {'PASS' if test1_passed else 'FAIL'}")
    print(f"✅ Real Embeddings: {'PASS' if test2_passed else 'FAIL'}")
    print(f"✅ Database Insertion: {'PASS' if test3_passed else 'FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED - Vector format fix is working!")
        print("✅ LIST ERROR issues should be resolved")
        return True
    else:
        print("\n❌ Some tests failed - vector format fix needs more work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)