#!/usr/bin/env python3
"""
Simple verification that ColBERT dimension fix is working.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_colbert_dimension_fix():
    """Verify ColBERT query encoder uses correct dimensions."""
    
    print("🔍 Verifying ColBERT dimension fix...")
    
    try:
        from common.utils import get_colbert_query_encoder_func, get_config_value
        
        # Check config value
        expected_dimension = get_config_value("embedding_model.dimension", 384)
        print(f"📊 Expected embedding dimension from config: {expected_dimension}")
        
        # Get ColBERT query encoder
        encoder_func = get_colbert_query_encoder_func()
        
        # Test with a query
        test_query = "What is machine learning?"
        query_embeddings = encoder_func(test_query)
        
        print(f"🔍 Test query: '{test_query}'")
        print(f"📝 Generated {len(query_embeddings)} token embeddings")
        
        if query_embeddings:
            actual_dimension = len(query_embeddings[0])
            print(f"📊 Actual embedding dimension: {actual_dimension}")
            
            if actual_dimension == expected_dimension:
                print("✅ SUCCESS: ColBERT query encoder is using correct dimensions!")
                print(f"✅ Dimension match: {actual_dimension}D embeddings match config")
                return True
            else:
                print(f"❌ FAILURE: Dimension mismatch - expected {expected_dimension}, got {actual_dimension}")
                return False
        else:
            print("❌ FAILURE: No embeddings generated")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_colbert_dimension_fix()
    print(f"\n{'🎉 VERIFICATION PASSED' if success else '❌ VERIFICATION FAILED'}")
    sys.exit(0 if success else 1)