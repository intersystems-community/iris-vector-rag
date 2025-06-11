#!/usr/bin/env python3
"""
Quick test to verify ColBERT pipeline works with existing data.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from common.utils import get_iris_connector_for_embedded, get_llm_func, get_colbert_query_encoder_func
from core_pipelines.colbert_pipeline import run_colbert_pipeline

def test_colbert_with_existing_data():
    """Test ColBERT pipeline with existing data."""
    
    print("🔍 Testing ColBERT pipeline with existing data...")
    
    try:
        # Get required functions
        iris_connector = get_iris_connector_for_embedded()
        llm_func = get_llm_func()
        colbert_query_encoder = get_colbert_query_encoder_func()
        
        print("✅ Successfully loaded all required functions")
        
        # Test query
        test_query = "What is machine learning?"
        print(f"🔍 Testing query: '{test_query}'")
        
        start_time = time.time()
        
        # Run ColBERT pipeline
        result = run_colbert_pipeline(
            query=test_query,
            iris_connector=iris_connector,
            colbert_query_encoder_func=colbert_query_encoder,
            llm_func=llm_func,
            top_k=3
        )
        
        execution_time = time.time() - start_time
        
        print(f"✅ ColBERT pipeline executed successfully!")
        print(f"⏱️  Execution time: {execution_time:.2f}s")
        print(f"📝 Query: {result['query']}")
        print(f"🔧 Technique: {result['technique']}")
        print(f"📄 Retrieved documents: {len(result['retrieved_documents'])}")
        print(f"💬 Answer length: {len(result['answer'])} characters")
        
        # Check if we got meaningful results
        if len(result['retrieved_documents']) > 0:
            print("✅ ColBERT successfully retrieved documents!")
            print(f"📊 First document score: {result['retrieved_documents'][0].get('score', 'N/A')}")
            return True
        else:
            print("⚠️  ColBERT didn't retrieve any documents")
            return False
            
    except Exception as e:
        print(f"❌ ColBERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_colbert_with_existing_data()
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILURE'}: ColBERT test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)