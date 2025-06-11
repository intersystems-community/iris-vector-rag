#!/usr/bin/env python3
"""
Debug script to isolate the ColBERT retrieval issue.
"""

import os
import sys
import logging
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.colbert import ColBERTRAGPipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_colbert_retrieval():
    """Debug the ColBERT retrieval issue step by step."""
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Create pipeline
        pipeline = ColBERTRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        # Test query
        test_query = "What are the effects of diabetes?"
        
        print(f"🔍 Testing ColBERT retrieval with query: '{test_query}'")
        
        # Step 1: Test validation
        print("\n📋 Step 1: Validating setup...")
        is_valid = pipeline.validate_setup()
        print(f"   Setup valid: {is_valid}")
        
        if not is_valid:
            print("❌ Setup validation failed - cannot proceed")
            return False
        
        # Step 2: Test query encoding
        print("\n🔤 Step 2: Testing query encoding...")
        try:
            query_tokens = pipeline.colbert_query_encoder(test_query)
            print(f"   Query tokens shape: {np.array(query_tokens).shape}")
            print(f"   Query tokens type: {type(query_tokens)}")
        except Exception as e:
            print(f"   ❌ Query encoding failed: {e}")
            return False
        
        # Step 3: Test candidate retrieval
        print("\n🎯 Step 3: Testing candidate document retrieval...")
        try:
            candidates = pipeline._retrieve_candidate_documents_hnsw(test_query, k=10)
            print(f"   Candidates found: {len(candidates)}")
            if candidates:
                print(f"   First few candidate IDs: {candidates[:5]}")
        except Exception as e:
            print(f"   ❌ Candidate retrieval failed: {e}")
            return False
        
        # Step 4: Test token embedding loading
        print("\n💾 Step 4: Testing token embedding loading...")
        try:
            if candidates:
                token_embeddings = pipeline._load_token_embeddings_for_candidates(candidates[:5])
                print(f"   Token embeddings loaded for {len(token_embeddings)} documents")
                for doc_id, embeddings in list(token_embeddings.items())[:2]:
                    print(f"   Doc {doc_id}: {len(embeddings)} token embeddings")
            else:
                print("   ⚠️  No candidates to load token embeddings for")
        except Exception as e:
            print(f"   ❌ Token embedding loading failed: {e}")
            return False
        
        # Step 5: Test the problematic query method
        print("\n🔧 Step 5: Testing the query() method (this should fail)...")
        try:
            retrieved_docs = pipeline.query(test_query, top_k=5)
            print(f"   Retrieved documents: {len(retrieved_docs)}")
        except Exception as e:
            print(f"   ❌ Query method failed: {e}")
            print(f"   This is the root cause of the 0 documents issue!")
            return False
        
        # Step 6: Test the full pipeline
        print("\n🚀 Step 6: Testing full pipeline...")
        try:
            result = pipeline.run(test_query, top_k=5)
            print(f"   Pipeline result keys: {list(result.keys())}")
            print(f"   Retrieved documents: {len(result.get('retrieved_documents', []))}")
            print(f"   Answer length: {len(result.get('answer', ''))}")
        except Exception as e:
            print(f"   ❌ Full pipeline failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🪲 ColBERT Retrieval Debug Script")
    print("=" * 50)
    
    success = debug_colbert_retrieval()
    
    if success:
        print("\n✅ Debug completed successfully")
    else:
        print("\n❌ Debug revealed issues that need fixing")