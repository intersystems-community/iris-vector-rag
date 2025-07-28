#!/usr/bin/env python3
"""
Simple test to verify the vector search fix works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.vector_store_iris import IRISVectorStore
from common.iris_connection_manager import get_iris_connection
import numpy as np

def test_vector_search_fix():
    """Test that vector search works with mock connection manager."""
    print("🔍 Testing vector search fix...")
    
    # Create a mock connection manager (like the ones causing issues)
    mock_connection_manager = type('ConnectionManager', (), {
        'get_connection': lambda self: get_iris_connection()
    })()
    
    # Create config manager
    config_manager = ConfigurationManager()
    
    # Create vector store with mock connection manager
    vector_store = IRISVectorStore(
        config_manager=config_manager,
        connection_manager=mock_connection_manager
    )
    
    # Create a test embedding
    test_embedding = np.random.rand(384).tolist()  # MiniLM dimension
    
    try:
        # Try to perform a similarity search
        results = vector_store.similarity_search(
            query_embedding=test_embedding,
            top_k=5
        )
        print("✅ Vector search completed successfully!")
        print(f"   Found {len(results)} results")
        return True
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_search_fix()
    sys.exit(0 if success else 1)