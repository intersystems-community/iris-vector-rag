#!/usr/bin/env python3
"""
Test script to verify that the LLM cache can reuse existing IRIS connections.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cache_connection_reuse():
    """Test that cache can reuse existing IRIS connection."""
    
    print("🧪 Testing LLM Cache Connection Reuse")
    print("=" * 50)
    
    # Test 1: Check if we can get an IRIS connection
    print("\n1. Testing IRIS connection availability...")
    try:
        from common.iris_connection_manager import get_iris_connection
        connection = get_iris_connection()
        print("✅ IRIS connection available")
        connection_available = True
    except Exception as e:
        print(f"❌ IRIS connection failed: {e}")
        connection_available = False
    
    # Test 2: Test cache manager with connection reuse
    print("\n2. Testing cache manager with connection reuse...")
    try:
        from common.llm_cache_manager import LangchainCacheManager
        from common.llm_cache_config import CacheConfig
        
        # Create config for IRIS backend
        config = CacheConfig(
            enabled=True,
            backend='iris',
            ttl_seconds=3600,
            table_name='test_llm_cache'
        )
        
        # Test cache manager
        cache_manager = LangchainCacheManager(config)
        cache_instance = cache_manager.setup_cache()
        
        if cache_instance:
            print("✅ Cache manager successfully initialized with connection reuse")
            
            # Test basic cache operations
            print("\n3. Testing basic cache operations...")
            
            # Test the cache backend directly
            if hasattr(cache_manager, 'cache_backend') and cache_manager.cache_backend:
                backend = cache_manager.cache_backend
                
                # Test set operation
                backend.set("test_key", {"test": "value"}, ttl=60)
                print("✅ Cache set operation successful")
                
                # Test get operation
                result = backend.get("test_key")
                if result and result.get("test") == "value":
                    print("✅ Cache get operation successful")
                else:
                    print(f"❌ Cache get operation failed: {result}")
                
                # Test cache stats
                stats = backend.get_stats()
                print(f"📊 Cache stats: {stats}")
                
            else:
                print("❌ Cache backend not available")
        else:
            print("❌ Cache manager initialization failed")
            
    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test cache status check from run_ragas
    print("\n4. Testing cache status check...")
    try:
        # Temporarily unset IRIS_CONNECTION_URL to test connection reuse
        original_url = os.environ.get('IRIS_CONNECTION_URL')
        if 'IRIS_CONNECTION_URL' in os.environ:
            del os.environ['IRIS_CONNECTION_URL']
        
        from eval.run_ragas import check_cache_status
        status = check_cache_status()
        
        print(f"Cache Status: {status.get('cache_status', 'Unknown')}")
        print(f"Cache Configured: {status.get('cache_configured', False)}")
        print("Details:")
        for detail in status.get('details', []):
            print(f"  - {detail}")
        
        # Restore original URL if it existed
        if original_url:
            os.environ['IRIS_CONNECTION_URL'] = original_url
            
    except Exception as e:
        print(f"❌ Cache status check failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_cache_connection_reuse()