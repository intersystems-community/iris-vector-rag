#!/usr/bin/env python3
"""
Test script to verify MCP server functionality with diabetes query.
This script directly tests the Python bridge to ensure the schema validation fix worked.
"""

import sys
import os
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mcp_bridge_diabetes_query():
    """Test the MCP bridge with a diabetes query to verify schema validation fix."""
    try:
        # Import the MCP bridge
        from objectscript.mcp_bridge import MCPBridge
        
        print("🔧 Initializing MCP Bridge...")
        bridge = MCPBridge()
        
        print("📋 Testing basic RAG with diabetes query...")
        
        # Test parameters for basic RAG
        test_params = {
            "query": "diabetes",
            "options": {
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 1024,
                "include_sources": True
            }
        }
        
        print(f"🔍 Query parameters: {json.dumps(test_params, indent=2)}")
        
        # Execute basic RAG using the correct method
        import asyncio
        result = asyncio.run(bridge.invoke_technique('basic', test_params['query'], test_params))
        
        print("✅ MCP Bridge execution successful!")
        print(f"📊 Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"🔑 Result keys: {list(result.keys())}")
            if 'query' in result:
                print(f"📝 Query: {result['query']}")
            if 'answer' in result:
                print(f"💬 Answer preview: {result['answer'][:200]}...")
            if 'performance_metrics' in result:
                print(f"⚡ Performance metrics available: {bool(result['performance_metrics'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MCP bridge: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing MCP Bridge with Diabetes Query")
    print("=" * 50)
    
    success = test_mcp_bridge_diabetes_query()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("✅ Schema validation fix is working")
        print("✅ MCP server can handle diabetes queries")
    else:
        print("\n💥 Test failed!")
        print("❌ There may be issues with the implementation")
    
    sys.exit(0 if success else 1)