#!/usr/bin/env python3
"""
Use rag-templates to answer: "what are the main causes of diabetes"
This demonstrates the MCP server functionality with real user interaction.
"""

import sys
import os
import json
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def answer_diabetes_causes():
    """Use rag-templates to answer the user's question about diabetes causes."""
    try:
        # Import the MCP bridge
        from objectscript.mcp_bridge import MCPBridge
        
        print("🔧 Initializing RAG-Templates MCP Bridge...")
        bridge = MCPBridge()
        
        user_question = "what are the main causes of diabetes"
        print(f"❓ User Question: '{user_question}'")
        print("🔍 Searching PMC medical literature database...")
        
        # Configure search parameters
        search_params = {
            "query": user_question,
            "options": {
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 1024,
                "include_sources": True
            }
        }
        
        # Execute RAG search using basic technique
        print("📋 Processing with Basic RAG technique...")
        result = await bridge.invoke_technique('basic', search_params['query'], search_params)
        
        print("✅ RAG search completed successfully!")
        
        if isinstance(result, dict) and result.get('success') and 'result' in result:
            actual_result = result['result']
            
            print("\n" + "="*80)
            print("📋 ANSWER: Main Causes of Diabetes")
            print("="*80)
            
            if isinstance(actual_result, dict) and 'answer' in actual_result:
                print(actual_result['answer'])
            else:
                print("Answer format not as expected")
                print(f"Result structure: {actual_result}")
            
            print("="*80)
            
            # Show source information
            if isinstance(actual_result, dict) and 'retrieved_documents' in actual_result:
                docs = actual_result['retrieved_documents']
                print(f"\n📚 Sources: Retrieved {len(docs)} relevant medical documents from PMC database")
                
                # Show first few document titles if available
                for i, doc in enumerate(docs[:3]):
                    if isinstance(doc, dict):
                        title = doc.get('title', doc.get('content', 'Unknown')[:100] + '...')
                        print(f"   {i+1}. {title}")
            
            # Show performance metrics
            if 'performance' in result:
                perf = result['performance']
                if isinstance(perf, dict):
                    print(f"\n⚡ Performance: Query processed in {perf.get('execution_time', 'N/A')} seconds")
        
        else:
            print("❌ Error in RAG processing:")
            if isinstance(result, dict) and 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Unexpected result format: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error using rag-templates: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Using RAG-Templates to Answer: 'What are the main causes of diabetes?'")
    print("=" * 80)
    
    success = asyncio.run(answer_diabetes_causes())
    
    if success:
        print("\n🎉 Successfully answered user question using rag-templates!")
        print("✅ MCP server functionality validated with real user interaction")
    else:
        print("\n💥 Failed to answer question!")
        print("❌ There may be issues with the MCP implementation")
    
    sys.exit(0 if success else 1)