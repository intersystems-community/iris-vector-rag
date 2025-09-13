#!/usr/bin/env python3
"""
Test retrieval with queries appropriate for the PMC content in the database.
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.pipelines.basic import BasicRAGPipeline
from common.iris_connection_manager import IRISConnectionManager as ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from common.utils import get_llm_func


def test_retrieval():
    """Test retrieval with appropriate medical queries."""
    print("Testing RAG retrieval with PMC-appropriate queries...\n")
    
    # Initialize components
    try:
        # Initialize managers
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Get LLM function (using stub for testing)
        llm_func = get_llm_func(provider="stub")
        
        # Initialize pipeline
        pipeline = BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
        print("✅ Pipeline initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries appropriate for PMC content
    test_queries = [
        "What are the latest treatments for cancer?",
        "How does the immune system work?",
        "What are the symptoms of viral infections?",
        "Explain the diagnosis and treatment of hypertension",
        "What is the pathophysiology of inflammation?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")
        
        try:
            # Retrieve documents
            retrieved_docs = pipeline.retrieve(query, top_k=3)
            
            if retrieved_docs:
                print(f"\n✅ Retrieved {len(retrieved_docs)} documents:")
                for j, doc in enumerate(retrieved_docs, 1):
                    print(f"\n  Document {j}:")
                    print(f"    ID: {doc.id}")
                    print(f"    Score: {doc.score:.4f}" if doc.score else "    Score: N/A")
                    # Show content preview
                    content_preview = doc.content[:200] if doc.content else "N/A"
                    print(f"    Content: {content_preview}...")
                    
                # Try to generate a response
                try:
                    response = pipeline.generate(query, retrieved_docs)
                    print(f"\n📝 Generated Response:")
                    print(f"{response[:500]}...")
                except Exception as e:
                    print(f"\n⚠️  Generation failed: {e}")
            else:
                print("\n❌ No documents retrieved")
                
        except Exception as e:
            print(f"\n❌ Retrieval error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with the original queries to confirm they don't work
    print(f"\n\n{'='*80}")
    print("Testing original drug-specific queries (expected to fail):")
    print(f"{'='*80}")
    
    drug_queries = [
        "What are the cardiovascular benefits of metformin?",
        "How do SGLT2 inhibitors work?"
    ]
    
    for query in drug_queries:
        print(f"\nQuery: {query}")
        try:
            retrieved_docs = pipeline.retrieve(query, top_k=3)
            if retrieved_docs:
                print(f"  ✅ Surprisingly retrieved {len(retrieved_docs)} documents")
            else:
                print(f"  ❌ No documents retrieved (as expected - no metformin/SGLT2 content)")
        except Exception as e:
            print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    test_retrieval()