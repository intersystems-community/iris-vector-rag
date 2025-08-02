#!/usr/bin/env python3
"""
Simple API Demo for RAG Templates Library Consumption Framework.

This script demonstrates the zero-configuration Simple API that enables
immediate RAG usage with sensible defaults.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_templates import RAG

def main():
    """Demonstrate the Simple API functionality."""
    
    print("🚀 RAG Templates Simple API Demo")
    print("=" * 50)
    
    # Zero-configuration initialization
    print("\n1. Zero-Config Initialization:")
    rag = RAG()
    print(f"   ✅ RAG instance created: {rag}")
    
    # Add some sample documents
    print("\n2. Adding Documents:")
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from the world.",
        "Reinforcement learning is a type of machine learning where agents learn through interaction with an environment."
    ]
    
    rag.add_documents(documents)
    print(f"   ✅ Added {len(documents)} documents to knowledge base")
    print(f"   📊 Total documents: {rag.get_document_count()}")
    
    # Query the system
    print("\n3. Querying the System:")
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is NLP?"
    ]
    
    for query in queries:
        print(f"\n   🔍 Query: {query}")
        try:
            answer = rag.query(query)
            print(f"   💡 Answer: {answer}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Show configuration
    print("\n4. Configuration Information:")
    print(f"   🏠 Database Host: {rag.get_config('database:iris:host')}")
    print(f"   🔌 Database Port: {rag.get_config('database:iris:port')}")
    print(f"   🧠 Embedding Model: {rag.get_config('embeddings:model')}")
    print(f"   📏 Embedding Dimension: {rag.get_config('embeddings:dimension')}")
    
    # Validate configuration
    print("\n5. Configuration Validation:")
    try:
        is_valid = rag.validate_config()
        print(f"   ✅ Configuration is valid: {is_valid}")
    except Exception as e:
        print(f"   ⚠️  Configuration validation: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Simple API Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Zero-configuration initialization")
    print("• Simple document addition")
    print("• Easy querying with string responses")
    print("• Built-in configuration management")
    print("• Environment variable support")
    print("• Error handling with helpful messages")


if __name__ == "__main__":
    main()