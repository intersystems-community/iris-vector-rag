#!/usr/bin/env python3
"""
Example usage of the Basic RAG Pipeline.

This script demonstrates how to:
1. Create a Basic RAG Pipeline
2. Load documents into the knowledge base
3. Query the pipeline for answers
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from iris_rag import create_pipeline
from iris_rag.core.models import Document


def simple_llm_function(prompt: str) -> str:
    """
    A simple mock LLM function for demonstration.
    In a real implementation, this would call an actual LLM service.
    """
    return f"Based on the provided context, here's a summary: {prompt[:200]}..."


def main():
    """Main example function."""
    print("Basic RAG Pipeline Example")
    print("=" * 50)
    
    # Create sample documents
    sample_documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            metadata={"source": "ml_intro.txt", "topic": "machine_learning"}
        ),
        Document(
            page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"source": "dl_intro.txt", "topic": "deep_learning"}
        ),
        Document(
            page_content="Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
            metadata={"source": "nlp_intro.txt", "topic": "nlp"}
        )
    ]
    
    try:
        # Create the pipeline
        print("1. Creating Basic RAG Pipeline...")
        pipeline = create_pipeline(
            pipeline_type="basic",
            config_path=None,  # Will use default configuration
            llm_func=simple_llm_function
        )
        print("   ✓ Pipeline created successfully")
        
        # Load documents
        print("\n2. Loading documents into knowledge base...")
        pipeline.load_documents(
            documents_path="",  # Not used when providing documents directly
            documents=sample_documents,
            chunk_documents=False,  # Keep documents as-is for this example
            generate_embeddings=True
        )
        print(f"   ✓ Loaded {len(sample_documents)} documents")
        
        # Query the pipeline
        print("\n3. Querying the pipeline...")
        queries = [
            "What is machine learning?",
            "Tell me about deep learning",
            "How does NLP work?"
        ]
        
        for query in queries:
            print(f"\n   Query: {query}")
            result = pipeline.query(query, top_k=2)
            
            print(f"   Answer: {result['answer']}")
            print(f"   Retrieved {len(result['retrieved_documents'])} documents")
            print(f"   Processing time: {result['metadata']['processing_time']:.3f}s")
            
            # Show sources
            if result.get('sources'):
                print("   Sources:")
                for source in result['sources']:
                    print(f"     - {source['source']} (ID: {source['document_id'][:8]}...)")
        
        # Show pipeline statistics
        print(f"\n4. Pipeline Statistics:")
        print(f"   Total documents in knowledge base: {pipeline.get_document_count()}")
        print(f"   Embedding dimension: {pipeline.embedding_manager.get_embedding_dimension()}")
        print(f"   Available embedding backends: {pipeline.embedding_manager.get_available_backends()}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This example requires:")
        print("- sentence-transformers library (pip install sentence-transformers)")
        print("- Proper IRIS database configuration (if using real storage)")
        print("- The example uses fallback embeddings if sentence-transformers is not available")


if __name__ == "__main__":
    main()