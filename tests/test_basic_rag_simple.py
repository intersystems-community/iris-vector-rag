"""
Simple test to verify BasicRAG retrieval works with actual database content.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from basic_rag.pipeline import BasicRAGPipeline
from common.utils import get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection


def test_basic_rag_retrieval():
    """Test that BasicRAG can retrieve documents for relevant queries."""
    
    # Initialize components
    iris_connection = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Initialize BasicRAG pipeline
    pipeline = BasicRAGPipeline(
        iris_connector=iris_connection,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test queries that should match database content
    test_cases = [
        ("olfactory perception", "olfactory"),
        ("microRNA regulation", "microRNA"),
        ("honeybee behavior", "honeybee"),
        ("smell receptors", "smell OR olfactory"),
        ("gene expression regulation", "gene OR expression"),
    ]
    
    print("\n" + "="*80)
    print("BasicRAG Simple Retrieval Test")
    print("="*80)
    
    results = []
    
    for query, expected_terms in test_cases:
        print(f"\nTesting query: '{query}'")
        print(f"Expected terms: {expected_terms}")
        print("-" * 40)
        
        try:
            # Run the pipeline
            result = pipeline.run(query)
            
            # Check basic results
            retrieved_docs = result.get("retrieved_documents", [])
            answer = result.get("answer", "")
            
            success = len(retrieved_docs) > 0
            
            print(f"✓ Documents retrieved: {len(retrieved_docs)}")
            print(f"✓ Answer generated: {'Yes' if answer and answer != 'I could not find specific information to answer your question.' else 'No'}")
            
            if success:
                print(f"✓ SUCCESS: Retrieved {len(retrieved_docs)} documents")
                # Just verify we have Document objects
                first_doc = retrieved_docs[0]
                print(f"  First document type: {type(first_doc).__name__}")
                print(f"  Has content: {'Yes' if hasattr(first_doc, 'content') else 'No'}")
                print(f"  Has score: {'Yes' if hasattr(first_doc, 'score') else 'No'}")
            else:
                print("✗ FAILED: No documents retrieved")
            
            results.append({
                'query': query,
                'success': success,
                'num_docs': len(retrieved_docs),
                'has_answer': bool(answer and answer != 'I could not find specific information to answer your question.')
            })
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append({
                'query': query,
                'success': False,
                'num_docs': 0,
                'has_answer': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Total queries tested: {total}")
    print(f"Successful retrievals: {successful}")
    print(f"Failed retrievals: {total - successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    
    # Detailed results
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{status} {r['query']}: {r['num_docs']} docs, answer: {r['has_answer']}")
        if 'error' in r:
            print(f"  Error: {r['error']}")
    
    # Overall assessment
    print("\n" + "="*80)
    if successful > 0:
        print("✓ BasicRAG is working! It can retrieve documents from the database.")
        print(f"  Success rate: {(successful/total)*100:.1f}%")
    else:
        print("✗ BasicRAG is not retrieving any documents.")
    
    return successful > 0


if __name__ == "__main__":
    print("Running BasicRAG simple retrieval test...")
    success = test_basic_rag_retrieval()
    exit(0 if success else 1)