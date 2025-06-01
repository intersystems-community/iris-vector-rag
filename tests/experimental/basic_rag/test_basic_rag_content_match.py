"""
Test BasicRAG with queries that match actual database content.
This test uses topics we know exist in the database from benchmark results.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.experimental.basic_rag.pipeline_final import BasicRAGPipeline
from src.common.utils import get_embedding_func, get_llm_func
from src.common.iris_connector_jdbc import get_iris_connection


def test_basic_rag_with_matching_content():
    """Test BasicRAG with queries that should match actual database content."""
    
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
    
    # Test queries based on actual database content
    test_queries = [
        "What is olfactory perception?",
        "How do microRNAs regulate gene expression?",
        "What are the characteristics of honeybees?",
        "Explain the role of microRNA in biological processes",
        "What is known about olfactory receptors?",
        "How do honeybees communicate?",
        "What are the mechanisms of smell perception?",
        "Describe microRNA biogenesis",
        "What is the social structure of honeybee colonies?",
        "How does the olfactory system work?"
    ]
    
    print("\n" + "="*80)
    print("Testing BasicRAG with content-matching queries")
    print("="*80)
    
    successful_retrievals = 0
    total_queries = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}/{total_queries}: {query}")
        print("-" * 60)
        
        try:
            # Run the query
            result = pipeline.run(query)
            
            # Check if documents were retrieved
            retrieved_docs = result.get("retrieved_documents", [])
            num_docs = len(retrieved_docs)
            
            print(f"Retrieved documents: {num_docs}")
            
            if num_docs > 0:
                successful_retrievals += 1
                print("✓ Documents retrieved successfully")
                
                # Show first document preview
                first_doc = retrieved_docs[0]
                content_preview = first_doc.get('content', '')[:200] + "..."
                print(f"First document preview: {content_preview}")
                
                # Show answer preview
                answer = result.get("answer", "No answer generated")
                answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"Answer preview: {answer_preview}")
            else:
                print("✗ No documents retrieved")
                
        except Exception as e:
            print(f"✗ Error during query: {str(e)}")
            
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total queries: {total_queries}")
    print(f"Successful retrievals: {successful_retrievals}")
    print(f"Failed retrievals: {total_queries - successful_retrievals}")
    print(f"Success rate: {(successful_retrievals/total_queries)*100:.1f}%")
    
    # Assert that at least some queries retrieved documents
    assert successful_retrievals > 0, "No queries retrieved any documents!"
    
    # We expect at least 50% success rate with matching content
    assert successful_retrievals >= total_queries * 0.5, \
        f"Too many failed retrievals: {successful_retrievals}/{total_queries}"


def test_basic_rag_specific_topics():
    """Test BasicRAG with very specific topics from the database."""
    
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
    
    # Very specific queries based on benchmark results
    specific_queries = [
        "olfactory",  # Simple keyword
        "microRNA",   # Simple keyword
        "honeybee",   # Simple keyword
        "olfactory perception mechanisms",
        "microRNA regulation",
        "honeybee behavior"
    ]
    
    print("\n" + "="*80)
    print("Testing BasicRAG with specific keywords")
    print("="*80)
    
    for query in specific_queries:
        print(f"\nTesting query: '{query}'")
        
        try:
            result = pipeline.run(query)
            retrieved_docs = result.get("retrieved_documents", [])
            
            print(f"Documents retrieved: {len(retrieved_docs)}")
            
            if retrieved_docs:
                # Check if content actually contains relevant terms
                first_doc_content = retrieved_docs[0].get('content', '').lower()
                query_terms = query.lower().split()
                
                matching_terms = [term for term in query_terms if term in first_doc_content]
                print(f"Matching terms found in first document: {matching_terms}")
                
        except Exception as e:
            print(f"Error: {str(e)}")


def test_basic_rag_debug_retrieval():
    """Debug test to understand what's happening with retrieval."""
    
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
    
    # Single test query
    test_query = "olfactory perception"
    
    print("\n" + "="*80)
    print("DEBUG: BasicRAG Retrieval Process")
    print("="*80)
    print(f"Query: {test_query}")
    
    try:
        # Get query embedding
        query_embedding = embedding_func(test_query)
        print(f"Query embedding shape: {len(query_embedding)}")
        print(f"Query embedding sample: {query_embedding[:5]}...")
        
        # Try direct vector search
        print("\nAttempting direct vector search...")
        
        # Check if we can query the database at all
        cursor = iris_connection.cursor()
        try:
            
            # First, check if we have any documents
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            print(f"Total documents in database: {doc_count}")
            
            # Check if we have any embeddings
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            embedding_count = cursor.fetchone()[0]
            print(f"Documents with embeddings: {embedding_count}")
            
            # Try to retrieve a sample document
            cursor.execute("SELECT TOP 1 doc_id, title, text_content FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            sample = cursor.fetchone()
            if sample:
                print(f"\nSample document:")
                print(f"  ID: {sample[0]}")
                print(f"  Title: {sample[1][:100]}...")
                print(f"  Content preview: {sample[2][:200]}...")
        finally:
            cursor.close()
        
        # Now run the full pipeline
        print("\nRunning full pipeline...")
        result = pipeline.run(test_query)
        
        retrieved_docs = result.get("retrieved_documents", [])
        print(f"\nRetrieved {len(retrieved_docs)} documents")
        
        if retrieved_docs:
            print("\nFirst retrieved document:")
            first_doc = retrieved_docs[0]
            for key, value in first_doc.items():
                if key == 'content':
                    print(f"  {key}: {str(value)[:200]}...")
                else:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"Error during debug: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests directly
    print("Running BasicRAG content matching tests...")
    
    # Run each test function
    test_basic_rag_with_matching_content()
    test_basic_rag_specific_topics()
    test_basic_rag_debug_retrieval()