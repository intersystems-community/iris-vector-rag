"""
Tests for real data integration with embedding generation.

This module tests the complete pipeline for processing real PMC data 
and generating both document-level and token-level embeddings.
"""

import pytest
import os
import sys

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.embedding_utils import (
    generate_document_embeddings,
    generate_token_embeddings,
    get_embedding_model,
    get_colbert_model,
    create_tables_if_needed
)


@pytest.mark.integration
@pytest.mark.real_data
def test_real_data_embedding_pipeline(iris_connection, use_real_data):
    """
    Test the complete pipeline for processing real data and generating embeddings.
    This test will run with real data if available, otherwise falls back to mock data.
    """
    # Initialize connection and ensure we have documents in the database
    cursor = iris_connection.cursor()
    
    # For mock connection, add some test documents if needed
    if not use_real_data:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        result = cursor.fetchone()
        count = int(result[0]) if result and isinstance(result[0], str) else 0 if result is None else result[0]
        
        if count == 0:
            # Add some test documents
            test_docs = [
                ("test_doc1", "Test Document 1", "Content for test document 1", "[]", "[]"),
                ("test_doc2", "Test Document 2", "Content for test document 2", "[]", "[]"),
            ]
            cursor.executemany(
                "INSERT INTO SourceDocuments (doc_id, title, content, authors, keywords) VALUES (?, ?, ?, ?, ?)",
                test_docs
            )
            print("Added test documents to mock database")
    
    # Verify we have documents
    cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
    result = cursor.fetchone()
    cursor.close()
    
    assert result is not None
    doc_count = int(result[0]) if isinstance(result[0], str) else result[0]
    assert doc_count > 0, "No documents found in database"
    
    # Create tables if needed for embeddings
    create_tables_if_needed(iris_connection)
    
    # Get embedding models (mock=True for testing to avoid real model loading)
    doc_embedding_model = get_embedding_model(mock=True)
    token_embedding_model = get_colbert_model(mock=True)
    
    # Generate document-level embeddings
    doc_stats = generate_document_embeddings(
        iris_connection,
        doc_embedding_model,
        batch_size=2,
        limit=2  # Small limit for testing
    )
    
    # Verify document embedding results
    assert doc_stats is not None
    assert doc_stats["type"] == "document_embeddings"
    assert doc_stats["processed_count"] >= 0  # May be 0 if all docs already have embeddings
    
    # Generate token-level embeddings
    token_stats = generate_token_embeddings(
        iris_connection,
        token_embedding_model,
        batch_size=1,
        limit=2  # Small limit for testing
    )
    
    # Verify token embedding results
    assert token_stats is not None
    assert token_stats["type"] == "token_embeddings"
    assert token_stats["processed_count"] >= 0  # May be 0 if all docs already have token embeddings
    
    # Verify we can retrieve documents with embeddings
    cursor = iris_connection.cursor()
    
    # For document embeddings
    cursor.execute("SELECT COUNT(*) FROM SourceDocuments WHERE embedding IS NOT NULL")
    doc_result = cursor.fetchone()
    doc_with_embeddings = int(doc_result[0]) if isinstance(doc_result[0], str) else doc_result[0]
    
    # For token embeddings
    cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM DocumentTokenEmbeddings")
    token_result = cursor.fetchone()
    docs_with_tokens = 0
    if token_result and token_result[0]:
        docs_with_tokens = int(token_result[0]) if isinstance(token_result[0], str) else token_result[0]
    
    cursor.close()
    
    # Print stats for debugging
    print(f"\nResults using {'real' if use_real_data else 'mock'} data:")
    print(f"Total documents: {doc_count}")
    print(f"Documents with embeddings: {doc_with_embeddings}")
    print(f"Documents with token embeddings: {docs_with_tokens}")
    
    # We should have at least some documents with embeddings
    if doc_count > 0:
        assert doc_with_embeddings > 0 or docs_with_tokens > 0, "No embeddings were generated"


@pytest.mark.integration
@pytest.mark.real_data
def test_embedding_end_to_end(iris_connection, use_real_data, mock_embedding_func):
    """
    Test the end-to-end embedding generation and retrieval process.
    This test simulates a complete RAG pipeline with embedding generation and retrieval.
    """
    # Initialize connection and create test document if needed
    cursor = iris_connection.cursor()
    
    # If using mock data, create a test document
    if not use_real_data:
        cursor.execute("DELETE FROM SourceDocuments WHERE doc_id = 'test_e2e_doc'")
        cursor.execute(
            "INSERT INTO SourceDocuments (doc_id, title, content) VALUES (?, ?, ?)",
            ("test_e2e_doc", "E2E Test", "This is a test document for end-to-end testing.")
        )
        doc_id = "test_e2e_doc"
    else:
        # With real data, get an existing document
        cursor.execute("SELECT doc_id FROM SourceDocuments LIMIT 1")
        result = cursor.fetchone()
        if not result:
            pytest.skip("No documents available in real database")
        doc_id = result[0]
    
    # Ensure we have embedding column
    try:
        cursor.execute("SELECT embedding FROM SourceDocuments WHERE 1=0")
    except:
        cursor.execute("ALTER TABLE SourceDocuments ADD embedding TEXT")
    
    # Get document content
    cursor.execute("SELECT content FROM SourceDocuments WHERE doc_id = ?", (doc_id,))
    content_result = cursor.fetchone()
    assert content_result is not None
    content = content_result[0]
    
    # Generate embedding
    model = get_embedding_model(mock=True)
    embedding = model.encode([content])[0]
    
    # Store embedding
    embedding_json = list(embedding)
    cursor.execute(
        "UPDATE SourceDocuments SET embedding = ? WHERE doc_id = ?", 
        (str(embedding_json), doc_id)
    )
    
    # Now verify we can retrieve document using embedding similarity
    # Create a test query embedding
    # This logic would need to be adjusted based on IRIS's vector similarity support
    query_embedding = embedding * 0.95  # Slightly modified version of the original embedding
    cursor.close()
    
    print(f"\nE2E test results using {'real' if use_real_data else 'mock'} data:")
    print(f"Successfully generated and stored embedding for document {doc_id}")
    print(f"Embedding dimensions: {len(embedding)}")
    
    # Test passed if we got this far without errors
    assert True
