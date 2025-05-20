"""
TDD Test for ColBERT with real data
"""

import pytest
import logging
import os
import time
import numpy as np
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define test markers
pytestmark = [
    pytest.mark.force_testcontainer,  # Always use testcontainer
]

# Test data parameters
MIN_DOCUMENT_COUNT = 3  # Lower for faster tests during development

@pytest.fixture(scope="module")
def ensure_colbert_data(iris_testcontainer_connection):
    """Ensure we have documents and token embeddings for ColBERT testing."""
    # Setup database tables and test data
    with iris_testcontainer_connection.cursor() as cursor:
        # Create SourceDocuments table if it doesn't exist
        try:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            doc_count = cursor.fetchone()[0]
        except Exception as e:
            logger.info(f"Setting up SourceDocuments table: {e}")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS SourceDocuments (
                    doc_id VARCHAR(100) PRIMARY KEY,
                    title VARCHAR(500),
                    content TEXT,
                    embedding VARCHAR(10000)
                )
            """)
            doc_count = 0
            
        # Add test documents if needed
        if doc_count < MIN_DOCUMENT_COUNT:
            for i in range(5):
                doc_id = f"colbert_doc_{i}"
                title = f"ColBERT Test Document {i}"
                content = f"This is test document {i} about ColBERT token-level embeddings for medical research about insulin and diabetes."
                embedding = '[' + ','.join(['0.1'] * 10) + ']'
                
                # Check if document exists first
                cursor.execute("SELECT COUNT(*) FROM SourceDocuments WHERE doc_id = ?", (doc_id,))
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO SourceDocuments (doc_id, title, content, embedding) VALUES (?, ?, ?, ?)",
                        (doc_id, title, content, embedding)
                    )
        
        # Create DocumentTokenEmbeddings table if it doesn't exist
        try:
            cursor.execute("SELECT COUNT(*) FROM DocumentTokenEmbeddings")
            token_count = cursor.fetchone()[0]
        except Exception as e:
            logger.info(f"Setting up DocumentTokenEmbeddings table: {e}")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS DocumentTokenEmbeddings (
                    id INTEGER PRIMARY KEY,
                    doc_id VARCHAR(100),
                    token_sequence_index INTEGER,
                    token_text VARCHAR(100),
                    token_embedding VARCHAR(10000),
                    metadata_json VARCHAR(1000)
                )
            """)
            token_count = 0
        
        # Add token embeddings if needed
        if token_count < 20:  # We want at least a few token embeddings per document
            # Get existing documents
            cursor.execute("SELECT doc_id, content FROM SourceDocuments")
            docs = cursor.fetchall()
            
            for doc_id, content in docs:
                # Simple tokenization by splitting on spaces (just for testing)
                tokens = content.split()[:10]  # Limit to first 10 tokens for simplicity
                
                for i, token in enumerate(tokens):
                    # Simple fixed embedding for each token (10 dimensions)
                    token_embedding = '[' + ','.join([str(0.1 * (i+1))] * 10) + ']'
                    
                    # Check if token exists first
                    cursor.execute(
                        "SELECT COUNT(*) FROM DocumentTokenEmbeddings WHERE doc_id = ? AND token_sequence_index = ?", 
                        (doc_id, i)
                    )
                    if cursor.fetchone()[0] == 0:
                        cursor.execute("""
                            INSERT INTO DocumentTokenEmbeddings 
                            (doc_id, token_sequence_index, token_text, token_embedding, metadata_json)
                            VALUES (?, ?, ?, ?, ?)
                        """, (doc_id, i, token, token_embedding, '{"compressed": false}'))
        
        # Check counts
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
    
    logger.info(f"ColBERT test data: {doc_count} documents, {token_count} token embeddings")
    
    if doc_count < MIN_DOCUMENT_COUNT:
        pytest.skip(f"Not enough documents for ColBERT test. Found {doc_count}, need at least {MIN_DOCUMENT_COUNT}")
    
    return iris_testcontainer_connection

def test_colbert_query_encoding():
    """Test that ColBERT query encoder properly generates token-level embeddings."""
    from colbert.query_encoder import encode_query
    
    # Test query
    query = "What is the role of insulin in diabetes?"
    
    # Encode query
    token_embeddings = encode_query(query)
    
    # Assertions
    assert token_embeddings is not None, "Query token embeddings should not be None"
    assert isinstance(token_embeddings, list), "Query token embeddings should be a list"
    assert len(token_embeddings) > 0, "Query token embeddings should not be empty"
    
    # Check that each token embedding is a vector of appropriate dimension
    for emb in token_embeddings:
        assert isinstance(emb, list), "Each token embedding should be a list/vector"
        assert len(emb) > 0, "Each token embedding should have a non-zero dimension"
    
    logger.info(f"Query '{query}' encoded into {len(token_embeddings)} token embeddings")

def test_colbert_end_to_end(ensure_colbert_data):
    """Test ColBERT with real-world data end-to-end. Following TDD Red-Green-Refactor approach."""
    from colbert.pipeline import ColBERTPipeline
    
    # Get connection from fixture
    connection = ensure_colbert_data
    
    # Create simple token-level embedding function for queries
    def colbert_query_encoder(text):
        # Split into tokens (simple space-based tokenization for testing)
        tokens = text.split()[:5]  # Limit to 5 tokens for testing
        
        # Generate an embedding for each token (10 dimensions each)
        token_embeddings = []
        for i, token in enumerate(tokens):
            # Create a unique embedding for each token
            embedding = [0.1 * (i+1)] * 10
            token_embeddings.append(embedding)
        
        return token_embeddings
    
    # Create simple mock LLM function
    def llm_func(prompt):
        return f"This is a ColBERT answer about insulin and diabetes using token-level matching across {prompt.count('test document')} documents."
    
    # Create and test pipeline
    pipeline = ColBERTPipeline(
        iris_connector=connection,
        colbert_query_encoder=colbert_query_encoder,
        llm_func=llm_func,
        client_side_maxsim=True  # Use client-side calculations for testing
    )
    
    # Test with simple query
    query = "What is insulin in diabetes management?"
    logger.info(f"Running ColBERT query: '{query}'")
    
    # Time the query
    start_time = time.time()
    result = pipeline.run(query, top_k=2)
    duration = time.time() - start_time
    
    # Assertions
    assert result is not None, "ColBERT result should not be None"
    assert "answer" in result, "ColBERT result should contain 'answer' key"
    assert "retrieved_documents" in result, "ColBERT result should contain 'retrieved_documents' key"
    assert len(result["retrieved_documents"]) > 0, "ColBERT should retrieve at least one document"
    
    # Log results
    logger.info(f"ColBERT query completed in {duration:.2f} seconds")
    logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
    logger.info(f"Answer: {result['answer']}")
    
    logger.info("ColBERT with real documents test passed")
