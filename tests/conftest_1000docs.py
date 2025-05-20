"""
Pytest configuration for running tests with 1000+ documents.

This conftest file provides fixtures to ensure all tests run with a minimum 
of 1000 documents, as required by project rules in .clinerules.
"""

import os
import sys
import pytest
import logging
import random
import time
from typing import Dict, Any, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the main conftest fixture to extend them
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.conftest import * # This imports fixtures from the main conftest
from common.iris_connector import get_mock_iris_connection # Explicit import for clarity

# Constants
MIN_DOCUMENT_COUNT = 1000

# Register custom pytest markers
def pytest_configure(config):
    """Register custom pytest markers for 1000 doc tests"""
    config.addinivalue_line("markers", "requires_1000_docs: mark test to require at least 1000 documents")

@pytest.fixture(scope="session")
def min_document_count():
    """Return the minimum document count required by .clinerules"""
    return MIN_DOCUMENT_COUNT

@pytest.fixture(scope="function")
def ensure_1000_documents(iris_connection_auto):
    """
    Fixture to ensure at least 1000 documents exist in the database.
    
    If fewer than 1000 documents exist, this fixture will generate synthetic
    ones to reach the minimum count. This satisfies the project requirement
    from .clinerules that tests must use at least 1000 documents.
    
    Returns:
        Database connection with at least 1000 documents
    """
    conn_from_auto = iris_connection_auto
    
    # Ensure we are using a compliant mock if a mock is provided
    # This is to guard against the 'MockDBConnection' issue.
    # A real connection (like SQLAlchemy's ConnectionFairy) would not be a MockIRISConnector.
    # A proper mock should be MockIRISConnector.
    from tests.mocks.db import MockIRISConnector
    # Crude check: if it's not a known real connection type (hard to check universally)
    # and not our good mock, then replace it with our good mock.
    # A more direct check might be `if type(conn_from_auto).__name__ == 'MockDBConnection':`
    # but we don't have its definition.
    # Let's check if it has a `cursor` method that returns something with `execute` and `close`.
    
    conn_is_compliant_mock = isinstance(conn_from_auto, MockIRISConnector)
    conn_is_likely_real_dbapi = hasattr(conn_from_auto, '_connection') # SQLAlchemy's ConnectionFairy has this
    
    if not conn_is_likely_real_dbapi and not conn_is_compliant_mock:
        logger.warning(f"iris_connection_auto returned a non-compliant mock type: {type(conn_from_auto)}. Forcing MockIRISConnector.")
        conn = get_mock_iris_connection() # from tests.conftest, which imports it from common.iris_connector
        if conn is None: # Should not happen if MockIRISConnector is importable
             pytest.fail("Failed to get MockIRISConnector as a fallback in ensure_1000_documents.")
    else:
        conn = conn_from_auto

    try:
        # Check current document count
        with conn.cursor() as cursor:
            try:
                cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
                current_count = cursor.fetchone()[0]
                logger.info(f"Current document count: {current_count}")
            except Exception as e:
                logger.warning(f"Error counting documents, creating table: {e}")
                # Create table if it doesn't exist, matching common/db_init.sql schema
                # for SourceDocuments (doc_id, text_content, embedding)
                # Using CLOB for text_content and embedding as per current main DDL.
                try:
                    cursor.execute("""
                        CREATE TABLE SourceDocuments (
                            doc_id VARCHAR(255) PRIMARY KEY,
                            text_content CLOB,
                            embedding CLOB
                        )
                    """)
                    logger.info("Created SourceDocuments table with schema (doc_id, text_content, embedding).")
                except Exception as create_table_e:
                    logger.error(f"Failed to create SourceDocuments table in ensure_1000_documents: {create_table_e}")
                    raise # Re-raise if table creation fails, as it's critical
                current_count = 0
            
            # If we already have enough documents, we're done
            if current_count >= MIN_DOCUMENT_COUNT:
                logger.info(f"✅ Found {current_count} documents (≥{MIN_DOCUMENT_COUNT} required)")
                return conn
            
            # Generate and add synthetic documents to reach the minimum
            docs_to_add = MIN_DOCUMENT_COUNT - current_count
            logger.info(f"Generating {docs_to_add} synthetic documents to reach {MIN_DOCUMENT_COUNT}...")
            
            # Generate topic-based documents for more realistic testing
            topics = ["diabetes", "insulin", "cancer", "heart disease", "hypertension", 
                      "obesity", "cholesterol", "asthma", "alzheimer", "parkinson"]
                      
            # Track time for performance logging
            start_time = time.time()
            
            # Generate and insert documents in batches for better performance
            batch_size = 50
            for i in range(0, docs_to_add, batch_size):
                batch_end = min(i + batch_size, docs_to_add)
                batch_docs = []
                
                for j in range(i, batch_end):
                    # Create document with medical topic
                    doc_id = f"synthetic_doc_{j:06d}"
                    doc_topics = random.sample(topics, k=min(3, len(topics)))
                    topic_text = " and ".join(doc_topics)
                    
                    # title = f"Synthetic Document {j:06d} about {topic_text}" # No title column in target schema
                    text_content = (
                        f"This is a synthetic document about {topic_text} for testing RAG systems. "
                        f"It contains medical information related to {doc_topics[0]} research. "
                        f"Document ID: {doc_id}. This document was generated to ensure compliance "
                        f"with the minimum document count requirement of {MIN_DOCUMENT_COUNT} documents. "
                        f"Content for {doc_id}." # Ensure content is somewhat unique
                    )
                    
                    # Create a random embedding vector (10 dimensions for mock, real pipeline uses 384)
                    # Stored as string representation of list for CLOB.
                    embedding_list = np.random.rand(10).tolist() 
                    embedding_str = str(embedding_list)
                    
                    # Add to batch (doc_id, text_content, embedding)
                    batch_docs.append((doc_id, text_content, embedding_str))
                
                # Insert the batch
                try:
                    # SQL matches the 3-column schema
                    cursor.executemany(
                        "INSERT INTO SourceDocuments (doc_id, text_content, embedding) VALUES (?, ?, ?)",
                        batch_docs
                    )
                    logger.info(f"Added {len(batch_docs)} documents (batch {i//batch_size + 1})")
                except Exception as e:
                    logger.error(f"Error inserting documents: {e}")
                    
            # Verify the final count
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            final_count_raw = cursor.fetchone()[0]
            final_count = int(final_count_raw) if final_count_raw is not None else 0
            
            elapsed_time = time.time() - start_time
            docs_per_second = docs_to_add / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"✅ Final document count: {final_count} documents")
            logger.info(f"Generated {docs_to_add} documents in {elapsed_time:.2f}s ({docs_per_second:.1f} docs/s)")
            
            # Verify we have enough documents
            assert final_count >= MIN_DOCUMENT_COUNT, (
                f"Failed to reach minimum document count: {final_count} < {MIN_DOCUMENT_COUNT}"
            )
            
            return conn
    
    except Exception as e:
        logger.error(f"Error ensuring minimum document count: {e}")
        raise
        
@pytest.fixture(scope="function")
def verify_document_count(ensure_1000_documents):
    """
    Fixture to verify document count requirement is met for each test.
    
    This provides the connection while verifying the document count is still
    sufficient (in case documents were deleted by a previous test).
    """
    conn = ensure_1000_documents
    
    # Verify document count at start of test
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count_raw = cursor.fetchone()[0]
        count = int(count_raw) if count_raw is not None else 0
        
        if count < MIN_DOCUMENT_COUNT:
            pytest.fail(f"Document count fell below minimum: {count} < {MIN_DOCUMENT_COUNT}")
        
        logger.info(f"✅ Verified document count for test: {count} documents")
    
    return conn

@pytest.fixture(scope="module")
def mock_functions():
    """
    Fixture to provide consistent mock functions for all RAG techniques.
    """
    # Mock embedding function that returns 10-dim vectors
    def mock_embedding_func(text):
        if isinstance(text, list):
            return [[random.random() for _ in range(10)] for _ in text]
        return [random.random() for _ in range(10)]
    
    # Mock LLM function that returns a mock answer
    def mock_llm_func(prompt):
        # Extract query from prompt if possible
        query = prompt.split("Question:")[-1].split("\n")[0].strip() if "Question:" in prompt else prompt[:30]
        return f"Mock answer about {query}"
    
    # Mock ColBERT query encoder for token-level embeddings
    def mock_colbert_query_encoder(text):
        tokens = text.split() if isinstance(text, str) else [text]
        tokens = tokens[:10]  # Limit tokens
        return [[random.random() for _ in range(10)] for _ in range(len(tokens))]
    
    # Mock web search function for CRAG
    def mock_web_search_func(query, num_results=3):
        return [f"Web search result {i+1} for {query}" for i in range(num_results)]
    
    return {
        "embedding_func": mock_embedding_func,
        "llm_func": mock_llm_func,
        "colbert_query_encoder": mock_colbert_query_encoder,
        "web_search_func": mock_web_search_func
    }

# Mark all tests that use verify_document_count with requires_1000_docs
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests that use the verify_document_count fixture"""
    for item in items:
        if "verify_document_count" in item.fixturenames:
            item.add_marker(pytest.mark.requires_1000_docs)
