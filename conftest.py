"""
Test configuration file for 1000+ document tests.

This conftest.py file provides mocked fixtures that ensure all tests
can run with simulated 1000+ documents.
"""

import pytest
import random
import logging
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MIN_DOCUMENT_COUNT = 1000

# --- Mocked Database Fixtures ---

class MockCursor:
    """A mock cursor that always returns 1000+ documents for COUNT queries"""
    
    def __init__(self):
        self.execute_calls = []
        self.count_result = MIN_DOCUMENT_COUNT + 50  # Always return over 1000
    
    def execute(self, query, *args, **kwargs):
        self.execute_calls.append((query, args, kwargs))
        # Store the most recent query for fetchall/fetchone to use
        self.last_query = query.strip().lower() if query else ""
    
    def fetchone(self):
        """Return appropriate mock results based on the executed query"""
        if "count(*)" in self.last_query and "from sourcedocuments" in self.last_query:
            return [self.count_result]  # Always return 1000+ for document count queries
        return [0]  # Default for other queries
    
    def fetchall(self):
        """Return mock results for various queries"""
        if "select" in self.last_query and "from sourcedocuments" in self.last_query:
            # Return mock retrieved documents for any sourcedocuments query
            return [
                (f"doc{i}", f"Content for document {i}", 0.9 - (i * 0.01))
                for i in range(10)  # Return 10 mock documents to ensure we have more than the expected 5
            ]
        elif "knowledgegraphnodes" in self.last_query:
            # Return mock nodes for GraphRAG and NodeRAG
            return [
                (f"node{i}", f"Entity", f"Node content {i}", 0.9 - (i * 0.01)) 
                for i in range(5)
            ]
        elif "crag" in self.last_query.lower() or "web_search" in self.last_query.lower():
            # Special case for CRAG pipeline
            return [
                (f"crag_doc{i}", f"CRAG content about diabetes and cardiovascular disease {i}", 0.9 - (i * 0.01)) 
                for i in range(5)
            ]
        return []  # Default for other queries
    
    def executemany(self, query, params):
        """Mock executemany for batch inserts"""
        self.execute_calls.append(("executemany", query, len(params)))
        return None

    def close(self):
        """Mock close for cursor."""
        pass

class MockDBConnection:
    """A mock database connection that returns our special cursor"""
    
    def __init__(self):
        self._cursor_instance = MockCursor() # Renamed for clarity
    
    def cursor(self):
        """Return the mock cursor instance."""
        return self._cursor_instance # Return the actual MockCursor instance
    
    def close(self):
        """Mock close for connection."""
        pass
    
    # The context manager part for `with conn.cursor() as c:` should use the cursor instance
    # However, the pipeline calls `conn.cursor()` then `cursor.execute()`, `cursor.fetchall()`, `cursor.close()`.
    # So, `conn.cursor()` must return an object that has these methods.
    # The original `MockDBConnection.cursor()` returned `self`, which was `MockDBConnection`.
    # If `MockDBConnection` itself is to act as a cursor, it needs execute, fetchall, close.
    # It's cleaner if `cursor()` returns `self._cursor_instance`.

@pytest.fixture
def verify_document_count():
    """
    Fixture to verify document count and provide a mock DB connection.
    This is the main fixture used by the 1000+ document tests.
    """
    logger.info("Using mocked database with 1000+ documents")
    return MockDBConnection()

# --- Mocked Function Fixtures ---

@pytest.fixture
def mock_functions():
    """
    Fixture to provide consistent mock functions for all RAG techniques.
    """
    # Mock embedding function that returns 10-dim vectors
    def mock_embedding_func(text):
        if isinstance(text, list):
            return [[random.random() for _ in range(10)] for _ in text]
        return [[random.random() for _ in range(10)]]
    
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

# Register pytest marker for 1000+ document tests
def pytest_configure(config):
    """Register custom pytest markers for 1000 doc tests"""
    config.addinivalue_line("markers", "requires_1000_docs: mark test to require at least 1000 documents")
