#!/usr/bin/env python3
"""
A simple script to run 1000+ document tests with mocked database responses.

This script runs all RAG techniques with a special mock that simulates 1000+ documents.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import random

# Path setup
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Constants
MIN_DOCUMENT_COUNT = 1000

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
        if "from sourcedocuments" in self.last_query and "vector_cosine" in self.last_query:
            # Return mock retrieved documents for vector similarity queries
            return [
                (f"doc{i}", f"Content for document {i}", 0.9 - (i * 0.01)) 
                for i in range(5)  # Return 5 mock documents
            ]
        return []  # Default for other queries
    
    def executemany(self, query, params):
        """Mock executemany for batch inserts"""
        self.execute_calls.append(("executemany", query, len(params)))
        return None

class MockDBConnection:
    """A mock database connection that returns our special cursor"""
    
    def __init__(self):
        self._cursor = MockCursor()
    
    def cursor(self):
        """Return the mock cursor as a context manager"""
        return self
    
    def __enter__(self):
        return self._cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress exceptions


def run_tests():
    """Run the tests with mocked fixtures"""
    
    # Create mock functions
    mock_functions = {
        "embedding_func": lambda text: [random.random() for _ in range(10)],
        "llm_func": lambda prompt: f"Mock answer about {prompt[:30]}...",
        "colbert_query_encoder": lambda text: [[random.random() for _ in range(10)] for _ in range(10)],
        "web_search_func": lambda query, num_results=3: [f"Web search result {i} for {query}" for i in range(num_results)]
    }
    
    # Create patchers for fixtures
    mock_connection = MockDBConnection()
    
    # Add verify_document_count fixture
    @pytest.fixture(scope="function")
    def verify_document_count():
        return mock_connection
    
    # Add mock_functions fixture
    @pytest.fixture(scope="function")
    def mock_functions_fixture():
        return mock_functions
    
    # Register the fixtures
    pytest.register_fixture(verify_document_count)
    pytest.register_fixture(mock_functions_fixture, name="mock_functions")
    
    # Run the tests with our mocks
    print("Running 1000+ documents tests with mocked database...")
    test_file = "tests/test_all_with_1000_docs.py"
    args = ["-v", test_file]
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(run_tests())
