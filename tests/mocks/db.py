"""
Database mock classes for testing.

This module provides mock implementations of IRIS database connections and cursors
that can be used in unit tests to avoid requiring a real database connection.
"""

from typing import Any, List, Tuple, Optional, Union
import json


class MockIRISCursor:
    """
    Mock implementation of an IRIS database cursor.
    
    Provides the same interface as a real IRIS cursor but with configurable
    return values for testing purposes.
    """
    
    def __init__(self):
        self.fetchall_results = []
        self.fetchone_results = []
        self.execute_calls = []
        self._fetchall_index = 0
        self._fetchone_index = 0
        self.closed = False
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> None:
        """Mock execute method that records the query and parameters."""
        self.execute_calls.append((query, params))
    
    def fetchall(self) -> List[Tuple]:
        """Mock fetchall that returns pre-configured results."""
        if self._fetchall_index < len(self.fetchall_results):
            result = self.fetchall_results[self._fetchall_index]
            self._fetchall_index += 1
            return result
        return []
    
    def fetchone(self) -> Optional[Tuple]:
        """Mock fetchone that returns pre-configured results."""
        if self._fetchone_index < len(self.fetchone_results):
            result = self.fetchone_results[self._fetchone_index]
            self._fetchone_index += 1
            return result
        return None
    
    def close(self) -> None:
        """Mock close method."""
        self.closed = True
    
    def set_fetchall_results(self, results: List[List[Tuple]]) -> None:
        """Configure the results that fetchall should return."""
        self.fetchall_results = results
        self._fetchall_index = 0
    
    def set_fetchone_results(self, results: List[Tuple]) -> None:
        """Configure the results that fetchone should return."""
        self.fetchone_results = results
        self._fetchone_index = 0


class MockIRISConnector:
    """
    Mock implementation of an IRIS database connector.
    
    Provides the same interface as a real IRIS connector but with configurable
    behavior for testing purposes.
    """
    
    def __init__(self):
        self.cursors = []
        self.closed = False
        self.committed = False
        self.rolled_back = False
    
    def cursor(self) -> MockIRISCursor:
        """Create and return a new mock cursor."""
        cursor = MockIRISCursor()
        self.cursors.append(cursor)
        return cursor
    
    def close(self) -> None:
        """Mock close method."""
        self.closed = True
        for cursor in self.cursors:
            cursor.close()
    
    def commit(self) -> None:
        """Mock commit method."""
        self.committed = True
    
    def rollback(self) -> None:
        """Mock rollback method."""
        self.rolled_back = True
    
    def configure_cursor_results(self, fetchall_results: List[List[Tuple]], 
                                fetchone_results: List[Tuple] = None) -> None:
        """
        Configure the results that cursors created by this connector should return.
        
        Args:
            fetchall_results: List of result sets for fetchall calls
            fetchone_results: List of single results for fetchone calls
        """
        # Configure the next cursor that will be created
        self._next_fetchall_results = fetchall_results
        self._next_fetchone_results = fetchone_results or []
    
    def cursor_with_results(self, fetchall_results: List[List[Tuple]], 
                           fetchone_results: List[Tuple] = None) -> MockIRISCursor:
        """
        Create a cursor with pre-configured results.
        
        Args:
            fetchall_results: List of result sets for fetchall calls
            fetchone_results: List of single results for fetchone calls
            
        Returns:
            MockIRISCursor with configured results
        """
        cursor = self.cursor()
        cursor.set_fetchall_results(fetchall_results)
        if fetchone_results:
            cursor.set_fetchone_results(fetchone_results)
        return cursor


def create_colbert_mock_connector() -> MockIRISConnector:
    """
    Create a mock connector specifically configured for ColBERT tests.
    
    Returns:
        MockIRISConnector with ColBERT-specific test data
    """
    connector = MockIRISConnector()
    
    # Mock data for ColBERT tests
    mock_doc_ids_data = [("doc_c1",), ("doc_c2",), ("doc_c3",), ("doc_c4",), ("doc_c5",)]
    token_embeddings_for_docs = [
        [(json.dumps([0.11]*10),), (json.dumps([0.12]*10),)],  # doc_c1
        [(json.dumps([0.21]*10),), (json.dumps([0.22]*10),)],  # doc_c2
        [(json.dumps([0.31]*10),), (json.dumps([0.32]*10),)],  # doc_c3
        [(json.dumps([0.41]*10),), (json.dumps([0.42]*10),)],  # doc_c4
        [(json.dumps([0.51]*10),), (json.dumps([0.52]*10),)],  # doc_c5
    ]
    content_for_docs = [
        ("Content for doc_c1.",), ("Content for doc_c2.",), ("Content for doc_c3.",),
        ("Content for doc_c4.",), ("Content for doc_c5.",),
    ]
    
    # Combined sequence for fetchall calls:
    # 1st call: all doc_ids
    # 2nd-6th calls: token embeddings for each of the 5 docs
    combined_fetchall_data = [mock_doc_ids_data] + token_embeddings_for_docs
    
    connector.configure_cursor_results(combined_fetchall_data, content_for_docs)
    
    return connector