"""
Standardized mock implementations for database components.
These mocks provide consistent behavior for testing database interactions
without requiring a real database connection.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging # Added import

logger = logging.getLogger(__name__) # Added logger definition


class MockIRISCursor:
    """Standardized mock cursor for testing without a real database connection."""
    
    def __init__(self):
        """Initialize with empty storage."""
        self.stored_docs = {}  # doc_id -> document data
        self.stored_token_embeddings = {}  # doc_id -> list of token embeddings
        self.stored_kg_nodes = {}  # node_id -> node data 
        self.stored_kg_edges = []  # list of edge tuples
        self.results = []  # Current query results
        self.last_sql = ""  # Last executed SQL
        self.last_params = None  # Last parameters
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        pass
    
    def execute(self, sql: str, params=None):
        """Execute a SQL query on mock data."""
        self.last_sql = sql
        self.last_params = params
        
        print(f"Mock SQL: {sql[:80]}...")
        
        # Special handling for embedding tests
        if "embedding IS NULL" in sql:
            # For document embedding tests
            self.results = [
                ("doc1", "Test Content 1"), 
                ("doc2", "Test Content 2")
            ]
            return self
        elif "LEFT JOIN" in sql and "DocumentTokenEmbeddings" in sql:
            # For token embedding tests
            self.results = [
                ("doc1", "Test Content 1"), 
                ("doc2", "Test Content 2")
            ]
            return self
        elif "WHERE doc_id = ?" in sql and params and params[0] == "test_e2e_doc" and "content" in sql.lower():
            # Special handling for the end-to-end test
            self.results = [("This is a test document for end-to-end testing.",)]
            return self

        # Vector similarity search (prioritize this check)
        if "VECTOR_COSINE_SIMILARITY" in sql or "ORDER BY" in sql and "FROM SourceDocuments" in sql : # Made more specific
            # Mock vector search results
            if self.stored_docs:
                # If there are stored docs, try to return them in the expected format
                self.results = [
                    (doc_id, data.get("content", f"Mock content for {doc_id}"), data.get("score", 0.85))
                    for doc_id, data in list(self.stored_docs.items())[:3] # Return up to 3
                ]
            else:
                # Default mock results if no specific data stored in the mock
                self.results = [
                    ("mock_retrieved_doc1", "Content for mock retrieved doc 1", 0.91),
                    ("mock_retrieved_doc2", "Content for mock retrieved doc 2", 0.89),
                    ("mock_retrieved_doc3", "Content for mock retrieved doc 3", 0.87)
                ]
        # Document retrieval (general, non-similarity)
        elif "FROM SourceDocuments" in sql: # Changed from if to elif
            # Handle COUNT queries specifically
            if "COUNT(*)" in sql and "embedding IS NOT NULL" in sql:
                # Return the count of documents with embeddings - for testing, just return 2
                self.results = [("2",)]
            elif "COUNT(*)" in sql:
                # Return the count of stored documents
                self.results = [(str(len(self.stored_docs)),)]
            elif params and "doc_id" in sql:
                # Filter by specific document ID
                doc_id = params[0] if isinstance(params, (list, tuple)) else params
                if doc_id in self.stored_docs:
                    self.results = [(doc_id, self.stored_docs[doc_id].get("content", ""))]
                else:
                    self.results = []
            else:
                # Return all documents
                self.results = [(doc_id, doc.get("content", "")) 
                               for doc_id, doc in self.stored_docs.items()]
        
        # Token embeddings query
        elif "FROM DocumentTokenEmbeddings" in sql:
            # Special case for counting distinct doc_ids
            if "COUNT(DISTINCT doc_id)" in sql:
                # Return count of documents with token embeddings
                self.results = [(str(len(self.stored_token_embeddings)),)]
                return self
                
            doc_id = params[0] if params else None
            if doc_id and doc_id in self.stored_token_embeddings:
                # Return token embeddings for specific document
                self.results = [(embed.get("embedding", "[]"), embed.get("metadata", "{}")) 
                               for embed in self.stored_token_embeddings[doc_id]]
            else:
                # Mock data for testing
                np.random.seed(42)
                self.results = [
                    (str(np.random.randn(10).tolist()), 
                     "{'compressed': False, 'scale_factor': 0.1, 'bits': 4}")
                    for _ in range(5)
                ]
        
        # Vector similarity search
        elif "VECTOR_COSINE_SIMILARITY" in sql or "ORDER BY" in sql:
            # Mock vector search results
            # Get existing doc IDs if available, otherwise use doc1, doc2, doc3
            if self.stored_docs:
                # If there are stored docs, try to return them in the expected format
                self.results = [
                    (doc_id, data.get("content", f"Mock content for {doc_id}"), data.get("score", 0.85))
                    for doc_id, data in list(self.stored_docs.items())[:3] # Return up to 3
                ]
            else:
                # Default mock results if no specific data stored in the mock
                self.results = [
                    ("mock_retrieved_doc1", "Content for mock retrieved doc 1", 0.91),
                    ("mock_retrieved_doc2", "Content for mock retrieved doc 2", 0.89),
                    ("mock_retrieved_doc3", "Content for mock retrieved doc 3", 0.87)
                ]
        
        # KG nodes query
        elif "FROM KnowledgeGraphNodes" in sql:
            self.results = [(node_id, data.get("type", ""), data.get("name", ""),
                            data.get("description", ""), data.get("metadata", "{}"))
                           for node_id, data in self.stored_kg_nodes.items()]
        
        # KG edges query
        elif "FROM KnowledgeGraphEdges" in sql:
            self.results = [(i, edge[0], edge[1], edge[2], edge[3]) 
                           for i, edge in enumerate(self.stored_kg_edges)]
        
        # Other queries return empty by default
        else:
            self.results = []
        
        return self
    
    def fetchall(self):
        """Return all results from the last query."""
        logger.info(f"MockIRISCursor: fetchall() called. Returning {len(self.results)} results: {self.results[:2]}...") # Added logger
        return self.results
    
    def fetchone(self):
        """Return first result from the last query."""
        if self.results:
            return self.results[0]
        return None
    
    def executemany(self, sql: str, param_list: List):
        """Execute a batch operation."""
        self.last_sql = sql
        print(f"Mock batch SQL: {sql[:50]}... ({len(param_list)} rows)")
        
        # Store documents
        if "INSERT INTO SourceDocuments" in sql:
            for params in param_list:
                doc_id = params[0]
                content = params[1]
                embedding = params[2] if len(params) > 2 else None
                self.stored_docs[doc_id] = {
                    "content": content,
                    "embedding": embedding
                }
        
        # Store token embeddings
        elif "INSERT INTO DocumentTokenEmbeddings" in sql:
            for params in param_list:
                doc_id = params[0]
                token_idx = params[1]
                token_text = params[2]
                token_embedding = params[3]
                metadata = params[4] if len(params) > 4 else "{}"
                
                if doc_id not in self.stored_token_embeddings:
                    self.stored_token_embeddings[doc_id] = []
                
                self.stored_token_embeddings[doc_id].append({
                    "idx": token_idx,
                    "text": token_text,
                    "embedding": token_embedding,
                    "metadata": metadata
                })
        
        # Store KG nodes
        elif "INSERT INTO KnowledgeGraphNodes" in sql:
            for params in param_list:
                node_id = params[0]
                node_type = params[1]
                node_name = params[2]
                description = params[3]
                metadata = params[4] if len(params) > 4 else "{}"
                embedding = params[5] if len(params) > 5 else None
                
                self.stored_kg_nodes[node_id] = {
                    "type": node_type,
                    "name": node_name,
                    "description": description,
                    "metadata": metadata,
                    "embedding": embedding
                }
        
        # Store KG edges
        elif "INSERT INTO KnowledgeGraphEdges" in sql:
            for params in param_list:
                source_id = params[0]
                target_id = params[1]
                rel_type = params[2]
                weight = params[3] if len(params) > 3 else 1.0
                properties = params[4] if len(params) > 4 else "{}"
                
                self.stored_kg_edges.append((source_id, target_id, rel_type, weight, properties))
        
        return self
    
    def close(self):
        """Close the cursor (no-op for mock)."""
        pass


class MockIRISConnector:
    """Standardized mock IRIS connector that properly supports context manager protocol."""
    
    def __init__(self):
        """Initialize with a cursor."""
        self._cursor = MockIRISCursor()
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        pass
    
    def cursor(self):
        """Return the cursor."""
        return self._cursor
    
    def close(self):
        """Close the connection (no-op for mock)."""
        pass
    
    def commit(self):
        """Commit transaction (no-op for mock)."""
        pass
    
    def rollback(self):
        """Rollback transaction (no-op for mock)."""
        pass
