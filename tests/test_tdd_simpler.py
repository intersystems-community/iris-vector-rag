"""
Simplified TDD tests for all RAG techniques
"""

import pytest
import logging
import time
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockIRISCursor:
    """Mock cursor for testing"""
    def __init__(self):
        self.queries = []
        self.data = {
            "SourceDocuments": [
                {"doc_id": "doc1", "content": "This is about diabetes and insulin."},
                {"doc_id": "doc2", "content": "Information about cancer treatment."},
                {"doc_id": "doc3", "content": "Vaccines protect against diseases."}
            ],
            "DocumentTokenEmbeddings": {
                "doc1": [
                    {"token_id": 1, "token_text": "diabetes", "token_embedding": "[0.1, 0.2, 0.3]", "metadata_json": "{}"},
                    {"token_id": 2, "token_text": "insulin", "token_embedding": "[0.2, 0.3, 0.4]", "metadata_json": "{}"}
                ],
                "doc2": [
                    {"token_id": 3, "token_text": "cancer", "token_embedding": "[0.3, 0.4, 0.5]", "metadata_json": "{}"},
                    {"token_id": 4, "token_text": "treatment", "token_embedding": "[0.4, 0.5, 0.6]", "metadata_json": "{}"}
                ],
                "doc3": [
                    {"token_id": 5, "token_text": "vaccines", "token_embedding": "[0.5, 0.6, 0.7]", "metadata_json": "{}"},
                    {"token_id": 6, "token_text": "diseases", "token_embedding": "[0.6, 0.7, 0.8]", "metadata_json": "{}"}
                ]
            }
        }
    
    def execute(self, query, params=None):
        self.queries.append((query, params))
        return self
    
    def fetchall(self):
        # Return mock data based on the last query
        last_query = self.queries[-1][0].lower() if self.queries else ""
        
        # For document retrieval
        if "select" in last_query and "sourcedocuments" in last_query:
            # Include a mock score (0.95) for each document
            return [(d["doc_id"], d["content"], 0.95) for d in self.data["SourceDocuments"]]
        
        # For ColBERT token embeddings
        if "select" in last_query and "documenttokenembeddings" in last_query:
            # Get doc_id from parameters if possible
            doc_id = None
            if len(self.queries) > 0 and self.queries[-1][1] is not None:
                # Try to find doc_id in parameters
                params = self.queries[-1][1]
                if isinstance(params, tuple) and len(params) > 0:
                    doc_id = params[0]
            
            # If we found a doc_id and it's in our data, return its tokens
            if doc_id and doc_id in self.data["DocumentTokenEmbeddings"]:
                return [(token["token_embedding"], token["metadata_json"]) 
                        for token in self.data["DocumentTokenEmbeddings"][doc_id]]
            
            # Otherwise return tokens for the first document as a fallback
            first_doc_id = list(self.data["DocumentTokenEmbeddings"].keys())[0]
            return [(token["token_embedding"], token["metadata_json"]) 
                    for token in self.data["DocumentTokenEmbeddings"][first_doc_id]]
        
        # For node/edge retrieval (if we add NodeRAG/GraphRAG tests)
        if "select" in last_query and "knowledgegraphnodes" in last_query:
            # Return some mock node data
            return [("node1", "Entity", "Diabetes", "A metabolic disorder", "[0.1,0.2,0.3]", "{}")]
        
        return []
    
    def fetchone(self):
        # For COUNT queries
        last_query = self.queries[-1][0].lower() if self.queries else ""
        if "count" in last_query and "sourcedocuments" in last_query:
            return [len(self.data["SourceDocuments"])]
        return [0]
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

class MockIRISConnector:
    """Mock IRIS connector for testing"""
    def __init__(self):
        self._cursor = MockIRISCursor()
    
    def cursor(self):
        return self._cursor

def test_basic_rag_minimal():
    """Test BasicRAG with minimal mocks"""
    from basic_rag.pipeline import BasicRAGPipeline
    
    # Create mock objects
    connector = MockIRISConnector()
    
    # Simple embedding function
    def embedding_func(text):
        return [[0.1, 0.2, 0.3, 0.4, 0.5]]
    
    # Simple LLM function
    def llm_func(prompt):
        return f"Answer based on {prompt.count('diabetes')} mentions of diabetes."
    
    # Create pipeline
    pipeline = BasicRAGPipeline(
        iris_connector=connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test the pipeline
    result = pipeline.run("What is the role of insulin in diabetes?", top_k=2)
    
    # Assertions
    assert result is not None, "Result should not be None"
    assert "answer" in result, "Result should have 'answer' key"
    assert "retrieved_documents" in result, "Result should have 'retrieved_documents' key"
    
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
    
    # Check if any SQL was executed
    cursor = connector._cursor
    assert len(cursor.queries) > 0, "No SQL queries were executed"
    
    logger.info("BasicRAG minimal test passed")

def test_colbert_minimal():
    """Test ColBERT with minimal mocks"""
    from colbert.pipeline import ColBERTPipeline
    
    # Create mock objects
    connector = MockIRISConnector()
    
    # Token-level embedding function
    def colbert_query_encoder(text):
        tokens = text.split()[:3]  # First 3 tokens
        return [[0.1, 0.2, 0.3] for _ in tokens]  # Same embedding for each token
    
    # Simple LLM function
    def llm_func(prompt):
        return f"ColBERT answer based on {prompt.count('diabetes')} mentions of diabetes."
    
    # Create pipeline
    pipeline = ColBERTPipeline(
        iris_connector=connector,
        colbert_query_encoder=colbert_query_encoder,
        llm_func=llm_func,
        client_side_maxsim=True  # Use client-side for testing
    )
    
    # Test the pipeline
    result = pipeline.run("What is insulin?", top_k=2)
    
    # Assertions
    assert result is not None, "Result should not be None"
    assert "answer" in result, "Result should have 'answer' key"
    assert "retrieved_documents" in result, "Result should have 'retrieved_documents' key"
    
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
    
    logger.info("ColBERT minimal test passed")
