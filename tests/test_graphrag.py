# tests/test_graphrag.py

import pytest
from unittest.mock import MagicMock, patch
import os
import sys
# import sqlalchemy # No longer needed
from typing import Any # For mock type hints

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphrag.pipeline import GraphRAGPipeline
from common.utils import Document

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnectionTypes, Cursor as IRISCursorTypes
except ImportError:
    IRISConnectionTypes = Any
    IRISCursorTypes = Any

# --- Mock Fixtures ---

@pytest.fixture
def mock_iris_connector_for_graphrag():
    """
    Mock for IRIS connection specifically for GraphRAG tests.
    Needs to return KG node and edge data.
    """
    mock_conn = MagicMock(spec=IRISConnectionTypes)
    mock_cursor_method = MagicMock()
    mock_conn.cursor = mock_cursor_method
    
    mock_cursor_instance = MagicMock(spec=IRISCursorTypes)
    mock_cursor_method.return_value = mock_cursor_instance
    
    # Mock fetchall to return different results based on the executed query
    def mock_fetchall_side_effect_graphrag(): # Renamed for clarity
        # Ensure call_args is not None before accessing it
        if mock_cursor_instance.execute.call_args is None or not mock_cursor_instance.execute.call_args[0]:
            return []

        sql = mock_cursor_instance.execute.call_args[0][0].strip().lower()
        
        if "from rag.knowledgegraphnodes" in sql and "vector_cosine" in sql: # Updated table name and function
            # Mock results for _find_start_nodes (node_id, score)
            return [
                ("node_kg_1", 0.95), # Diabetes
                ("node_kg_2", 0.90), # Insulin
            ]
        elif "from rag.knowledgegraphnodes" in sql and "where node_id in" in sql: # Updated table name
            # Mock results for _get_context_from_traversed_nodes (node_id, description_text)
            params = mock_cursor_instance.execute.call_args[0][1]
            requested_ids = set(params) if isinstance(params, (list, tuple)) else {params}
            
            mock_node_data = {
                "node_kg_1": "A chronic disease.",
                "node_kg_2": "A hormone used to treat diabetes.",
                "node_kg_3": "Summary of Doc1 content.",
                "node_kg_4": "Organ producing insulin.",
            }
            
            return [(node_id, mock_node_data.get(node_id, "Mock content")) for node_id in requested_ids if node_id in mock_node_data]
            
        elif "with recursive pathcte" in sql: # This would also need RAG.KnowledgeGraphEdges if implemented
            # Mock results for _traverse_kg_recursive_cte (end_node)
            # This is a placeholder, returning a fixed set of traversed nodes
            return [("node_kg_1",), ("node_kg_2",), ("node_kg_4",)] # Example: Diabetes -> Insulin -> Pancreas
        
        # Default for other queries
        return []
    
    mock_cursor_instance.fetchall = MagicMock(side_effect=mock_fetchall_side_effect_graphrag)
    mock_cursor_instance.execute = MagicMock()
    mock_cursor_instance.close = MagicMock()
    mock_conn.close = MagicMock()
    return mock_conn

@pytest.fixture
def mock_embedding_func():
    """Mocks the embedding function."""
    return MagicMock(return_value=[[0.1]*384]) # Returns a single embedding

@pytest.fixture
def mock_llm_func():
    """Mocks the LLM function."""
    return MagicMock(return_value="Mocked GraphRAG LLM answer.")


@pytest.fixture
def graphrag_pipeline(mock_iris_connector_for_graphrag, mock_embedding_func, mock_llm_func):
    """Initializes GraphRAGPipeline with mock dependencies."""
    return GraphRAGPipeline(
        iris_connector=mock_iris_connector_for_graphrag,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func
    )

# --- Unit Tests ---

def test_find_start_nodes(graphrag_pipeline, mock_iris_connector_for_graphrag, mock_embedding_func):
    """Tests the _find_start_nodes method."""
    query_text = "Find nodes about diabetes"
    top_n = 2
    
    mock_cursor = mock_iris_connector_for_graphrag.cursor.return_value
    
    start_node_ids = graphrag_pipeline._find_start_nodes(query_text, top_n=top_n)

    mock_embedding_func.assert_called_once_with([query_text])
    mock_iris_connector_for_graphrag.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert f"SELECT TOP {top_n}" in executed_sql
    assert "FROM RAG.KnowledgeGraphNodes" in executed_sql # Schema qualified
    assert "VECTOR_COSINE(embedding, TO_VECTOR(" in executed_sql # Correct function and start of TO_VECTOR
    assert "'DOUBLE', 768" in executed_sql # Check for type and dimension
    
    mock_cursor.fetchall.assert_called_once()
    assert start_node_ids == ["node_kg_1", "node_kg_2"] # Based on mock fetchall side_effect

def test_traverse_kg_recursive_cte_placeholder(graphrag_pipeline):
    """Tests the placeholder _traverse_kg_recursive_cte method."""
    start_node_ids = ["node_kg_1", "node_kg_3"]
    max_depth = 2
    
    # With the placeholder implementation, it should just return the start nodes as a set
    relevant_nodes = graphrag_pipeline._traverse_kg_recursive_cte(start_node_ids, max_depth=max_depth)

    assert isinstance(relevant_nodes, set)
    assert relevant_nodes == set(start_node_ids) # Placeholder returns seeds

    # Test with empty seeds
    assert graphrag_pipeline._traverse_kg_recursive_cte([], max_depth=max_depth) == set() # Fixed typo


def test_get_context_from_traversed_nodes(graphrag_pipeline, mock_iris_connector_for_graphrag):
    """Tests the _get_context_from_traversed_nodes method."""
    node_ids = {"node_kg_1", "node_kg_4"} # Set of node IDs
    
    mock_cursor = mock_iris_connector_for_graphrag.cursor.return_value
    
    retrieved_docs = graphrag_pipeline._get_context_from_traversed_nodes(node_ids)

    mock_iris_connector_for_graphrag.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    
    # Check for key parts of the SQL query, ignoring formatting and comments
    condensed_sql = " ".join(executed_sql.split()) # Normalize whitespace
    assert "SELECT node_id, description_text" in condensed_sql
    assert "FROM RAG.KnowledgeGraphNodes" in condensed_sql # Schema qualified
    assert "WHERE node_id IN (" in condensed_sql # Check for IN clause start
    
    # Also check the parameters passed to execute
    executed_params = mock_cursor.execute.call_args[0][1]
    assert set(executed_params) == node_ids # Ensure the correct node IDs were passed

    mock_cursor.fetchall.assert_called_once()
    
    assert len(retrieved_docs) == 2 # Based on mock fetchall side_effect for content
    assert all(isinstance(doc, Document) for doc in retrieved_docs)
    # Check content based on mock fetchall side_effect
    fetched_ids = {doc.id for doc in retrieved_docs}
    assert fetched_ids == {"node_kg_1", "node_kg_4"}
    
    # Test with empty node_ids
    assert graphrag_pipeline._get_context_from_traversed_nodes(set()) == []


def test_retrieve_documents_via_kg_flow(graphrag_pipeline):
    """Tests the retrieve_documents_via_kg orchestration."""
    query_text = "KG retrieval query"
    
    # Mock sub-methods to test orchestration
    graphrag_pipeline._find_start_nodes = MagicMock(return_value=["node_kg_1", "node_kg_3"])
    graphrag_pipeline._traverse_kg_recursive_cte = MagicMock(return_value={"node_kg_1", "node_kg_2", "node_kg_4"}) # Traversal finds nodes
    graphrag_pipeline._get_context_from_traversed_nodes = MagicMock(return_value=[
        Document(id="node_kg_1", content="Content 1"),
        Document(id="node_kg_2", content="Content 2"),
        Document(id="node_kg_4", content="Content 4"),
    ])

    retrieved_docs = graphrag_pipeline.retrieve_documents_via_kg(query_text)

    graphrag_pipeline._find_start_nodes.assert_called_once_with(query_text, top_n=3) # Default top_n_start_nodes
    graphrag_pipeline._traverse_kg_recursive_cte.assert_called_once_with(["node_kg_1", "node_kg_3"])
    graphrag_pipeline._get_context_from_traversed_nodes.assert_called_once_with({"node_kg_1", "node_kg_2", "node_kg_4"})

    assert len(retrieved_docs) == 3
    # Order might not be guaranteed, but check if the expected node IDs are present
    retrieved_ids = {doc.id for doc in retrieved_docs}
    assert retrieved_ids == {"node_kg_1", "node_kg_2", "node_kg_4"}


def test_generate_answer(graphrag_pipeline, mock_llm_func):
    """Tests the generate_answer method."""
    query_text = "GraphRAG final answer query"
    retrieved_docs = [Document(id="nodeA", content="Node info A"), Document(id="nodeB", content="Node info B")]
    
    answer = graphrag_pipeline.generate_answer(query_text, retrieved_docs)

    expected_context = "Node info A\n\nNode info B"
    expected_prompt = f"""You are a helpful AI assistant. Answer the question based on the provided information from a knowledge graph.
If the information does not contain the answer, state that you cannot answer based on the provided information.

Information from Knowledge Graph:
{expected_context}

Question: {query_text}

Answer:"""
    mock_llm_func.assert_called_once_with(expected_prompt)
    assert answer == "Mocked GraphRAG LLM answer."

def test_run_orchestration(graphrag_pipeline, mock_llm_func):
    """Tests the full run method orchestration."""
    query_text = "Run GraphRAG query"
    
    # Mock sub-methods to test orchestration
    graphrag_pipeline.retrieve_documents_via_kg = MagicMock(return_value=[Document(id="node_final", content="Final node info")])
    graphrag_pipeline.generate_answer = MagicMock(return_value="Final GraphRAG Answer")

    result = graphrag_pipeline.run(query_text, top_n_start_nodes=5) # Use different top_n_start_nodes

    graphrag_pipeline.retrieve_documents_via_kg.assert_called_once_with(query_text, top_n_start_nodes=5)
    graphrag_pipeline.generate_answer.assert_called_once_with(query_text, graphrag_pipeline.retrieve_documents_via_kg.return_value)

    assert result["query"] == query_text
    assert result["answer"] == "Final GraphRAG Answer"
    assert len(result["retrieved_documents"]) == 1
    assert result["retrieved_documents"][0]['id'] == "node_final" # Access as dict
