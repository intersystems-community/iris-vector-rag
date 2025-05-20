# tests/test_noderag.py

import pytest
from unittest.mock import MagicMock, patch
import os
import sys
# import sqlalchemy # No longer needed
from typing import Any # For mock type hints

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from noderag.pipeline import NodeRAGPipeline
from common.utils import Document

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnectionTypes, Cursor as IRISCursorTypes
except ImportError:
    IRISConnectionTypes = Any
    IRISCursorTypes = Any

# --- Mock Fixtures ---

@pytest.fixture
def mock_iris_connector_for_noderag():
    """
    Mock for IRIS connection specifically for NodeRAG tests.
    Needs to return KG node and edge data.
    """
    mock_conn = MagicMock(spec=IRISConnectionTypes)
    mock_cursor_method = MagicMock()
    mock_conn.cursor = mock_cursor_method
    
    mock_cursor_instance = MagicMock(spec=IRISCursorTypes)
    mock_cursor_method.return_value = mock_cursor_instance
    
    # Mock fetchall to return different results based on the executed query
    def mock_fetchall_side_effect_noderag(): # Renamed to avoid conflict if other tests use a similar name
        # Ensure call_args is not None before accessing it
        if mock_cursor_instance.execute.call_args is None or not mock_cursor_instance.execute.call_args[0]:
            return [] # Or raise an error, or handle as appropriate for no call

        sql = mock_cursor_instance.execute.call_args[0][0].strip().lower()
        
        if "from rag.knowledgegraphnodes" in sql and "vector_cosine" in sql: # Updated table name and function
            # Mock results for _identify_initial_search_nodes (node_id, score)
            return [
                ("node_kg_1", 0.95), # Diabetes
                ("node_kg_2", 0.90), # Insulin
                ("node_kg_3", 0.85), # Doc1
            ]
        elif "from rag.knowledgegraphnodes" in sql and "where node_id in" in sql: # Updated table name
            # Mock results for _retrieve_content_for_nodes (node_id, description_text)
            # Need to check which node_ids were requested in params
            params = mock_cursor_instance.execute.call_args[0][1]
            requested_ids = set(params) if isinstance(params, (list, tuple)) else {params}
            
            mock_node_data = {
                "node_kg_1": "A chronic disease.",
                "node_kg_2": "A hormone used to treat diabetes.",
                "node_kg_3": "Summary of Doc1 content.",
            }
            
            return [(node_id, mock_node_data.get(node_id, "Mock content")) for node_id in requested_ids if node_id in mock_node_data]
            
        elif "from rag.knowledgegraphedges" in sql: # Updated table name
            # Mock results for fetching edges (edge_id, source, target, type, weight, properties)
            # This would be used by _traverse_graph if implemented
            return [
                ("edge1", "node_kg_1", "node_kg_2", "treated_by", 1.0, "{}"),
                ("edge2", "node_kg_3", "node_kg_1", "mentions", 1.0, "{}"),
            ]
        
        # Default for other queries
        return []
    
    mock_cursor_instance.fetchall = MagicMock(side_effect=mock_fetchall_side_effect_noderag)
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
    return MagicMock(return_value="Mocked NodeRAG LLM answer.")

@pytest.fixture
def mock_graph_lib():
    """Mocks a graph library (e.g., networkx)."""
    mock_lib = MagicMock()
    # Mock specific methods that _traverse_graph might call if implemented
    # e.g., mock_lib.Graph.return_value = MagicMock()
    # mock_lib.single_source_shortest_path.return_value = {"node_kg_1": {}, "node_kg_2": {}} # Example traversal result
    return mock_lib


@pytest.fixture
def noderag_pipeline(mock_iris_connector_for_noderag, mock_embedding_func, mock_llm_func, mock_graph_lib):
    """Initializes NodeRAGPipeline with mock dependencies."""
    return NodeRAGPipeline(
        iris_connector=mock_iris_connector_for_noderag,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func,
        graph_lib=mock_graph_lib
    )

# --- Unit Tests ---

def test_identify_initial_search_nodes(noderag_pipeline, mock_iris_connector_for_noderag, mock_embedding_func):
    """Tests the _identify_initial_search_nodes method."""
    query_text = "Find nodes about diabetes"
    top_n_seed = 3
    
    mock_cursor = mock_iris_connector_for_noderag.cursor.return_value
    
    initial_node_ids = noderag_pipeline._identify_initial_search_nodes(query_text, top_n_seed=top_n_seed)

    mock_embedding_func.assert_called_once_with([query_text])
    mock_iris_connector_for_noderag.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert f"SELECT TOP {top_n_seed}" in executed_sql
    assert "FROM RAG.KnowledgeGraphNodes" in executed_sql # Schema qualified
    assert "VECTOR_COSINE(embedding, TO_VECTOR(" in executed_sql # Correct function and start of TO_VECTOR
    assert "'DOUBLE', 768" in executed_sql # Check for type and dimension in TO_VECTOR
    
    mock_cursor.fetchall.assert_called_once()
    assert initial_node_ids == ["node_kg_1", "node_kg_2", "node_kg_3"] # Based on mock fetchall side_effect

def test_traverse_graph_placeholder(noderag_pipeline):
    """Tests the placeholder _traverse_graph method."""
    seed_node_ids = ["node_kg_1", "node_kg_3"]
    query_text = "Traversal query"
    
    # With the placeholder implementation, it should just return the seed nodes as a set
    relevant_nodes = noderag_pipeline._traverse_graph(seed_node_ids, query_text)

    assert isinstance(relevant_nodes, set)
    assert relevant_nodes == set(seed_node_ids) # Placeholder returns seeds

    # Test with empty seeds
    assert noderag_pipeline._traverse_graph([], query_text) == set()


def test_retrieve_content_for_nodes(noderag_pipeline, mock_iris_connector_for_noderag):
    """Tests the _retrieve_content_for_nodes method."""
    node_ids = {"node_kg_1", "node_kg_3"} # Set of node IDs
    
    mock_cursor = mock_iris_connector_for_noderag.cursor.return_value
    
    retrieved_docs = noderag_pipeline._retrieve_content_for_nodes(node_ids)

    mock_iris_connector_for_noderag.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    # Condense all whitespace (including newlines, tabs) to single spaces for robust checking
    condensed_sql = " ".join(executed_sql.split())
    # Check for schema qualified table name and IN clause structure
    assert "FROM RAG.KnowledgeGraphNodes WHERE node_id IN (" in condensed_sql
    # Verify that parameters were passed to execute for the IN clause
    assert mock_cursor.execute.call_args[0][1] is not None # Check that parameters tuple exists
    assert len(mock_cursor.execute.call_args[0][1]) == len(node_ids) # Check number of parameters matches number of node_ids
    
    mock_cursor.fetchall.assert_called_once()
    
    assert len(retrieved_docs) == 2 # Based on mock fetchall side_effect for content
    assert all(isinstance(doc, Document) for doc in retrieved_docs)
    # Check content based on mock fetchall side_effect
    fetched_ids = {doc.id for doc in retrieved_docs}
    assert fetched_ids == {"node_kg_1", "node_kg_3"}
    
    # Test with empty node_ids
    assert noderag_pipeline._retrieve_content_for_nodes(set()) == []


def test_retrieve_documents_from_graph_flow(noderag_pipeline):
    """Tests the retrieve_documents_from_graph orchestration."""
    query_text = "Graph retrieval query"
    
    # Mock sub-methods to test orchestration
    noderag_pipeline._identify_initial_search_nodes = MagicMock(return_value=["node_kg_1", "node_kg_3"])
    noderag_pipeline._traverse_graph = MagicMock(return_value={"node_kg_1", "node_kg_2", "node_kg_3"}) # Traversal finds more nodes
    noderag_pipeline._retrieve_content_for_nodes = MagicMock(return_value=[
        Document(id="node_kg_1", content="Content 1"),
        Document(id="node_kg_2", content="Content 2"),
        Document(id="node_kg_3", content="Content 3"),
    ])

    retrieved_docs = noderag_pipeline.retrieve_documents_from_graph(query_text)

    noderag_pipeline._identify_initial_search_nodes.assert_called_once_with(query_text, top_n_seed=5) # Changed to keyword arg
    noderag_pipeline._traverse_graph.assert_called_once_with(["node_kg_1", "node_kg_3"], query_text)
    noderag_pipeline._retrieve_content_for_nodes.assert_called_once_with({"node_kg_1", "node_kg_2", "node_kg_3"})

    assert len(retrieved_docs) == 3
    # Order might not be guaranteed, but check if the expected node IDs are present
    retrieved_ids = {doc.id for doc in retrieved_docs}
    assert retrieved_ids == {"node_kg_1", "node_kg_2", "node_kg_3"}


def test_generate_answer(noderag_pipeline, mock_llm_func):
    """Tests the generate_answer method."""
    query_text = "NodeRAG final answer query"
    retrieved_docs = [Document(id="node1", content="Node content A"), Document(id="node2", content="Node content B")]
    
    answer = noderag_pipeline.generate_answer(query_text, retrieved_docs)

    expected_context = "Node content A\n\nNode content B"
    expected_prompt = f"""You are a helpful AI assistant. Answer the question based on the provided information from a knowledge graph.
If the information does not contain the answer, state that you cannot answer based on the provided information.

Information from Knowledge Graph:
{expected_context}

Question: {query_text}

Answer:"""
    mock_llm_func.assert_called_once_with(expected_prompt)
    assert answer == "Mocked NodeRAG LLM answer."

def test_run_orchestration(noderag_pipeline, mock_llm_func):
    """Tests the full run method orchestration."""
    query_text = "Run NodeRAG query"
    
    # Mock sub-methods to test run orchestration
    noderag_pipeline.retrieve_documents_from_graph = MagicMock(return_value=[Document(id="node_final", content="Final node content")])
    noderag_pipeline.generate_answer = MagicMock(return_value="Final NodeRAG Answer")

    result = noderag_pipeline.run(query_text, top_k_seeds=3) # Use different top_k_seeds

    noderag_pipeline.retrieve_documents_from_graph.assert_called_once_with(query_text, top_k_seeds=3) # Changed to keyword arg
    noderag_pipeline.generate_answer.assert_called_once_with(query_text, noderag_pipeline.retrieve_documents_from_graph.return_value)

    assert result["query"] == query_text
    assert result["answer"] == "Final NodeRAG Answer"
    assert len(result["retrieved_documents"]) == 1
    assert result["retrieved_documents"][0]['id'] == "node_final" # Access as dict
