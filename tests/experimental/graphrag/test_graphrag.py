# tests/test_graphrag.py

import pytest
from unittest.mock import MagicMock, patch
import os
import sys
import logging # Added for debug logging
# import sqlalchemy # No longer needed
from typing import Any # For mock type hints

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from common.utils import Document # Updated import

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnectionTypes, Cursor as IRISCursorTypes
except ImportError:
    IRISConnectionTypes = Any
    IRISCursorTypes = Any

logger = logging.getLogger(__name__) # Added logger
logger.setLevel(logging.DEBUG) # Set to DEBUG for tests

# --- Mock Fixtures ---

@pytest.fixture
def mock_iris_connector_for_graphrag():
    """
    Mock for IRIS connection specifically for GraphRAG tests.
    Reflects KnowledgeGraph schema and methods in OriginalGraphRAGPipeline.
    """
    mock_conn = MagicMock(spec=IRISConnectionTypes)
    mock_cursor_method = MagicMock()
    mock_conn.cursor = mock_cursor_method
    
    mock_cursor_instance = MagicMock(spec=IRISCursorTypes)
    mock_cursor_method.return_value = mock_cursor_instance
    
    def mock_fetchall_side_effect_graphrag():
        if mock_cursor_instance.execute.call_args is None or not mock_cursor_instance.execute.call_args[0]:
            logger.debug("mock_fetchall_side_effect_graphrag: No execute call_args found.")
            return []

        sql = mock_cursor_instance.execute.call_args[0][0].strip().lower()
        params = mock_cursor_instance.execute.call_args[0][1] if len(mock_cursor_instance.execute.call_args[0]) > 1 else ()
        logger.debug(f"mock_fetchall_side_effect_graphrag: SQL executed: {sql[:200]}... with params: {params}")

        # For _find_seed_entities - keyword part
        if "from knowledgegraph.entities" in sql and "lower(entity_name) like" in sql:
            logger.debug("mock_fetchall_side_effect_graphrag: Matched _find_seed_entities keyword query.")
            # Returns (entity_id, entity_name, entity_type, source_doc_id)
            return [
                ("ent_kw_1", "Diabetes", "DISEASE", "doc_1"),
            ]
        # For _find_seed_entities - embedding part
        elif "from knowledgegraph.entities" in sql and "vector_cosine(to_vector(embedding), to_vector(?))" in sql:
            logger.debug("mock_fetchall_side_effect_graphrag: Matched _find_seed_entities embedding query.")
            # Returns (entity_id, entity_name, entity_type, similarity)
            return [
                ("ent_emb_1", "Insulin", "DRUG", 0.95),
                ("ent_emb_2", "Pancreas", "ORG", 0.90),
            ]
        # For _traverse_knowledge_graph
        elif "from knowledgegraph.entityrelationships r" in sql and "join knowledgegraph.entities e on" in sql:
            logger.debug("mock_fetchall_side_effect_graphrag: Matched _traverse_knowledge_graph query.")
            # Returns (entity_id, entity_name, entity_type, rel_type)
            # Simulate finding 'ent_rel_1' related to 'ent_emb_1' (Insulin)
            if params and 'ent_emb_1' in params:
                 return [("ent_rel_1", "Glucose Regulation", "PROCESS", "REGULATES")]
            return [
                ("ent_rel_1", "Glucose Regulation", "PROCESS", "REGULATES"), # Related to Insulin
                ("ent_rel_2", "Endocrine System", "SYSTEM", "PART_OF"),   # Related to Pancreas
            ]
        # For _get_documents_from_entities
        elif "from knowledgegraph.documententities de" in sql and "join knowledgegraph.sourcedocuments sd on" in sql:
            logger.debug("mock_fetchall_side_effect_graphrag: Matched _get_documents_from_entities query.")
            # Returns (doc_id, text_content)
            # Simulate documents linked to ent_kw_1 (Diabetes) and ent_rel_1 (Glucose Regulation)
            if params and ('ent_kw_1' in params or 'ent_rel_1' in params):
                return [
                    ("doc_1", "Content about Diabetes and its management."),
                    ("doc_2", "Content about Glucose Regulation and Insulin."),
                ]
            return []
        
        logger.debug(f"mock_fetchall_side_effect_graphrag: No specific match for SQL: {sql[:100]}...")
        return []
    
    mock_cursor_instance.fetchall = MagicMock(side_effect=mock_fetchall_side_effect_graphrag)
    mock_cursor_instance.execute = MagicMock()
    mock_cursor_instance.close = MagicMock()
    mock_conn.close = MagicMock()
    return mock_conn

@pytest.fixture
def mock_embedding_func():
    """Mocks the embedding function."""
    mock_ef = MagicMock(return_value=[[0.1]*768]) # Ensure correct dimension if checked by pipeline
    return mock_ef

@pytest.fixture
def mock_llm_func():
    """Mocks the LLM function."""
    return MagicMock(return_value="Mocked GraphRAG LLM answer.")


@pytest.fixture
def graphrag_pipeline_orig(mock_iris_connector_for_graphrag, mock_embedding_func, mock_llm_func): # Renamed fixture
    """Initializes OriginalGraphRAGPipeline with mock dependencies."""
    return OriginalGraphRAGPipeline( # Use OriginalGraphRAGPipeline
        iris_connector=mock_iris_connector_for_graphrag,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func
    )

# --- Unit Tests ---

def test_find_seed_entities(graphrag_pipeline_orig, mock_iris_connector_for_graphrag, mock_embedding_func): # Renamed test and fixture
    """Tests the _find_seed_entities method."""
    query_text = "diabetes treatment"
    top_n = 3 # Requesting 3 entities
    
    mock_cursor = mock_iris_connector_for_graphrag.cursor.return_value
    
    # Call the actual method name from OriginalGraphRAGPipeline
    seed_entities = graphrag_pipeline_orig._find_seed_entities(query_text, top_k=top_n)

    # _find_seed_entities calls cursor.execute twice (keyword then embedding if needed)
    assert mock_iris_connector_for_graphrag.cursor.call_count >= 1 # Called at least once
    
    # Check keyword query call
    keyword_call = None
    embedding_call = None
    for call_args in mock_cursor.execute.call_args_list:
        sql = call_args[0][0].lower()
        if "lower(entity_name) like" in sql:
            keyword_call = call_args
        elif "vector_cosine(to_vector(embedding), to_vector(?))" in sql:
            embedding_call = call_args
    
    assert keyword_call is not None, "Keyword search SQL was not executed"
    keyword_sql = keyword_call[0][0]
    assert f"select top {top_n}" in keyword_sql.lower() # Initial top_k for keyword
    assert "from knowledgegraph.entities" in keyword_sql.lower()
    assert "lower(entity_name) like ?" in keyword_sql.lower()
    # Params for keyword: ['%diabetes%', '%treatment%']
    assert keyword_call[0][1] == ['%diabetes%', '%treatment%']


    # Check embedding query call (it might not be called if keyword search yields enough)
    # Based on mock, keyword returns 1, embedding returns 2. top_n is 3. So embedding should be called for 2.
    assert embedding_call is not None, "Embedding search SQL was not executed"
    embedding_sql = embedding_call[0][0]
    # remaining_top_k for embedding query will be top_n - len(keyword_results) = 3 - 1 = 2
    assert f"select top {top_n - 1}" in embedding_sql.lower()
    assert "from knowledgegraph.entities" in embedding_sql.lower()
    assert "vector_cosine(to_vector(embedding), to_vector(?))" in embedding_sql.lower()
    mock_embedding_func.assert_called_once_with([query_text]) # Embedding func called for query

    # Check the final returned list of tuples (entity_id, entity_name, relevance_score)
    # Expected: 1 from keyword, 2 from embedding
    assert len(seed_entities) == 3
    assert seed_entities[0] == ("ent_kw_1", "Diabetes", 0.9) # Keyword match
    assert seed_entities[1] == ("ent_emb_1", "Insulin", 0.95)  # Embedding match
    assert seed_entities[2] == ("ent_emb_2", "Pancreas", 0.90) # Embedding match
    assert all(isinstance(item, tuple) and len(item) == 3 for item in seed_entities)
    assert all(isinstance(item[2], float) for item in seed_entities) # Relevance score is float

def test_traverse_knowledge_graph(graphrag_pipeline_orig): # Renamed test and fixture
    """Tests the _traverse_knowledge_graph method."""
    seed_entities_data = [("ent_emb_1", "Insulin", 0.95), ("ent_other_1", "Other Seed", 0.8)]
    max_depth = 1 # Keep depth shallow for predictable mock
    max_entities = 5
    
    mock_cursor = mock_iris_connector_for_graphrag.cursor.return_value
    
    # Call the actual method name
    relevant_entity_ids = graphrag_pipeline_orig._traverse_knowledge_graph(seed_entities_data, max_depth=max_depth, max_entities=max_entities)

    # Check SQL execution for traversal
    # Based on mock, it should be called once for depth 1
    traversal_sql_calls = [
        call for call in mock_cursor.execute.call_args_list
        if "from knowledgegraph.entityrelationships r" in call[0][0].lower()
    ]
    assert len(traversal_sql_calls) >= 1 # Should be called at least once for depth 1

    # Expected relevant entities: initial seeds + one related from mock
    # Seeds: "ent_emb_1", "ent_other_1"
    # Mock relates "ent_rel_1" to "ent_emb_1"
    expected_ids = {"ent_emb_1", "ent_other_1", "ent_rel_1"}
    assert isinstance(relevant_entity_ids, set)
    assert relevant_entity_ids == expected_ids

    # Test with empty seeds
    assert graphrag_pipeline_orig._traverse_knowledge_graph([], max_depth=max_depth, max_entities=max_entities) == set()


def test_get_documents_from_entities(graphrag_pipeline_orig, mock_iris_connector_for_graphrag): # Renamed test and fixture
    """Tests the _get_documents_from_entities method."""
    entity_ids = {"ent_kw_1", "ent_rel_1"} # Entities that have mock documents
    top_k = 5
    
    mock_cursor = mock_iris_connector_for_graphrag.cursor.return_value
    
    # Call the actual method name
    retrieved_docs = graphrag_pipeline_orig._get_documents_from_entities(entity_ids, top_k=top_k)

    # Check SQL execution
    doc_sql_call = None
    for call_args in mock_cursor.execute.call_args_list:
        sql = call_args[0][0].lower()
        if "from knowledgegraph.documententities de" in sql:
            doc_sql_call = call_args
            break
    assert doc_sql_call is not None, "Document retrieval SQL not executed"
    
    executed_sql = doc_sql_call[0][0]
    assert f"select top {top_k}" in executed_sql.lower()
    assert "from knowledgegraph.documententities de" in executed_sql.lower()
    assert "join knowledgegraph.sourcedocuments sd on" in executed_sql.lower()
    assert "where de.entity_id in (" in executed_sql.lower()
    
    # Check params (should be list of entity_ids)
    # The mock is set up to return 2 docs for these entities
    assert len(retrieved_docs) == 2
    assert all(isinstance(doc, Document) for doc in retrieved_docs)
    retrieved_doc_ids = {doc.id for doc in retrieved_docs}
    assert retrieved_doc_ids == {"doc_1", "doc_2"}
    
    # Test with empty entity_ids
    assert graphrag_pipeline_orig._get_documents_from_entities(set(), top_k=top_k) == []


def test_retrieve_documents_via_kg_flow(graphrag_pipeline_orig): # Use original pipeline
    """Tests the retrieve_documents_via_kg orchestration and adds logging."""
    query_text = "diabetes and insulin"
    top_k_retrieval = 5 # For retrieve_documents_via_kg

    # Mock the sub-methods to control their output for this orchestration test
    # and to check if they are called correctly.
    # The mock_iris_connector will still be used by these if they make DB calls.
    
    # Mock return for _find_seed_entities: (entity_id, entity_name, relevance_score)
    mock_seed_entities_result = [
        ("ent_kw_1", "Diabetes", 0.9),
        ("ent_emb_1", "Insulin", 0.95)
    ]
    graphrag_pipeline_orig._find_seed_entities = MagicMock(return_value=mock_seed_entities_result)
    
    # Mock return for _traverse_knowledge_graph: set of entity_ids
    mock_traversed_ids_result = {"ent_kw_1", "ent_emb_1", "ent_rel_1"} # Diabetes, Insulin, Glucose Regulation
    graphrag_pipeline_orig._traverse_knowledge_graph = MagicMock(return_value=mock_traversed_ids_result)

    # Mock return for _get_documents_from_entities: List[Document]
    mock_kg_docs_result = [
        Document(id="doc_1", content="Content about Diabetes.", score=0.8),
        Document(id="doc_2", content="Content about Insulin and Glucose Regulation.", score=0.75)
    ]
    graphrag_pipeline_orig._get_documents_from_entities = MagicMock(return_value=mock_kg_docs_result)

    logger.info(f"\n--- test_retrieve_documents_via_kg_flow: START for query '{query_text}' ---")
    retrieved_docs, method = graphrag_pipeline_orig.retrieve_documents_via_kg(query_text, top_k=top_k_retrieval)
    logger.info(f"--- test_retrieve_documents_via_kg_flow: END. Method: {method} ---")

    # Assertions for method calls
    graphrag_pipeline_orig._find_seed_entities.assert_called_once_with(query_text, top_k=10) # Default top_k for seeds
    
    # _traverse_knowledge_graph is called with the result of _find_seed_entities
    graphrag_pipeline_orig._traverse_knowledge_graph.assert_called_once_with(
        mock_seed_entities_result, # Pass the actual data _find_seed_entities would return
        max_depth=2,
        max_entities=100
    )
    
    # _get_documents_from_entities is called with the result of _traverse_knowledge_graph
    graphrag_pipeline_orig._get_documents_from_entities.assert_called_once_with(
        mock_traversed_ids_result,
        top_k_retrieval # top_k from the main call
    )

    # Assertions for results
    assert method == "knowledge_graph_traversal" # Expecting KG success
    assert len(retrieved_docs) == 2
    retrieved_ids = {doc.id for doc in retrieved_docs}
    assert retrieved_ids == {"doc_1", "doc_2"}
    
    # Log the actual retrieved documents for debugging
    logger.info(f"Retrieved documents for '{query_text}':")
    for doc in retrieved_docs:
        logger.info(f"  ID: {doc.id}, Score: {doc.score}, Content: {doc.content[:50]}...")
    
    # Test fallback scenario: if _find_seed_entities returns empty
    graphrag_pipeline_orig._find_seed_entities.return_value = []
    # Mock fallback vector search to check if it's called
    graphrag_pipeline_orig._fallback_vector_search = MagicMock(return_value=[Document(id="fallback_doc", content="Fallback content", score=0.5)])
    
    logger.info(f"\n--- test_retrieve_documents_via_kg_flow: FALLBACK TEST for query '{query_text}' ---")
    retrieved_docs_fallback, method_fallback = graphrag_pipeline_orig.retrieve_documents_via_kg(query_text, top_k=top_k_retrieval)
    logger.info(f"--- test_retrieve_documents_via_kg_flow: FALLBACK TEST END. Method: {method_fallback} ---")
    
    assert method_fallback == "fallback_vector_search"
    assert len(retrieved_docs_fallback) == 1
    assert retrieved_docs_fallback[0].id == "fallback_doc"
    graphrag_pipeline_orig._fallback_vector_search.assert_called_once_with(query_text, top_k_retrieval)


def test_generate_answer(graphrag_pipeline_orig, mock_llm_func): # Use original pipeline
    """Tests the generate_answer method."""
    query_text = "GraphRAG final answer query"
    retrieved_docs = [Document(id="doc_1", content="Node info A"), Document(id="doc_2", content="Node info B")]
    
    answer = graphrag_pipeline_orig.generate_answer(query_text, retrieved_docs)

    # Context will be "Document doc_1: Node info A\n\nDocument doc_2: Node info B"
    expected_context_part1 = "Document doc_1: Node info A"
    expected_context_part2 = "Document doc_2: Node info B"
    
    # Check that the prompt passed to LLM contains these parts
    prompt_arg = mock_llm_func.call_args[0][0]
    assert expected_context_part1 in prompt_arg
    assert expected_context_part2 in prompt_arg
    assert f"Question: {query_text}" in prompt_arg
    assert answer == "Mocked GraphRAG LLM answer."

def test_run_orchestration(graphrag_pipeline_orig, mock_llm_func): # Use original pipeline
    """Tests the full run method orchestration."""
    query_text = "Run GraphRAG query"
    
    # Mock the main retrieval method for this orchestration test
    mock_retrieved_docs = [Document(id="node_final", content="Final node info")]
    graphrag_pipeline_orig.retrieve_documents_via_kg = MagicMock(return_value=(mock_retrieved_docs, "knowledge_graph_traversal"))
    
    # generate_answer is already part of graphrag_pipeline_orig and uses mock_llm_func
    # We can spy on it or trust the previous test_generate_answer
    # For full orchestration, let's ensure generate_answer is called correctly.
    graphrag_pipeline_orig.generate_answer = MagicMock(return_value="Final GraphRAG Answer from Orchestration")


    result = graphrag_pipeline_orig.run(query_text, top_k=5) # top_k here is for retrieve_documents_via_kg

    graphrag_pipeline_orig.retrieve_documents_via_kg.assert_called_once_with(query_text, 5)
    graphrag_pipeline_orig.generate_answer.assert_called_once_with(query_text, mock_retrieved_docs)

    assert result["query"] == query_text
    assert result["answer"] == "Final GraphRAG Answer from Orchestration"
    assert len(result["retrieved_documents"]) == 1
    assert result["retrieved_documents"][0]['id'] == "node_final"
    assert result["method"] == "knowledge_graph_traversal"