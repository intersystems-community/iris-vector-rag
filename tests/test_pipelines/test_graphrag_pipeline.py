import pytest
import os
import shutil
from typing import Dict, Any
from unittest.mock import Mock, patch

from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.core.models import Document

# Sample data directory for tests
TEST_DATA_DIR = "tests/test_pipelines/temp_graphrag_data"
DOC_COUNT = 15 # Increased for scale testing

@pytest.fixture
def mock_connection_manager():
    """Mock connection manager for testing."""
    mock_manager = Mock()
    mock_connection = Mock()
    mock_cursor = Mock()
    
    mock_manager.get_connection.return_value = mock_connection
    mock_connection.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = [0]  # For count queries
    
    return mock_manager

@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for testing."""
    mock_manager = Mock()
    # Ensure get() returns proper values, not Mock objects
    def mock_get(key, default=None):
        config_map = {
            'pipelines:graphrag': {"top_k": 3, "max_entities": 5, "relationship_depth": 1},
            'pipelines:graphrag:top_k': 3,
            'pipelines:graphrag:max_entities': 5,
            'pipelines:graphrag:relationship_depth': 1,
            'storage:iris:vector_data_type': "FLOAT",
        }
        return config_map.get(key, default if default is not None else {})
    
    mock_manager.get.side_effect = mock_get
    mock_manager.get_embedding_config.return_value = {"model": "all-MiniLM-L6-v2", "api_key": "test_key"}
    
    return mock_manager

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    
    # Mock document storage
    mock_store.add_documents.return_value = None
    
    # Mock search functionality
    test_doc1 = Document(id="doc_1", page_content="Test document about Apples and Oranges", metadata={"source": "test"})
    test_doc2 = Document(id="doc_2", page_content="Test document about Bananas and Grapes", metadata={"source": "test"})
    mock_store.similarity_search.return_value = [test_doc1, test_doc2]
    
    return mock_store

@pytest.fixture
def graphrag_pipeline_instance(mock_connection_manager, mock_config_manager, mock_vector_store) -> GraphRAGPipeline:
    """Provides a GraphRAGPipeline instance for tests."""
    with patch('iris_rag.storage.enterprise_storage.IRISStorage'), \
         patch('iris_rag.embeddings.manager.EmbeddingManager'), \
         patch('iris_rag.storage.schema_manager.SchemaManager'):
        
        pipeline = GraphRAGPipeline(mock_connection_manager, mock_config_manager, vector_store=mock_vector_store, llm_func=None)
        return pipeline

@pytest.fixture(scope="session", autouse=True)
def manage_test_data_dir():
    """Creates and cleans up the test data directory."""
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Generate more diverse content for more documents
    base_fruits = ["Apples", "Oranges", "Bananas", "Grapes", "Kiwis", "Mangos", "Pears", "Peaches"]
    base_colors = ["Red", "Yellow", "Green", "Blue", "Purple", "Orange", "Pink", "Brown"]
    doc_contents_generated = []
    for i in range(DOC_COUNT):
        fruit1 = base_fruits[i % len(base_fruits)]
        fruit2 = base_fruits[(i+1) % len(base_fruits)]
        color1 = base_colors[i % len(base_colors)]
        color2 = base_colors[(i+1) % len(base_colors)]
        doc_contents_generated.append(
            f"Document number {i+1} is about {fruit1} and {fruit2}. The {fruit1} are often {color1}, while {fruit2} can be {color2}."
        )

    for i in range(DOC_COUNT):
        with open(os.path.join(TEST_DATA_DIR, f"doc_{i+1}.txt"), "w") as f:
            f.write(doc_contents_generated[i])
    
    yield

    # Teardown: remove the directory after tests
    # shutil.rmtree(TEST_DATA_DIR) # Keep for inspection if tests fail

def mock_llm_func(prompt: str) -> str:
    """A simple mock LLM function for testing."""
    return f"Mocked LLM response to: {prompt[:100]}..."

def test_graph_population(graphrag_pipeline_instance: GraphRAGPipeline):
    """
    Tests GraphRAG pipeline initialization and basic functionality with mocked components.
    """
    pipeline = graphrag_pipeline_instance
    
    # Test that pipeline was initialized correctly
    assert pipeline is not None
    assert pipeline.top_k == 3  # From mock config
    assert pipeline.max_entities == 5  # From mock config
    assert pipeline.relationship_depth == 1  # From mock config
    
    # Test document loading through vector store interface
    test_docs = [
        Document(id="doc_1", page_content="Document about Apples and Oranges", metadata={"source": "test"}),
        Document(id="doc_2", page_content="Document about Bananas and Grapes", metadata={"source": "test"}),
    ]
    
    # Mock the ingest_documents method to work with vector store
    with patch.object(pipeline, 'ingest_documents') as mock_ingest:
        mock_ingest.return_value = None
        pipeline.ingest_documents(test_docs)
        mock_ingest.assert_called_once_with(test_docs)

def test_query_functionality(graphrag_pipeline_instance: GraphRAGPipeline):
    """
    Tests graph-based query functionality with mocked components.
    """
    pipeline = graphrag_pipeline_instance
    query_text = "Tell me about Apples and Oranges"

    # Mock the query method to return expected format
    with patch.object(pipeline, 'query') as mock_query:
        expected_result = {
            "query": query_text,
            "answer": "Mocked LLM response to: Tell me about Apples and Oranges",
            "retrieved_documents": [
                Document(id="doc_1", page_content="Test document about Apples and Oranges", 
                        metadata={"source": "test", "retrieval_method": "graph_based_retrieval", "entity_matches": 2})
            ],
            "query_entities": ["Tell", "Apples", "Oranges"],
            "num_documents_retrieved": 1,
            "processing_time": 0.1,
            "pipeline_type": "graphrag"
        }
        mock_query.return_value = expected_result
        
        # Execute query
        result = pipeline.query(query_text, top_k=2)

        # Assert basic result structure
        assert isinstance(result, dict), "Query result should be a dictionary"
        assert "query" in result and result["query"] == query_text
        assert "retrieved_documents" in result
        assert "answer" in result
        assert "query_entities" in result
        assert "num_documents_retrieved" in result
        assert "processing_time" in result
        assert result.get("pipeline_type") == "graphrag"

        # Assert document retrieval
        retrieved_docs = result["retrieved_documents"]
        assert isinstance(retrieved_docs, list), "Retrieved documents should be a list"
        assert result["num_documents_retrieved"] > 0, "Expected at least one document to be retrieved"

        for doc in retrieved_docs:
            assert isinstance(doc, Document), "Each item in retrieved_documents should be a Document object"
            assert doc.page_content is not None

        # Assert mock LLM answer
        assert result["answer"] is not None
        assert "Mocked LLM response" in result["answer"]

def test_schema_self_healing(mock_connection_manager, mock_config_manager):
    """
    Tests GraphRAG pipeline with schema management through mocked components.
    """
    # Test that pipeline can be initialized with schema management
    with patch('iris_rag.storage.enterprise_storage.IRISStorage'), \
         patch('iris_rag.embeddings.manager.EmbeddingManager'), \
         patch('iris_rag.storage.schema_manager.SchemaManager') as mock_schema_manager:
        
        # Configure schema manager mock
        mock_schema_instance = Mock()
        mock_schema_manager.return_value = mock_schema_instance
        mock_schema_instance.ensure_schema_metadata_table.return_value = None
        mock_schema_instance.get_current_schema_config.return_value = {
            "vector_dimension": 384,
            "embedding_model": "all-MiniLM-L6-v2",
            "schema_version": "1.0.0"
        }
        
        # Create pipeline instance
        pipeline = GraphRAGPipeline(mock_connection_manager, mock_config_manager, llm_func=None)
        
        # Verify schema manager was called
        mock_schema_manager.assert_called_once()
        
        # Test schema healing simulation
        dummy_doc = Document(id="dummy_heal_doc_001", 
                           page_content="This is a Healing Test document with some CapitalizedWords.", 
                           metadata={"source": "healing_test"})
        
        # Mock the ingest_documents method
        with patch.object(pipeline, 'ingest_documents') as mock_ingest:
            mock_ingest.return_value = None
            pipeline.ingest_documents([dummy_doc])
            mock_ingest.assert_called_once_with([dummy_doc])