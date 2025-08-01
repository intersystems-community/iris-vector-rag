"""
Test suite for refactored RAG pipelines to verify standardized interface compliance.

This module tests that all refactored pipelines:
1. Use the standardized execute() method
2. Return the correct format with Document objects
3. Handle VectorStore integration properly
4. Maintain backward compatibility
"""

import pytest
from unittest.mock import Mock, patch
from iris_rag.core.models import Document
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline


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
            'pipelines:crag:top_k': 5,
            'pipelines:noderag:top_k': 5,
            'pipelines:graphrag:top_k': 5,
            'pipelines:hybrid_ifind:top_k': 5,
            'pipelines:colbert:top_k': 5,
        }
        return config_map.get(key, default if default is not None else {})
    
    mock_manager.get.side_effect = mock_get
    mock_manager.get_embedding_config.return_value = {'model': 'test', 'dimension': 384}
    mock_manager.get_vector_index_config.return_value = {'type': 'HNSW'}
    return mock_manager


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    mock_store.similarity_search.return_value = [
        (Document(id="test1", page_content="Test content 1", metadata={}), 0.9),
        (Document(id="test2", page_content="Test content 2", metadata={}), 0.8)
    ]
    mock_store.add_documents.return_value = ["test1", "test2"]
    mock_store.fetch_documents_by_ids.return_value = [
        Document(id="test1", page_content="Test content 1", metadata={}),
        Document(id="test2", page_content="Test content 2", metadata={})
    ]
    return mock_store


@pytest.fixture
def mock_embedding_func():
    """Mock embedding function."""
    return lambda texts: [[0.1, 0.2, 0.3] * 128 for _ in texts]


@pytest.fixture
def mock_llm_func():
    """Mock LLM function."""
    return lambda prompt: "This is a test answer based on the provided context."


class TestCRAGPipeline:
    """Test CRAG pipeline refactoring."""
    
    def test_crag_initialization_with_vector_store(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test CRAG pipeline initializes correctly with vector store."""
        with patch('iris_rag.pipelines.crag.RetrievalEvaluator'):
            pipeline = CRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
        
        assert pipeline.vector_store == mock_vector_store
        assert hasattr(pipeline, 'execute')
        assert hasattr(pipeline, 'load_documents')
        assert hasattr(pipeline, 'query')
    
    def test_crag_execute_method(self, mock_connection_manager, mock_config_manager, mock_vector_store, mock_embedding_func, mock_llm_func):
        """Test CRAG execute method returns standardized format."""
        with patch('iris_rag.pipelines.crag.RetrievalEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator.evaluate.return_value = "confident"
            mock_evaluator_class.return_value = mock_evaluator
            
            pipeline = CRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store,
                embedding_func=mock_embedding_func,
                llm_func=mock_llm_func
            )
            
            result = pipeline.query("test query")
            
            assert isinstance(result, dict)
            assert "query" in result
            assert "answer" in result
            assert "retrieved_documents" in result
            assert isinstance(result["retrieved_documents"], list)
    
    def test_crag_load_documents_uses_vector_store(self, mock_connection_manager, mock_config_manager, mock_vector_store, mock_embedding_func):
        """Test CRAG load_documents uses vector store helper."""
        with patch('iris_rag.pipelines.crag.RetrievalEvaluator'):
            pipeline = CRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store,
                embedding_func=mock_embedding_func
            )
            
            test_docs = [Document(id="test", page_content="test content", metadata={})]
            pipeline.load_documents("dummy_path", documents=test_docs)
            
            # Verify vector store was called
            mock_vector_store.add_documents.assert_called_once()


class TestNodeRAGPipeline:
    """Test NodeRAG pipeline refactoring."""
    
    def test_noderag_initialization_with_vector_store(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test NodeRAG pipeline initializes correctly with vector store."""
        with patch('iris_rag.pipelines.noderag.EmbeddingManager'), \
             patch('common.utils.get_llm_func'):
            pipeline = NodeRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
        
        assert pipeline.vector_store == mock_vector_store
        assert hasattr(pipeline, 'execute')
        assert hasattr(pipeline, 'load_documents')
        assert hasattr(pipeline, 'query')
    
    def test_noderag_execute_method(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test NodeRAG execute method returns standardized format."""
        with patch('iris_rag.pipelines.noderag.EmbeddingManager') as mock_em, \
             patch('common.utils.get_llm_func') as mock_llm:
            
            mock_embedding_manager = Mock()
            mock_embedding_manager.embed_text.return_value = [0.1, 0.2, 0.3]
            mock_em.return_value = mock_embedding_manager
            mock_llm.return_value = lambda x: "test answer"
            
            pipeline = NodeRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
            
            result = pipeline.query("test query")
            
            assert isinstance(result, dict)
            assert "query" in result
            assert "answer" in result
            assert "retrieved_documents" in result


class TestGraphRAGPipeline:
    """Test GraphRAG pipeline refactoring."""
    
    def test_graphrag_initialization_with_vector_store(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test GraphRAG pipeline initializes correctly with vector store."""
        with patch('iris_rag.pipelines.graphrag.IRISStorage'), \
             patch('iris_rag.pipelines.graphrag.EmbeddingManager'), \
             patch('iris_rag.pipelines.graphrag.SchemaManager'):
            pipeline = GraphRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
        
        assert pipeline.vector_store == mock_vector_store
        assert hasattr(pipeline, 'execute')
        assert hasattr(pipeline, 'load_documents')
        assert hasattr(pipeline, 'query')
    
    def test_graphrag_execute_method(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test GraphRAG execute method returns standardized format."""
        with patch('iris_rag.pipelines.graphrag.IRISStorage'), \
             patch('iris_rag.pipelines.graphrag.EmbeddingManager') as mock_em, \
             patch('iris_rag.pipelines.graphrag.SchemaManager'):
            
            mock_embedding_manager = Mock()
            mock_embedding_manager.embed_text.return_value = [0.1, 0.2, 0.3]
            mock_em.return_value = mock_embedding_manager
            
            pipeline = GraphRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store,
                llm_func=lambda x: "test answer"
            )
            
            result = pipeline.query("test query")
            
            assert isinstance(result, dict)
            assert "query" in result
            assert "answer" in result
            assert "retrieved_documents" in result


class TestHybridIFindRAGPipeline:
    """Test HybridIFindRAG pipeline refactoring."""
    
    def test_hybrid_ifind_initialization_with_vector_store(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test HybridIFindRAG pipeline initializes correctly with vector store."""
        with patch('iris_rag.pipelines.hybrid_ifind.IRISStorage'), \
             patch('iris_rag.pipelines.hybrid_ifind.EmbeddingManager'):
            pipeline = HybridIFindRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
        
        assert pipeline.vector_store == mock_vector_store
        assert hasattr(pipeline, 'execute')
        assert hasattr(pipeline, 'load_documents')
        assert hasattr(pipeline, 'query')
    
    def test_hybrid_ifind_execute_returns_documents(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test HybridIFindRAG execute method returns Document objects."""
        from iris_rag.core.models import Document
        
        with patch('iris_rag.pipelines.hybrid_ifind.IRISStorage'), \
             patch('iris_rag.pipelines.hybrid_ifind.EmbeddingManager') as mock_em:
            
            mock_embedding_manager = Mock()
            mock_embedding_manager.embed_text.return_value = [0.1, 0.2, 0.3]
            mock_em.return_value = mock_embedding_manager
            
            # Mock the vector store's hybrid_search method to return proper format
            test_doc1 = Document(page_content="Content 1", metadata={"doc_id": "test1", "title": "Test 1"})
            test_doc2 = Document(page_content="Content 2", metadata={"doc_id": "test2", "title": "Test 2"})
            mock_vector_store.hybrid_search.return_value = [
                (test_doc1, 0.9),
                (test_doc2, 0.8)
            ]
            
            pipeline = HybridIFindRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store,
                llm_func=lambda x: "test answer"
            )
            
            result = pipeline.query("test query")
            
            assert isinstance(result, dict)
            assert "query" in result
            assert "answer" in result
            assert "retrieved_documents" in result
            assert isinstance(result["retrieved_documents"], list)
            
            # Verify returned documents are Document objects
            if result["retrieved_documents"]:
                assert isinstance(result["retrieved_documents"][0], Document)
                assert hasattr(result["retrieved_documents"][0], 'page_content')
                assert hasattr(result["retrieved_documents"][0], 'metadata')


class TestStandardizedInterface:
    """Test that all pipelines conform to standardized interface."""
    
    def test_crag_pipeline_has_execute_method(self, mock_connection_manager, mock_config_manager):
        """Test CRAG pipeline has execute method."""
        with patch('iris_rag.pipelines.crag.RetrievalEvaluator'):
            pipeline = CRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager
            )
        assert hasattr(pipeline, 'execute')
        assert callable(getattr(pipeline, 'execute'))

    def test_noderag_pipeline_has_execute_method(self, mock_connection_manager, mock_config_manager):
        """Test NodeRAG pipeline has execute method."""
        with patch('iris_rag.pipelines.noderag.EmbeddingManager'):
            pipeline = NodeRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager
            )
        assert hasattr(pipeline, 'execute')
        assert callable(getattr(pipeline, 'execute'))

    def test_graphrag_pipeline_has_execute_method(self, mock_connection_manager, mock_config_manager):
        """Test GraphRAG pipeline has execute method."""
        with patch('iris_rag.pipelines.graphrag.IRISStorage'), \
             patch('iris_rag.pipelines.graphrag.EmbeddingManager'), \
             patch('iris_rag.pipelines.graphrag.SchemaManager'):
            pipeline = GraphRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager
            )
        assert hasattr(pipeline, 'execute')
        assert callable(getattr(pipeline, 'execute'))

    def test_hybrid_ifind_pipeline_has_execute_method(self, mock_connection_manager, mock_config_manager):
        """Test HybridIFindRAG pipeline has execute method."""
        with patch('iris_rag.pipelines.hybrid_ifind.IRISStorage'), \
             patch('iris_rag.pipelines.hybrid_ifind.EmbeddingManager'):
            pipeline = HybridIFindRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager
            )
        assert hasattr(pipeline, 'execute')
        assert callable(getattr(pipeline, 'execute'))
    
    def test_evaluation_framework_compatibility(self, mock_connection_manager, mock_config_manager, mock_vector_store):
        """Test that pipelines work with evaluation framework's standardized call."""
        with patch('iris_rag.pipelines.crag.RetrievalEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator.evaluate.return_value = "confident"
            mock_evaluator_class.return_value = mock_evaluator
            
            pipeline = CRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                vector_store=mock_vector_store,
                embedding_func=lambda texts: [[0.1] * 384 for _ in texts],
                llm_func=lambda prompt: "test answer"
            )
            
            # This is how the evaluation framework now calls pipelines
            result = pipeline.query("test query")
            
            assert isinstance(result, dict)
            assert "query" in result
            assert "answer" in result
            assert "retrieved_documents" in result
            
            # Verify retrieved_documents contains Document objects with string content
            for doc in result["retrieved_documents"]:
                if isinstance(doc, Document):
                    assert isinstance(doc.page_content, str)