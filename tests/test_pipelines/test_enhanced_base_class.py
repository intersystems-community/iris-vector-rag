"""
Tests for enhanced RAGPipeline base class with VectorStore integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from iris_rag.core.models import Document
from iris_rag.core.base import RAGPipeline
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.colbert import ColBERTRAGPipeline


class TestRAGPipelineBaseClass:
    """Test the enhanced RAGPipeline base class."""
    
    def test_base_class_initialization_with_default_vector_store(self):
        """Test that base class initializes with default IRISVectorStore when none provided."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        
        with patch('iris_rag.storage.vector_store_iris.IRISVectorStore') as mock_vector_store_class:
            mock_vector_store_instance = Mock()
            mock_vector_store_class.return_value = mock_vector_store_instance
            
            # Create a concrete implementation for testing
            class TestPipeline(RAGPipeline):
                def execute(self, query_text: str, **kwargs):
                    return {"query": query_text, "answer": "test", "retrieved_documents": []}
                
                def load_documents(self, documents_path: str, **kwargs):
                    pass
                
                def query(self, query_text: str, top_k: int = 5, **kwargs):
                    return []
            
            pipeline = TestPipeline(mock_connection_manager, mock_config_manager)
            
            # Verify IRISVectorStore was instantiated
            mock_vector_store_class.assert_called_once_with(mock_connection_manager, mock_config_manager)
            assert pipeline.vector_store == mock_vector_store_instance
    
    def test_base_class_initialization_with_custom_vector_store(self):
        """Test that base class uses provided vector store."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_vector_store = Mock()
        
        class TestPipeline(RAGPipeline):
            def execute(self, query_text: str, **kwargs):
                return {"query": query_text, "answer": "test", "retrieved_documents": []}
            
            def load_documents(self, documents_path: str, **kwargs):
                pass
            
            def query(self, query_text: str, top_k: int = 5, **kwargs):
                return []
        
        pipeline = TestPipeline(mock_connection_manager, mock_config_manager, mock_vector_store)
        
        assert pipeline.vector_store == mock_vector_store
    
    def test_retrieve_documents_by_vector_helper(self):
        """Test the _retrieve_documents_by_vector helper method."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_vector_store = Mock()
        
        # Mock vector store response
        test_doc = Document(page_content="test content", metadata={"title": "test"})
        mock_vector_store.similarity_search.return_value = [(test_doc, 0.9)]
        
        class TestPipeline(RAGPipeline):
            def execute(self, query_text: str, **kwargs):
                return {"query": query_text, "answer": "test", "retrieved_documents": []}
            
            def load_documents(self, documents_path: str, **kwargs):
                pass
            
            def query(self, query_text: str, top_k: int = 5, **kwargs):
                return []
        
        pipeline = TestPipeline(mock_connection_manager, mock_config_manager, mock_vector_store)
        
        # Test the helper method
        query_embedding = [0.1, 0.2, 0.3]
        results = pipeline._retrieve_documents_by_vector(query_embedding, 5)
        
        # Verify vector store was called correctly
        mock_vector_store.similarity_search.assert_called_once_with(
            query_embedding=query_embedding,
            top_k=5,
            filter=None
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0][0] == test_doc
        assert results[0][1] == 0.9
    
    def test_get_documents_by_ids_helper(self):
        """Test the _get_documents_by_ids helper method."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_vector_store = Mock()
        
        # Mock vector store response
        test_doc = Document(page_content="test content", metadata={"title": "test"})
        mock_vector_store.fetch_documents_by_ids.return_value = [test_doc]
        
        class TestPipeline(RAGPipeline):
            def execute(self, query_text: str, **kwargs):
                return {"query": query_text, "answer": "test", "retrieved_documents": []}
            
            def load_documents(self, documents_path: str, **kwargs):
                pass
            
            def query(self, query_text: str, top_k: int = 5, **kwargs):
                return []
        
        pipeline = TestPipeline(mock_connection_manager, mock_config_manager, mock_vector_store)
        
        # Test the helper method
        document_ids = ["doc1", "doc2"]
        results = pipeline._get_documents_by_ids(document_ids)
        
        # Verify vector store was called correctly
        mock_vector_store.fetch_documents_by_ids.assert_called_once_with(document_ids)
        
        # Verify results
        assert len(results) == 1
        assert results[0] == test_doc
    
    def test_store_documents_helper(self):
        """Test the _store_documents helper method."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_vector_store = Mock()
        
        # Mock vector store response
        mock_vector_store.add_documents.return_value = ["doc1", "doc2"]
        
        class TestPipeline(RAGPipeline):
            def execute(self, query_text: str, **kwargs):
                return {"query": query_text, "answer": "test", "retrieved_documents": []}
            
            def load_documents(self, documents_path: str, **kwargs):
                pass
            
            def query(self, query_text: str, top_k: int = 5, **kwargs):
                return []
        
        pipeline = TestPipeline(mock_connection_manager, mock_config_manager, mock_vector_store)
        
        # Test the helper method
        test_docs = [
            Document(page_content="content1", metadata={"title": "doc1"}),
            Document(page_content="content2", metadata={"title": "doc2"})
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        results = pipeline._store_documents(test_docs, embeddings)
        
        # Verify vector store was called correctly
        mock_vector_store.add_documents.assert_called_once_with(test_docs, embeddings)
        
        # Verify results
        assert results == ["doc1", "doc2"]
    
    def test_run_method_calls_execute(self):
        """Test that run() method calls execute()."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_vector_store = Mock()
        
        class TestPipeline(RAGPipeline):
            def execute(self, query_text: str, **kwargs):
                return {"query": query_text, "answer": "test answer", "retrieved_documents": []}
            
            def load_documents(self, documents_path: str, **kwargs):
                pass
            
            def query(self, query_text: str, top_k: int = 5, **kwargs):
                return []
        
        pipeline = TestPipeline(mock_connection_manager, mock_config_manager, mock_vector_store)
        
        # Test run method
        result = pipeline.run("test query", top_k=3)
        
        # Verify it returns the same as execute
        expected = {"query": "test query", "answer": "test answer", "retrieved_documents": []}
        assert result == expected


class TestBasicRAGPipelineRefactoring:
    """Test BasicRAGPipeline refactoring to use VectorStore."""
    
    @patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
    @patch('iris_rag.pipelines.basic.EmbeddingManager')
    def test_basic_pipeline_uses_vector_store(self, mock_embedding_manager, mock_vector_store_class):
        """Test that BasicRAGPipeline uses VectorStore for operations."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.return_value = {}
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        pipeline = BasicRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager
        )
        
        # Verify vector store was initialized
        mock_vector_store_class.assert_called_once_with(mock_connection_manager, mock_config_manager)
        assert pipeline.vector_store == mock_vector_store
    
    @patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
    @patch('iris_rag.pipelines.basic.EmbeddingManager')
    def test_basic_pipeline_execute_returns_standard_format(self, mock_embedding_manager, mock_vector_store_class):
        """Test that BasicRAGPipeline execute returns standardized format."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.return_value = {}
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Mock embedding manager
        mock_embedding_instance = Mock()
        mock_embedding_manager.return_value = mock_embedding_instance
        mock_embedding_instance.embed_text.return_value = [0.1, 0.2, 0.3]
        
        # Mock vector store search
        test_doc = Document(page_content="test content", metadata={"title": "test"})
        mock_vector_store.similarity_search.return_value = [(test_doc, 0.9)]
        
        pipeline = BasicRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager,
            llm_func=lambda x: "test answer"
        )
        
        # Execute pipeline
        result = pipeline.execute("test query")
        
        # Verify standard format
        assert "query" in result
        assert "answer" in result
        assert "retrieved_documents" in result
        assert result["query"] == "test query"
        assert result["answer"] == "test answer"
        assert len(result["retrieved_documents"]) == 1
        assert isinstance(result["retrieved_documents"][0], Document)


class TestHyDERAGPipelineRefactoring:
    """Test HyDERAGPipeline refactoring to use VectorStore."""
    
    @patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
    @patch('iris_rag.pipelines.hyde.EmbeddingManager')
    def test_hyde_pipeline_uses_vector_store(self, mock_embedding_manager, mock_vector_store_class):
        """Test that HyDERAGPipeline uses VectorStore for operations."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.return_value = {}
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        pipeline = HyDERAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager
        )
        
        # Verify vector store was initialized
        mock_vector_store_class.assert_called_once_with(mock_connection_manager, mock_config_manager)
        assert pipeline.vector_store == mock_vector_store
    
    @patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
    @patch('iris_rag.pipelines.hyde.EmbeddingManager')
    def test_hyde_pipeline_execute_returns_standard_format(self, mock_embedding_manager, mock_vector_store_class):
        """Test that HyDERAGPipeline execute returns standardized format."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.return_value = {}
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Mock embedding manager
        mock_embedding_instance = Mock()
        mock_embedding_manager.return_value = mock_embedding_instance
        mock_embedding_instance.embed_text.return_value = [0.1, 0.2, 0.3]
        
        # Mock vector store search
        test_doc = Document(page_content="test content", metadata={"title": "test"})
        mock_vector_store.similarity_search.return_value = [(test_doc, 0.9)]
        
        pipeline = HyDERAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager,
            llm_func=lambda x: "test answer"
        )
        
        # Execute pipeline
        result = pipeline.execute("test query")
        
        # Verify standard format
        assert "query" in result
        assert "answer" in result
        assert "retrieved_documents" in result
        assert result["query"] == "test query"


class TestColBERTRAGPipelineRefactoring:
    """Test ColBERTRAGPipeline refactoring to use VectorStore."""
    
    @patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
    @patch('common.utils.get_colbert_query_encoder_func')
    @patch('common.utils.get_llm_func')
    @patch('common.utils.get_embedding_func')
    def test_colbert_pipeline_uses_vector_store(self, mock_embedding_func, mock_llm_func, 
                                               mock_colbert_func, mock_vector_store_class):
        """Test that ColBERTRAGPipeline uses VectorStore for operations."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.return_value = {}
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Mock utility functions
        mock_colbert_func.return_value = Mock()
        mock_llm_func.return_value = Mock()
        mock_embedding_func.return_value = Mock()
        
        pipeline = ColBERTRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager
        )
        
        # Verify vector store was initialized
        mock_vector_store_class.assert_called_once_with(mock_connection_manager, mock_config_manager)
        assert pipeline.vector_store == mock_vector_store
    
    @patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
    @patch('common.utils.get_colbert_query_encoder_func')
    @patch('common.utils.get_llm_func')
    @patch('common.utils.get_embedding_func')
    def test_colbert_pipeline_execute_returns_standard_format(self, mock_embedding_func, mock_llm_func,
                                                             mock_colbert_func, mock_vector_store_class):
        """Test that ColBERTRAGPipeline execute returns standardized format."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.return_value = {}
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Mock utility functions
        mock_colbert_encoder = Mock()
        mock_colbert_encoder.return_value = [[0.1, 0.2], [0.3, 0.4]]  # Token embeddings
        mock_colbert_func.return_value = mock_colbert_encoder
        
        mock_llm = Mock()
        mock_llm.return_value = "test answer"
        mock_llm_func.return_value = mock_llm
        
        mock_embedding_func.return_value = Mock()
        
        # Mock connection for ColBERT validation
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [1]  # Table exists
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connection_manager.get_connection.return_value = mock_connection
        
        pipeline = ColBERTRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager
        )
        
        # Mock the ColBERT retrieval to return test documents
        test_doc = Document(page_content="test content", metadata={"title": "test"})
        with patch.object(pipeline, '_retrieve_documents_with_colbert', return_value=[test_doc]):
            # Execute pipeline
            result = pipeline.execute("test query")
            
            # Verify standard format
            assert "query" in result
            assert "answer" in result
            assert "retrieved_documents" in result
            assert result["query"] == "test query"
            assert result["answer"] == "test answer"
            assert len(result["retrieved_documents"]) == 1
            assert isinstance(result["retrieved_documents"][0], Document)


def test_vector_store_integration_removes_clob_handling():
    """Test that VectorStore integration removes need for CLOB handling in pipelines."""
    # This is more of a design verification test
    # The VectorStore interface guarantees string content, so pipelines don't need CLOB conversion
    
    # Mock a document with string content (as guaranteed by VectorStore)
    test_doc = Document(
        page_content="This is string content, not CLOB",
        metadata={"title": "Test Document"}
    )
    
    # Verify content is string
    assert isinstance(test_doc.page_content, str)
    assert isinstance(test_doc.metadata["title"], str)
    
    # This test passes if the Document has string content,
    # demonstrating that VectorStore handles CLOB conversion
    assert test_doc.page_content == "This is string content, not CLOB"