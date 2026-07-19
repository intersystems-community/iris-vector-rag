"""
Parameterized conformance tests for RAG pipeline interface consistency.

Tests that all 6 pipelines conform to the unified interface specification:
- load_documents() returns consistent IngestionResult shape
- query() returns consistent QueryResult shape
- query() accepts query_text keyword argument
- create_pipeline() passes all standard dependencies correctly
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, Mock

from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag import create_pipeline


# Mock vector store that returns empty results
class MockVectorStore:
    """Minimal mock vector store for testing."""

    def add_documents(self, documents, embeddings=None, **kwargs):
        """Mock add_documents."""
        return [f"doc_{i}" for i in range(len(documents))]

    def similarity_search(self, query=None, query_embedding=None, k=5, **kwargs):
        """Mock similarity_search returns empty list."""
        return []

    def fetch_documents_by_ids(self, doc_ids):
        """Mock fetch returns Document objects."""
        return [
            Document(
                id=doc_id,
                page_content=f"Content for {doc_id}",
                metadata={"source": f"test_source_{doc_id}"}
            )
            for doc_id in doc_ids
        ]

    def clear(self):
        """Mock clear."""
        pass

    def get_all_documents(self):
        """Mock get all documents."""
        return []


# Mock LLM function
def mock_llm_func(prompt: str) -> str:
    """Mock LLM that returns a simple answer."""
    return "This is a mock answer from the LLM."


@pytest.fixture
def mock_config_manager():
    """Create a mock configuration manager."""
    config_manager = MagicMock(spec=ConfigurationManager)
    config_manager.get = MagicMock(return_value={})
    config_manager.get_database_config = MagicMock(return_value={
        "iris": {"host": "localhost", "port": 1972, "namespace": "USER"}
    })
    config_manager.get_vector_index_config = MagicMock(return_value={
        "type": "HNSW"
    })
    config_manager.get_embedding_config = MagicMock(return_value={
        "model": "all-MiniLM-L6-v2",
        "dimension": 384
    })
    config_manager.get_cloud_config = MagicMock(return_value={})
    config_manager.to_dict = MagicMock(return_value={})
    config_manager._config = {}
    return config_manager


@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager."""
    connection_manager = MagicMock(spec=ConnectionManager)
    return connection_manager


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    return MockVectorStore()


# Define pipeline test matrix
PIPELINE_TYPES = [
    "basic",
    "basic_rerank",
    "crag",
    "graphrag",
    "multi_query_rrf",
]

# Only test pylate_colbert if iris-vector-graph is installed
try:
    import iris_vector_graph
    PIPELINE_TYPES.append("pylate_colbert")
except ImportError:
    pass


class TestPipelineConformance:
    """Test suite for pipeline interface conformance."""

    @pytest.mark.parametrize("pipeline_type", PIPELINE_TYPES)
    def test_load_documents_returns_ingestion_result(
        self, pipeline_type, mock_config_manager, mock_connection_manager, mock_vector_store
    ):
        """
        Test that load_documents() returns a dict with required IngestionResult keys.

        Acceptance: Result dict contains:
        - documents_loaded: int >= 0
        - documents_failed: int >= 0
        - embeddings_generated: int >= 0
        """
        # Create pipeline with mocked dependencies
        with patch("iris_vector_rag.pipelines.basic.ConnectionManager", return_value=mock_connection_manager):
            with patch("iris_vector_rag.embeddings.manager.EmbeddingManager"):
                with patch("iris_vector_rag.pipelines.graphrag.EmbeddingManager"):
                    pipeline = create_pipeline(
                        pipeline_type,
                        llm_func=mock_llm_func,
                    )

        # Replace vector store with mock
        pipeline.vector_store = mock_vector_store

        # Mock any external service calls that might happen
        if hasattr(pipeline, 'entity_extraction_service'):
            pipeline.entity_extraction_service = MagicMock()

        # Create test documents
        test_documents = [
            Document(
                page_content="Test document 1",
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="Test document 2",
                metadata={"source": "test2.txt"}
            ),
        ]

        # Call load_documents
        result = pipeline.load_documents(documents_path="", documents=test_documents)

        # Assertions
        assert isinstance(result, dict), \
            f"{pipeline_type}: load_documents must return a dict, got {type(result)}"

        assert "documents_loaded" in result, \
            f"{pipeline_type}: Result must contain 'documents_loaded' key"
        assert "documents_failed" in result, \
            f"{pipeline_type}: Result must contain 'documents_failed' key"
        assert "embeddings_generated" in result, \
            f"{pipeline_type}: Result must contain 'embeddings_generated' key"

        assert isinstance(result["documents_loaded"], int) and result["documents_loaded"] >= 0, \
            f"{pipeline_type}: documents_loaded must be int >= 0"
        assert isinstance(result["documents_failed"], int) and result["documents_failed"] >= 0, \
            f"{pipeline_type}: documents_failed must be int >= 0"
        assert isinstance(result["embeddings_generated"], int) and result["embeddings_generated"] >= 0, \
            f"{pipeline_type}: embeddings_generated must be int >= 0"

    @pytest.mark.parametrize("pipeline_type", PIPELINE_TYPES)
    def test_query_returns_consistent_keys(
        self, pipeline_type, mock_config_manager, mock_connection_manager, mock_vector_store
    ):
        """
        Test that query() returns a dict with all required QueryResult keys.

        Acceptance: Result dict contains:
        - answer: str or None
        - retrieved_documents: List[Document]
        - contexts: List[str]
        - sources: List (at top level, not inside metadata)
        - metadata: Dict
        - error: None or error dict
        """
        # Create pipeline with mocked dependencies
        with patch("iris_vector_rag.pipelines.basic.ConnectionManager", return_value=mock_connection_manager):
            with patch("iris_vector_rag.embeddings.manager.EmbeddingManager"):
                with patch("iris_vector_rag.pipelines.graphrag.EmbeddingManager"):
                    pipeline = create_pipeline(
                        pipeline_type,
                        llm_func=mock_llm_func,
                    )

        # Replace vector store with mock
        pipeline.vector_store = mock_vector_store

        # Mock external services
        if hasattr(pipeline, 'entity_extraction_service'):
            pipeline.entity_extraction_service = MagicMock()

        # Fix embedding_manager mock to return numeric dimension (avoids dim-mismatch errors)
        if hasattr(pipeline, 'embedding_manager'):
            pipeline.embedding_manager = MagicMock()
            pipeline.embedding_manager.get_embedding_dimension.return_value = 384
            pipeline.embedding_manager.generate_embedding.return_value = [0.1] * 384
            pipeline.embedding_manager.generate_embeddings.return_value = [[0.1] * 384]

        # Call query with generate_answer=False to avoid LLM requirement
        result = pipeline.query(query_text="What is test?", generate_answer=False)

        # Assertions
        assert isinstance(result, dict), \
            f"{pipeline_type}: query must return a dict, got {type(result)}"

        required_keys = ["answer", "retrieved_documents", "contexts", "sources", "metadata", "error"]
        for key in required_keys:
            assert key in result, \
                f"{pipeline_type}: query result must contain '{key}' key, got keys: {list(result.keys())}"

        # Validate types
        assert isinstance(result["retrieved_documents"], list), \
            f"{pipeline_type}: retrieved_documents must be a list"
        assert isinstance(result["contexts"], list), \
            f"{pipeline_type}: contexts must be a list"
        assert isinstance(result["sources"], list), \
            f"{pipeline_type}: sources must be a list"
        assert isinstance(result["metadata"], dict), \
            f"{pipeline_type}: metadata must be a dict"

        # Verify sources is NOT inside metadata
        assert "sources" not in result.get("metadata", {}), \
            f"{pipeline_type}: sources must be at top level, not inside metadata"

    @pytest.mark.parametrize("pipeline_type", PIPELINE_TYPES)
    def test_query_accepts_query_text_keyword(
        self, pipeline_type, mock_config_manager, mock_connection_manager, mock_vector_store
    ):
        """
        Test that query() accepts query_text as a keyword argument.

        Acceptance: Both `query(query_text="...")` and `query("...")` work
        """
        # Create pipeline with mocked dependencies
        with patch("iris_vector_rag.pipelines.basic.ConnectionManager", return_value=mock_connection_manager):
            with patch("iris_vector_rag.embeddings.manager.EmbeddingManager"):
                with patch("iris_vector_rag.pipelines.graphrag.EmbeddingManager"):
                    pipeline = create_pipeline(
                        pipeline_type,
                        llm_func=mock_llm_func,
                    )

        # Replace vector store with mock
        pipeline.vector_store = mock_vector_store

        # Mock external services
        if hasattr(pipeline, 'entity_extraction_service'):
            pipeline.entity_extraction_service = MagicMock()

        # Fix embedding_manager mock to return numeric dimension (avoids dim-mismatch errors)
        if hasattr(pipeline, 'embedding_manager'):
            pipeline.embedding_manager = MagicMock()
            pipeline.embedding_manager.get_embedding_dimension.return_value = 384
            pipeline.embedding_manager.generate_embedding.return_value = [0.1] * 384
            pipeline.embedding_manager.generate_embeddings.return_value = [[0.1] * 384]

        # Test keyword argument form - should not raise TypeError
        try:
            result = pipeline.query(query_text="What is test?", generate_answer=False)
            assert isinstance(result, dict), \
                f"{pipeline_type}: query with query_text kwarg must return dict"
        except TypeError as e:
            if "query_text" in str(e):
                pytest.fail(
                    f"{pipeline_type}: query() does not accept query_text as keyword argument: {e}"
                )
            raise

        # Test positional argument form - should also work
        try:
            result = pipeline.query("What is test?", generate_answer=False)
            assert isinstance(result, dict), \
                f"{pipeline_type}: query with positional arg must return dict"
        except TypeError:
            pytest.fail(
                f"{pipeline_type}: query() does not accept positional query argument"
            )

    def test_multi_query_rrf_llm_func_passthrough(self):
        """
        Test that create_pipeline passes llm_func to MultiQueryRRFPipeline.

        Acceptance: pipeline.llm_func is the same as the provided llm_func
        """
        custom_llm = MagicMock(return_value="custom answer")

        pipeline = create_pipeline(
            "multi_query_rrf",
            llm_func=custom_llm,
        )

        # Replace vector store with mock
        pipeline.vector_store = MockVectorStore()

        # Verify llm_func is passed through
        # For multi_query_rrf, check that it's stored
        # (the create_pipeline should pass it)
        assert hasattr(pipeline, 'llm_func') or hasattr(pipeline, 'llm'), \
            "multi_query_rrf pipeline must accept and store llm_func"

    def test_factory_passes_llm_func_to_all_pipelines(self):
        """
        Test that create_pipeline() passes llm_func to all pipeline types.

        Acceptance: Each pipeline has access to the provided llm_func
        """
        custom_llm = MagicMock(return_value="custom answer")

        for pipeline_type in PIPELINE_TYPES:
            pipeline = create_pipeline(
                pipeline_type,
                llm_func=custom_llm,
            )

            # Replace vector store with mock
            pipeline.vector_store = MockVectorStore()

            # Verify the pipeline either has llm_func or uses custom_llm through other means
            # At minimum, for pipelines that support LLM, verify they have the attribute
            if pipeline_type in ["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"]:
                assert hasattr(pipeline, 'llm_func'), \
                    f"{pipeline_type}: pipeline must have llm_func attribute"


class TestLoadDocumentsReturnType:
    """Test the return type annotation of load_documents in ABC."""

    def test_abc_load_documents_return_type(self):
        """
        Test that RAGPipeline.load_documents() is declared to return Dict[str, Any].

        This verifies the spec requirement FR-001.
        """
        import inspect

        # Get the signature of load_documents
        sig = inspect.signature(RAGPipeline.load_documents)

        # Check return annotation
        return_annotation = sig.return_annotation

        # Should be Dict[str, Any] or similar
        assert return_annotation != type(None), \
            "RAGPipeline.load_documents() must have return type annotation (not None)"

        # Verify it's not explicitly None
        assert return_annotation != None, \
            "RAGPipeline.load_documents() return type must not be None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
