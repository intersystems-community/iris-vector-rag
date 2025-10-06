"""
E2E Tests for BasicRAG Pipeline

Comprehensive end-to-end tests with real IRIS database integration.
These tests exercise the full BasicRAG workflow from document loading to query generation.
"""

import pytest
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.storage.vector_store_iris import IRISVectorStore
from common.utils import get_llm_func


@pytest.fixture(scope="module")
def pipeline_dependencies():
    """Create real dependencies for E2E testing."""
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    llm_func = get_llm_func()
    vector_store = IRISVectorStore(connection_manager, config_manager)

    return {
        "config_manager": config_manager,
        "connection_manager": connection_manager,
        "llm_func": llm_func,
        "vector_store": vector_store,
    }


@pytest.fixture(scope="module")
def basic_pipeline(pipeline_dependencies):
    """Create BasicRAG pipeline instance."""
    return BasicRAGPipeline(
        connection_manager=pipeline_dependencies["connection_manager"],
        config_manager=pipeline_dependencies["config_manager"],
        llm_func=pipeline_dependencies["llm_func"],
        vector_store=pipeline_dependencies["vector_store"],
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            id="doc1",
            page_content="Python is a high-level programming language known for its simplicity and readability.",
        ),
        Document(
            id="doc2",
            page_content="Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms.",
        ),
        Document(
            id="doc3",
            page_content="Deep learning uses neural networks with multiple layers to learn from large datasets.",
        ),
        Document(
            id="doc4",
            page_content="Natural language processing enables computers to understand and generate human language.",
        ),
        Document(
            id="doc5",
            page_content="Computer vision allows machines to interpret and understand visual information from images.",
        ),
    ]


class TestBasicRAGPipelineDocumentLoading:
    """Test document loading functionality."""

    def test_load_documents_from_list(self, basic_pipeline, sample_documents):
        """Test loading documents from a list."""
        # load_documents returns None on success
        basic_pipeline.load_documents("", documents=sample_documents)

        # Verify documents can be queried
        result = basic_pipeline.query("Python programming", top_k=1, generate_answer=False)
        assert result is not None
        assert "contexts" in result

    def test_load_single_document(self, basic_pipeline):
        """Test loading a single document."""
        doc = Document(id="single1", page_content="Test document for single loading.")

        basic_pipeline.load_documents("", documents=[doc])

        # Verify document is searchable
        result = basic_pipeline.query("single loading", top_k=1, generate_answer=False)
        assert result is not None

    def test_load_documents_with_metadata(self, basic_pipeline):
        """Test loading documents with metadata."""
        docs = [
            Document(
                id="meta1",
                page_content="Document with metadata about testing frameworks.",
                metadata={"source": "test", "category": "example"},
            )
        ]

        basic_pipeline.load_documents("", documents=docs)

        # Verify document is searchable
        result = basic_pipeline.query("testing frameworks", top_k=1, generate_answer=False)
        assert result is not None

    def test_load_empty_document_list(self, basic_pipeline):
        """Test loading empty document list."""
        # Should not raise an error
        basic_pipeline.load_documents("", documents=[])

    def test_reload_same_documents(self, basic_pipeline, sample_documents):
        """Test reloading the same documents (upsert behavior)."""
        # First load
        basic_pipeline.load_documents("", documents=sample_documents[:2])

        # Reload same documents - should not error
        basic_pipeline.load_documents("", documents=sample_documents[:2])

        # Verify documents are still searchable
        result = basic_pipeline.query("Python", top_k=1, generate_answer=False)
        assert result is not None


class TestBasicRAGPipelineQuerying:
    """Test query functionality."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, basic_pipeline, sample_documents):
        """Load documents before each test."""
        basic_pipeline.load_documents("", documents=sample_documents)

    def test_simple_query(self, basic_pipeline):
        """Test a simple query."""
        result = basic_pipeline.query("What is Python?", top_k=3, generate_answer=True)

        assert result is not None
        assert "answer" in result
        assert "contexts" in result
        assert len(result["contexts"]) <= 3

    def test_query_with_top_k_variation(self, basic_pipeline):
        """Test queries with different top_k values."""
        for k in [1, 3, 5]:
            result = basic_pipeline.query("What is machine learning?", top_k=k, generate_answer=False)

            assert len(result["contexts"]) <= k

    def test_query_without_answer_generation(self, basic_pipeline):
        """Test query without LLM answer generation."""
        result = basic_pipeline.query("What is deep learning?", top_k=3, generate_answer=False)

        assert "contexts" in result
        # When generate_answer=False, answer field may be None or empty
        assert result.get("answer") is None or result.get("answer") == ""

    def test_query_with_answer_generation(self, basic_pipeline):
        """Test query with LLM answer generation."""
        result = basic_pipeline.query("What is NLP?", top_k=3, generate_answer=True)

        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_query_retrieval_quality(self, basic_pipeline):
        """Test that retrieved contexts are relevant."""
        result = basic_pipeline.query("What is Python programming?", top_k=3, generate_answer=False)

        # Check that Python document is retrieved
        contexts_text = " ".join(result["contexts"])
        assert "Python" in contexts_text or "python" in contexts_text.lower()

    def test_query_with_no_results(self, basic_pipeline):
        """Test query that should return few/no results."""
        result = basic_pipeline.query("quantum physics string theory", top_k=3, generate_answer=False)

        # Should still return a result structure
        assert "contexts" in result

    def test_multiple_sequential_queries(self, basic_pipeline):
        """Test multiple queries in sequence."""
        queries = [
            "What is Python?",
            "What is machine learning?",
            "What is deep learning?",
        ]

        for query in queries:
            result = basic_pipeline.query(query, top_k=3, generate_answer=False)
            assert "contexts" in result


class TestBasicRAGPipelineMetadata:
    """Test pipeline metadata and query results."""

    def test_query_result_structure(self, basic_pipeline, sample_documents):
        """Test that query results have expected structure."""
        basic_pipeline.load_documents("", documents=sample_documents)

        result = basic_pipeline.query("machine learning", top_k=3, generate_answer=False)

        # Verify result structure
        assert "contexts" in result
        assert "metadata" in result
        assert isinstance(result["contexts"], list)

    def test_query_metadata_tracking(self, basic_pipeline, sample_documents):
        """Test that query metadata is tracked."""
        basic_pipeline.load_documents("", documents=sample_documents)

        result = basic_pipeline.query("test query", top_k=3, generate_answer=False)

        # Check metadata exists
        assert "metadata" in result
        metadata = result["metadata"]
        assert "pipeline_type" in metadata

    def test_execution_time_tracking(self, basic_pipeline, sample_documents):
        """Test that execution time is tracked."""
        basic_pipeline.load_documents("", documents=sample_documents)

        result = basic_pipeline.query("test", top_k=3, generate_answer=False)

        # Check execution time is recorded
        assert "execution_time" in result or "processing_time" in result.get("metadata", {})


class TestBasicRAGPipelineConfiguration:
    """Test pipeline configuration options."""

    def test_pipeline_with_custom_vector_store(self, pipeline_dependencies):
        """Test creating pipeline with custom vector store."""
        custom_vector_store = IRISVectorStore(
            pipeline_dependencies["connection_manager"],
            pipeline_dependencies["config_manager"],
        )

        pipeline = BasicRAGPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            llm_func=pipeline_dependencies["llm_func"],
            vector_store=custom_vector_store,
        )

        assert pipeline.vector_store is custom_vector_store

    def test_pipeline_initialization_state(self, basic_pipeline):
        """Test that pipeline initializes in correct state."""
        assert hasattr(basic_pipeline, "vector_store")
        assert hasattr(basic_pipeline, "llm_func")
        assert hasattr(basic_pipeline, "config_manager")


class TestBasicRAGPipelineErrorHandling:
    """Test error handling in pipeline."""

    def test_query_with_invalid_top_k(self, basic_pipeline):
        """Test query with invalid top_k value."""
        # Should handle gracefully
        result = basic_pipeline.query("test", top_k=0, generate_answer=False)
        assert result is not None

    def test_load_documents_with_none(self, basic_pipeline):
        """Test loading None as documents."""
        # Should handle gracefully
        try:
            basic_pipeline.load_documents("", documents=None)
        except (TypeError, ValueError):
            # Expected to raise an error
            pass


class TestBasicRAGPipelineIntegration:
    """Test full integration workflows."""

    def test_complete_rag_workflow(self, basic_pipeline):
        """Test complete RAG workflow: load → query → answer."""
        # Load documents
        docs = [
            Document(id="wf1", page_content="The capital of France is Paris."),
            Document(id="wf2", page_content="Paris is known for the Eiffel Tower."),
        ]
        basic_pipeline.load_documents("", documents=docs)

        # Query
        result = basic_pipeline.query("What is the capital of France?", top_k=2, generate_answer=True)

        # Verify
        assert "answer" in result
        assert "contexts" in result
        assert len(result["contexts"]) > 0

    def test_large_batch_loading(self, basic_pipeline):
        """Test loading a larger batch of documents."""
        docs = [
            Document(id=f"batch{i}", page_content=f"Document {i} content about topic {i % 3}")
            for i in range(20)
        ]

        # Should not raise an error
        basic_pipeline.load_documents("", documents=docs)

        # Verify documents are searchable
        result = basic_pipeline.query("topic content", top_k=5, generate_answer=False)
        assert result is not None
        assert len(result["contexts"]) > 0

    def test_query_after_large_batch(self, basic_pipeline):
        """Test querying after loading large batch."""
        docs = [
            Document(id=f"qbatch{i}", page_content=f"Information about subject {i}")
            for i in range(15)
        ]
        basic_pipeline.load_documents("", documents=docs)

        result = basic_pipeline.query("subject information", top_k=5, generate_answer=False)

        assert len(result["contexts"]) <= 5
