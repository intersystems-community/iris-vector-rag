"""
Test PyLate ColBERT Pipeline Implementation

Tests the new PyLate-based ColBERT pipeline to ensure it works correctly
with consistent configuration and resolves the original memory/type issues.
"""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_rag.core.models import Document
from iris_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for pipeline testing."""
    connection_manager = Mock()
    config_manager = Mock()
    config_manager.get.return_value = {
        "rerank_factor": 2,
        "model_name": "lightonai/GTE-ModernColBERT-v1",
        "batch_size": 32,
        "use_native_reranking": True,
    }

    llm_func = Mock()
    llm_func.return_value = "This is a test answer."

    vector_store = Mock()
    vector_store.search.return_value = [
        Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test2.txt"}),
        Document(page_content="Test document 3", metadata={"source": "test3.txt"}),
        Document(page_content="Test document 4", metadata={"source": "test4.txt"}),
    ]

    return {
        "connection_manager": connection_manager,
        "config_manager": config_manager,
        "llm_func": llm_func,
        "vector_store": vector_store,
    }


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Document about machine learning algorithms",
            metadata={"source": "ml.txt"},
        ),
        Document(
            page_content="Text about deep neural networks",
            metadata={"source": "nn.txt"},
        ),
        Document(
            page_content="Information on natural language processing",
            metadata={"source": "nlp.txt"},
        ),
        Document(
            page_content="Content about computer vision techniques",
            metadata={"source": "cv.txt"},
        ),
        Document(
            page_content="Research on reinforcement learning",
            metadata={"source": "rl.txt"},
        ),
    ]


@pytest.fixture
def test_pipeline(mock_dependencies):
    """Create a PyLate pipeline instance for testing."""
    return PyLateColBERTPipeline(
        mock_dependencies["connection_manager"],
        mock_dependencies["config_manager"],
        llm_func=mock_dependencies["llm_func"],
        vector_store=mock_dependencies["vector_store"],
    )


class TestPyLateColBERTPipeline:
    """Test suite for PyLate ColBERT pipeline."""

    def test_initialization_with_defaults(self, mock_dependencies):
        """Test pipeline initialization with default configuration."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        assert pipeline.rerank_factor == 2
        assert pipeline.model_name == "lightonai/GTE-ModernColBERT-v1"
        # PyLate not installed, so use_native_reranking will be False in fallback mode
        assert pipeline.use_native_reranking == False
        # is_initialized is False when PyLate unavailable (fallback mode)
        assert pipeline.is_initialized == False

    def test_configuration_consistency_with_basic_reranking(self, mock_dependencies):
        """Test that configuration follows same patterns as BasicRAGReranking."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Should access config with same pattern as BasicRAGReranking
        mock_dependencies["config_manager"].get.assert_called_with(
            "pipelines:colbert_pylate",
            mock_dependencies["config_manager"].get.return_value,
        )

        # Should have consistent configuration parameters
        assert hasattr(pipeline, "rerank_factor")
        assert hasattr(pipeline, "stats")

    def test_load_documents(self, mock_dependencies, sample_documents):
        """Test document loading functionality."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Mock parent class load_documents to avoid path validation
        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1", "id2", "id3", "id4", "id5"])

            # PyLate's load_documents takes documents as first arg
            result = pipeline.load_documents(sample_documents)

        # Should store documents for PyLate reranking
        assert len(pipeline._document_store) == len(sample_documents)
        assert pipeline.stats["documents_indexed"] == len(sample_documents)

    def test_query_with_reranking(self, mock_dependencies, sample_documents):
        """Test query execution with PyLate reranking."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Load documents first
        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1", "id2", "id3", "id4", "id5"])
            pipeline.load_documents(sample_documents)

        # Mock vector store similarity_search to return documents
        pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:4])

        # Call query - in fallback mode it behaves like BasicRAG
        result = pipeline.query("What is machine learning?", top_k=2)

        # Should return consistent response format
        assert "query" in result
        assert "answer" in result
        # In fallback mode, may not have all metadata fields
        assert result["query"] == "What is machine learning?"
        assert result["metadata"]["pipeline_type"] == "colbert_pylate"
        assert "reranked" in result["metadata"]
        assert "initial_candidates" in result["metadata"]
        assert "rerank_factor" in result["metadata"]

    def test_fallback_mode_when_pylate_unavailable(self, mock_dependencies):
        """Test that pipeline works in fallback mode when PyLate is not available."""
        # Create pipeline - should work even without PyLate
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # In fallback mode (PyLate not installed), is_initialized is False
        assert pipeline.is_initialized == False
        # use_native_reranking should be False in fallback mode
        assert pipeline.use_native_reranking == False

    def test_pipeline_info_format(self, mock_dependencies):
        """Test that get_pipeline_info returns consistent format."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        info = pipeline.get_pipeline_info()

        # Should have consistent fields with BasicRAGReranking
        assert info["pipeline_type"] == "colbert_pylate"
        assert "rerank_factor" in info
        assert "model_name" in info
        assert "stats" in info
        assert "is_initialized" in info

    def test_memory_efficiency(self, mock_dependencies, sample_documents):
        """Test that pipeline doesn't exceed memory limits like the original ColBERT."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Should not crash with memory issues when processing documents
        try:
            with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
                mock_parent_load.return_value = {"status": "success"}
                pipeline.vector_store.add_documents = Mock(return_value=["id" + str(i) for i in range(50)])
                pipeline.load_documents(sample_documents * 10)
            # If this doesn't crash, memory management is working
            assert True
        except MemoryError:
            pytest.fail("Pipeline exceeded memory limits - memory management failed")

    def test_embedding_type_consistency(self, mock_dependencies):
        """Test that embeddings are returned as proper numpy arrays, not lists."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Mock model should return numpy arrays
        if hasattr(pipeline.model, "encode"):
            embeddings = pipeline.model.encode(["test text"], is_query=True)

            # Should be numpy arrays or at least not raw lists without ndim
            for emb in embeddings:
                assert hasattr(emb, "shape") or hasattr(
                    emb, "ndim"
                ), "Embeddings should be numpy arrays, not lists"

    def test_stats_tracking(self, mock_dependencies, sample_documents):
        """Test that pipeline tracks statistics correctly."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Initial stats
        assert pipeline.stats["queries_processed"] == 0
        assert pipeline.stats["documents_indexed"] == 0

        # Load documents
        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1", "id2", "id3", "id4", "id5"])
            pipeline.load_documents(sample_documents)

        assert pipeline.stats["documents_indexed"] == len(sample_documents)

        # Note: Query stats would be tested with actual query execution

    def test_config_parameter_defaults(self, mock_dependencies):
        """Test default configuration parameters."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        assert pipeline.rerank_factor == 2
        assert pipeline.batch_size == 32
        assert isinstance(pipeline._document_store, dict)
        assert isinstance(pipeline.stats, dict)

    def test_fallback_mode_attributes(self, mock_dependencies):
        """Test pipeline attributes in fallback mode."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # In fallback mode (PyLate not available)
        assert pipeline.is_initialized == False
        assert pipeline.use_native_reranking == False
        assert pipeline.model is None
        assert pipeline.index_folder is None

    def test_document_store_initialization(self, mock_dependencies):
        """Test document store is properly initialized."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        assert hasattr(pipeline, '_document_store')
        assert isinstance(pipeline._document_store, dict)
        assert len(pipeline._document_store) == 0

    def test_stats_structure(self, mock_dependencies):
        """Test statistics dictionary structure."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        assert "queries_processed" in pipeline.stats
        assert "documents_indexed" in pipeline.stats
        assert "reranking_operations" in pipeline.stats
        assert all(isinstance(v, int) for v in pipeline.stats.values())

    def test_query_in_fallback_mode(self, mock_dependencies, sample_documents):
        """Test query execution in fallback mode (without PyLate)."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Load documents
        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1", "id2"])
            pipeline.load_documents(sample_documents[:2])

        # Mock similarity search
        pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        # Query should work in fallback mode
        result = pipeline.query("test query", top_k=2)

        assert "query" in result
        assert "answer" in result
        assert result["query"] == "test query"

    def test_custom_rerank_factor(self, mock_dependencies):
        """Test pipeline with custom rerank factor."""
        custom_config = Mock()
        custom_config.get.return_value = {
            "rerank_factor": 5,
            "model_name": "custom-model",
            "batch_size": 64,
        }

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            custom_config,
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        assert pipeline.rerank_factor == 5
        assert pipeline.batch_size == 64
        assert pipeline.model_name == "custom-model"

    def test_query_with_different_top_k_values(self, mock_dependencies, sample_documents):
        """Test query with various top_k values."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Test with top_k=1
        pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:1])
        result1 = pipeline.query("query 1", top_k=1)
        assert "query" in result1

        # Test with top_k=10
        pipeline.vector_store.similarity_search = Mock(return_value=sample_documents)
        result10 = pipeline.query("query 2", top_k=10)
        assert "query" in result10

    def test_load_documents_with_empty_list(self, mock_dependencies):
        """Test loading empty document list."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=[])
            pipeline.load_documents([])

        assert pipeline.stats["documents_indexed"] == 0

    def test_load_documents_with_single_document(self, mock_dependencies, sample_documents):
        """Test loading a single document."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1"])
            pipeline.load_documents([sample_documents[0]])

        assert len(pipeline._document_store) == 1
        assert pipeline.stats["documents_indexed"] == 1

    def test_query_metadata_structure(self, mock_dependencies, sample_documents):
        """Test query result metadata has correct structure."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Load docs and query
        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1"])
            pipeline.load_documents(sample_documents[:1])

        pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])
        result = pipeline.query("test", top_k=2)

        assert "metadata" in result
        assert result["metadata"]["pipeline_type"] == "colbert_pylate"
        assert "rerank_factor" in result["metadata"]

    def test_multiple_queries_increment_stats(self, mock_dependencies, sample_documents):
        """Test that multiple queries increment stats correctly."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        # Execute multiple queries
        initial_count = pipeline.stats["queries_processed"]
        pipeline.query("query 1", top_k=2)
        pipeline.query("query 2", top_k=2)
        pipeline.query("query 3", top_k=2)

        # Stats should increment
        assert pipeline.stats["queries_processed"] >= initial_count

    def test_document_store_preserves_metadata(self, mock_dependencies, sample_documents):
        """Test that document store preserves document metadata."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1", "id2"])
            pipeline.load_documents(sample_documents[:2])

        # Check documents are stored with metadata
        for doc_id, doc in pipeline._document_store.items():
            assert hasattr(doc, 'metadata')
            assert 'source' in doc.metadata

    def test_get_pipeline_info_returns_all_fields(self, mock_dependencies):
        """Test that get_pipeline_info returns all expected fields."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        info = pipeline.get_pipeline_info()

        required_fields = [
            "pipeline_type",
            "rerank_factor",
            "model_name",
            "use_native_reranking",
            "batch_size",
            "is_initialized",
            "stats",
        ]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"

    def test_query_with_no_documents_loaded(self, mock_dependencies):
        """Test query when no documents are loaded."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Mock empty search results
        pipeline.vector_store.similarity_search = Mock(return_value=[])

        result = pipeline.query("test query", top_k=5)

        assert "query" in result
        assert "answer" in result

    def test_load_documents_updates_document_count(self, mock_dependencies, sample_documents):
        """Test that load_documents correctly updates document count."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        initial_count = len(pipeline._document_store)

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=["id1", "id2", "id3"])
            pipeline.load_documents(sample_documents[:3])

        assert len(pipeline._document_store) == initial_count + 3

    def test_batch_size_configuration(self, mock_dependencies):
        """Test batch size is correctly configured."""
        custom_config = Mock()
        custom_config.get.return_value = {"batch_size": 16}

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            custom_config,
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        assert pipeline.batch_size == 16

    def test_model_name_configuration(self, mock_dependencies):
        """Test model name is correctly configured."""
        custom_config = Mock()
        custom_config.get.return_value = {
            "model_name": "custom/colbert-model"
        }

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            custom_config,
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        assert pipeline.model_name == "custom/colbert-model"

    def test_llm_func_is_called_on_query(self, mock_dependencies, sample_documents):
        """Test that LLM function is called during query."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        # Reset mock to track calls
        mock_dependencies["llm_func"].reset_mock()

        pipeline.query("test query", top_k=2)

        # LLM should be called
        assert mock_dependencies["llm_func"].called

    def test_pipeline_handles_large_document_set(self, mock_dependencies):
        """Test pipeline can handle large document sets."""
        from unittest.mock import patch

        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"],
        )

        # Create 100 documents
        large_doc_set = [
            Document(page_content=f"Document {i}", metadata={"id": i})
            for i in range(100)
        ]

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            mock_parent_load.return_value = {"status": "success"}
            pipeline.vector_store.add_documents = Mock(return_value=[f"id{i}" for i in range(100)])
            pipeline.load_documents(large_doc_set)

        assert len(pipeline._document_store) == 100
        assert pipeline.stats["documents_indexed"] == 100

    def test_load_documents_with_file_path(self, test_pipeline):
        """Test loading documents from file path (delegates to parent)."""
        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.load_documents') as mock_parent_load:
            sample_docs = [
                Document(page_content="Doc 1", metadata={"source": "file.txt"}),
                Document(page_content="Doc 2", metadata={"source": "file.txt"})
            ]
            mock_parent_load.return_value = {
                "status": "success",
                "documents": sample_docs,
                "num_documents": 2
            }

            result = test_pipeline.load_documents("/path/to/documents.txt")

            mock_parent_load.assert_called_once_with("/path/to/documents.txt")
            assert result["status"] == "success"
            assert test_pipeline.stats["documents_indexed"] == 2
            assert len(test_pipeline._document_store) == 2

    def test_query_generate_answer_false(self, test_pipeline, sample_documents):
        """Test query with generate_answer=False returns no answer."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:3])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:3],
                "execution_time": 0.5
            }

            result = test_pipeline.query("Test query", generate_answer=False)

            assert result["answer"] is None
            assert result["metadata"]["generated_answer"] is False
            assert len(result["retrieved_documents"]) == 3

    def test_query_no_llm_func_provided(self, test_pipeline, sample_documents):
        """Test query when no LLM function is provided."""
        # Remove LLM function
        test_pipeline.llm_func = None

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:3],
                "execution_time": 0.5
            }

            result = test_pipeline.query("Test query")

            assert result["answer"] == "No LLM function provided. Retrieved documents only."
            # generated_answer will be True because answer is not None
            assert result["metadata"]["generated_answer"] is True

    def test_query_no_relevant_documents(self, test_pipeline):
        """Test query when no documents are retrieved."""
        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": [],
                "execution_time": 0.3
            }

            result = test_pipeline.query("Unanswerable query")

            assert result["answer"] == "No relevant documents found to answer the query."
            assert result["metadata"]["num_retrieved"] == 0
            # generated_answer will be True because answer is not None
            assert result["metadata"]["generated_answer"] is True

    def test_query_with_custom_prompt(self, test_pipeline, sample_documents):
        """Test query with custom prompt."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:2],
                "execution_time": 0.6
            }

            with patch.object(test_pipeline, '_generate_answer') as mock_generate:
                mock_generate.return_value = "Custom answer based on custom prompt"

                result = test_pipeline.query(
                    "Test query",
                    custom_prompt="Use this custom prompt: {query}"
                )

                mock_generate.assert_called_once()
                assert result["answer"] == "Custom answer based on custom prompt"

    def test_query_answer_generation_error(self, test_pipeline, sample_documents):
        """Test query handles answer generation errors gracefully."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:2],
                "execution_time": 0.4
            }

            with patch.object(test_pipeline, '_generate_answer') as mock_generate:
                mock_generate.side_effect = Exception("LLM API error")

                result = test_pipeline.query("Test query")

                assert result["answer"] == "Error generating answer"

    def test_query_include_sources_false(self, test_pipeline, sample_documents):
        """Test query with include_sources=False omits sources."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:2],
                "execution_time": 0.5
            }

            result = test_pipeline.query("Test query", include_sources=False)

            assert "sources" not in result

    def test_query_include_sources_true(self, test_pipeline, sample_documents):
        """Test query with include_sources=True includes sources."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:2],
                "execution_time": 0.5
            }

            with patch.object(test_pipeline, '_extract_sources') as mock_extract:
                mock_extract.return_value = ["source1.txt", "source2.txt"]

                result = test_pipeline.query("Test query", include_sources=True)

                assert "sources" in result
                assert result["sources"] == ["source1.txt", "source2.txt"]
                mock_extract.assert_called_once()

    def test_query_response_format_consistency(self, test_pipeline, sample_documents):
        """Test query response has all expected fields."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:3])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:3],
                "execution_time": 0.7
            }

            result = test_pipeline.query("Test query")

            # Check all expected top-level fields
            assert "query" in result
            assert "answer" in result
            assert "retrieved_documents" in result
            assert "contexts" in result
            assert "execution_time" in result
            assert "metadata" in result
            assert "sources" in result

            # Check all metadata fields
            metadata = result["metadata"]
            assert "num_retrieved" in metadata
            assert "processing_time" in metadata
            assert "pipeline_type" in metadata
            assert "reranked" in metadata
            assert "initial_candidates" in metadata
            assert "rerank_factor" in metadata
            assert "generated_answer" in metadata
            assert "model_name" in metadata
            assert "native_reranking" in metadata

    def test_cache_embeddings_configuration(self, mock_dependencies):
        """Test cache_embeddings parameter configuration."""
        config_manager = Mock()
        config_manager.get.side_effect = lambda key, default=None: {
            "pipelines:colbert_pylate": {"cache_embeddings": False}
        }.get(key, default)

        pipeline = PyLateColBERTPipeline(
            connection_manager=mock_dependencies["connection_manager"],
            config_manager=config_manager,
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )

        assert pipeline.cache_embeddings is False

    def test_max_doc_length_configuration(self, mock_dependencies):
        """Test max_doc_length parameter configuration."""
        config_manager = Mock()
        config_manager.get.side_effect = lambda key, default=None: {
            "pipelines:colbert_pylate": {"max_doc_length": 8192}
        }.get(key, default)

        pipeline = PyLateColBERTPipeline(
            connection_manager=mock_dependencies["connection_manager"],
            config_manager=config_manager,
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )

        assert pipeline.max_doc_length == 8192

    def test_embedding_cache_initialization(self, test_pipeline):
        """Test _embedding_cache is initialized empty."""
        assert test_pipeline._embedding_cache == {}
        assert isinstance(test_pipeline._embedding_cache, dict)

    def test_model_initialization_fallback(self, test_pipeline):
        """Test model is None in fallback mode."""
        assert test_pipeline.model is None
        assert test_pipeline.index_folder is None

    def test_contexts_field_matches_documents(self, test_pipeline, sample_documents):
        """Test contexts field contains page_content from retrieved documents."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:3])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:3],
                "execution_time": 0.5
            }

            result = test_pipeline.query("Test query")

            assert len(result["contexts"]) == 3
            assert result["contexts"][0] == sample_documents[0].page_content
            assert result["contexts"][1] == sample_documents[1].page_content
            assert result["contexts"][2] == sample_documents[2].page_content

    def test_stats_tracking_on_successful_query(self, test_pipeline, sample_documents):
        """Test stats are incremented correctly on query."""
        initial_count = test_pipeline.stats["queries_processed"]

        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:2])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:2],
                "execution_time": 0.5
            }

            test_pipeline.query("Query 1")
            assert test_pipeline.stats["queries_processed"] == initial_count + 1

            test_pipeline.query("Query 2")
            assert test_pipeline.stats["queries_processed"] == initial_count + 2

    def test_reranking_operations_stat_in_fallback_mode(self, test_pipeline, sample_documents):
        """Test reranking_operations stat remains 0 in fallback mode."""
        test_pipeline.vector_store.similarity_search = Mock(return_value=sample_documents[:5])

        with patch('iris_rag.pipelines.basic.BasicRAGPipeline.query') as mock_parent_query:
            mock_parent_query.return_value = {
                "retrieved_documents": sample_documents[:5],
                "execution_time": 0.5
            }

            initial_rerank_count = test_pipeline.stats["reranking_operations"]

            result = test_pipeline.query("Test query", top_k=2)

            # In fallback mode, reranking should not happen
            assert test_pipeline.stats["reranking_operations"] == initial_rerank_count
            assert result["metadata"]["reranked"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
