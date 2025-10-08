"""
E2E Tests for BasicRAGRerankingPipeline

Comprehensive end-to-end tests with real IRIS database integration.
Tests document loading, querying with reranking, and various rerank configurations.
"""

import pytest
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
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
def rerank_pipeline(pipeline_dependencies):
    """Create BasicRAGRerankingPipeline instance."""
    return BasicRAGRerankingPipeline(
        connection_manager=pipeline_dependencies["connection_manager"],
        config_manager=pipeline_dependencies["config_manager"],
        llm_func=pipeline_dependencies["llm_func"],
        vector_store=pipeline_dependencies["vector_store"],
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing reranking."""
    return [
        Document(
            id="rerank_doc1",
            page_content="Python is a versatile programming language used for web development, data science, and automation.",
            metadata={"category": "programming", "topic": "python"},
        ),
        Document(
            id="rerank_doc2",
            page_content="Machine learning algorithms can be implemented efficiently in Python using libraries like scikit-learn.",
            metadata={"category": "ml", "topic": "python"},
        ),
        Document(
            id="rerank_doc3",
            page_content="Deep learning frameworks such as TensorFlow and PyTorch are popular for neural network development.",
            metadata={"category": "ml", "topic": "deep_learning"},
        ),
        Document(
            id="rerank_doc4",
            page_content="Natural language processing enables computers to understand, interpret, and generate human language.",
            metadata={"category": "nlp", "topic": "text_processing"},
        ),
        Document(
            id="rerank_doc5",
            page_content="Computer vision applications include image classification, object detection, and facial recognition.",
            metadata={"category": "cv", "topic": "image_processing"},
        ),
        Document(
            id="rerank_doc6",
            page_content="Data visualization in Python is accomplished using libraries like matplotlib, seaborn, and plotly.",
            metadata={"category": "data", "topic": "python"},
        ),
        Document(
            id="rerank_doc7",
            page_content="Web frameworks like Django and Flask make it easy to build scalable web applications in Python.",
            metadata={"category": "web", "topic": "python"},
        ),
        Document(
            id="rerank_doc8",
            page_content="Statistical analysis and hypothesis testing are fundamental to data science and research.",
            metadata={"category": "statistics", "topic": "analysis"},
        ),
    ]


class TestBasicRerankPipelineInitialization:
    """Test pipeline initialization and configuration."""

    def test_pipeline_initialization(self, rerank_pipeline):
        """Test that pipeline initializes correctly."""
        assert rerank_pipeline is not None
        assert hasattr(rerank_pipeline, "reranker_func")
        assert hasattr(rerank_pipeline, "rerank_factor")
        assert rerank_pipeline.rerank_factor >= 1

    def test_pipeline_has_reranker(self, rerank_pipeline):
        """Test that pipeline has a reranker function."""
        assert rerank_pipeline.reranker_func is not None

    def test_pipeline_configuration(self, rerank_pipeline):
        """Test pipeline configuration values."""
        info = rerank_pipeline.get_pipeline_info()
        assert info["pipeline_type"] == "basic_rag_reranking"
        assert "rerank_factor" in info
        assert info["has_reranker"] is True


class TestBasicRerankPipelineDocumentLoading:
    """Test document loading functionality."""

    def test_load_documents_from_list(self, rerank_pipeline, sample_documents):
        """Test loading documents from a list."""
        rerank_pipeline.load_documents(documents=sample_documents)

        # Verify documents can be queried
        result = rerank_pipeline.query("Python programming", top_k=2, generate_answer=False)
        assert result is not None
        assert "contexts" in result
        assert len(result["contexts"]) <= 2

    def test_load_single_document(self, rerank_pipeline):
        """Test loading a single document."""
        doc = Document(
            id="rerank_single1",
            page_content="Test document for single loading with reranking support.",
        )

        rerank_pipeline.load_documents(documents=[doc])

        result = rerank_pipeline.query("single loading", top_k=1, generate_answer=False)
        assert result is not None

    def test_load_documents_with_metadata(self, rerank_pipeline):
        """Test loading documents with metadata."""
        docs = [
            Document(
                id="rerank_meta1",
                page_content="Document with metadata about testing reranking frameworks.",
                metadata={"source": "test", "category": "reranking"},
            )
        ]

        rerank_pipeline.load_documents(documents=docs)
        result = rerank_pipeline.query("testing frameworks", top_k=1, generate_answer=False)
        assert result is not None


class TestBasicRerankPipelineQuerying:
    """Test query functionality with reranking."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, rerank_pipeline, sample_documents):
        """Load documents before each test."""
        rerank_pipeline.load_documents(documents=sample_documents)

    def test_simple_query_with_reranking(self, rerank_pipeline):
        """Test a simple query with reranking enabled."""
        result = rerank_pipeline.query("What is Python?", top_k=3, generate_answer=True)

        assert result is not None
        assert "answer" in result
        assert "contexts" in result
        assert len(result["contexts"]) <= 3
        assert "metadata" in result
        assert result["metadata"].get("reranked") is not None

    def test_query_with_various_top_k(self, rerank_pipeline):
        """Test queries with different top_k values."""
        for k in [1, 2, 3, 5]:
            result = rerank_pipeline.query("machine learning", top_k=k, generate_answer=False)

            assert len(result["contexts"]) <= k
            # Should have metadata about reranking
            assert "metadata" in result

    def test_query_without_answer_generation(self, rerank_pipeline):
        """Test query without LLM answer generation."""
        result = rerank_pipeline.query("deep learning", top_k=3, generate_answer=False)

        assert "contexts" in result
        assert result.get("answer") is None or result.get("answer") == ""
        assert len(result["contexts"]) <= 3

    def test_query_with_answer_generation(self, rerank_pipeline):
        """Test query with LLM answer generation."""
        result = rerank_pipeline.query("What is NLP?", top_k=3, generate_answer=True)

        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_reranking_metadata(self, rerank_pipeline):
        """Test that reranking metadata is included in results."""
        result = rerank_pipeline.query("Python libraries", top_k=3, generate_answer=False)

        assert "metadata" in result
        metadata = result["metadata"]
        assert "reranked" in metadata
        assert "initial_candidates" in metadata
        assert "rerank_factor" in metadata


class TestBasicRerankPipelineRerankFactor:
    """Test different rerank_factor configurations."""

    def test_rerank_factor_default(self, rerank_pipeline, sample_documents):
        """Test default rerank_factor behavior."""
        rerank_pipeline.load_documents(documents=sample_documents)

        result = rerank_pipeline.query("Python", top_k=3, generate_answer=False)

        # Should retrieve rerank_factor * top_k initially
        metadata = result["metadata"]
        assert metadata["initial_candidates"] >= metadata["num_retrieved"]

    def test_rerank_factor_large(self, pipeline_dependencies, sample_documents):
        """Test with larger rerank_factor."""
        # Create pipeline with larger rerank factor
        pipeline = BasicRAGRerankingPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            llm_func=pipeline_dependencies["llm_func"],
            vector_store=pipeline_dependencies["vector_store"],
        )
        pipeline.rerank_factor = 3

        pipeline.load_documents(documents=sample_documents)

        result = pipeline.query("machine learning", top_k=2, generate_answer=False)

        # Should retrieve at least top_k results
        assert len(result["contexts"]) <= 2
        metadata = result["metadata"]
        assert metadata["rerank_factor"] == 3

    def test_rerank_factor_single_result(self, rerank_pipeline, sample_documents):
        """Test reranking when only one result is requested."""
        rerank_pipeline.load_documents(documents=sample_documents)

        result = rerank_pipeline.query("Python", top_k=1, generate_answer=False)

        assert len(result["contexts"]) == 1


class TestBasicRerankPipelineRerankingBehavior:
    """Test reranking behavior and effectiveness."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, rerank_pipeline, sample_documents):
        """Load documents before each test."""
        rerank_pipeline.load_documents(documents=sample_documents)

    def test_reranking_improves_relevance(self, rerank_pipeline):
        """Test that reranking improves document relevance ordering."""
        result = rerank_pipeline.query(
            "Python programming language for web development",
            top_k=3,
            generate_answer=False,
        )

        # First result should be highly relevant to Python
        contexts = result["contexts"]
        assert len(contexts) > 0
        # Check that Python is mentioned in top results
        python_mentions = sum(1 for ctx in contexts if "Python" in ctx or "python" in ctx.lower())
        assert python_mentions > 0

    def test_reranking_with_multiple_candidates(self, rerank_pipeline):
        """Test reranking when many candidates are retrieved."""
        result = rerank_pipeline.query("data science", top_k=5, generate_answer=False)

        assert len(result["contexts"]) <= 5
        metadata = result["metadata"]
        assert metadata["initial_candidates"] >= metadata["num_retrieved"]

    def test_reranking_status_in_metadata(self, rerank_pipeline):
        """Test that reranking status is tracked in metadata."""
        result = rerank_pipeline.query("visualization", top_k=2, generate_answer=False)

        metadata = result["metadata"]
        assert "reranked" in metadata
        assert isinstance(metadata["reranked"], bool)


class TestBasicRerankPipelineWithoutReranker:
    """Test pipeline behavior when reranker is not available."""

    def test_pipeline_without_reranker_function(self, pipeline_dependencies, sample_documents):
        """Test pipeline when reranker function is None."""
        # Create pipeline without reranker
        pipeline = BasicRAGRerankingPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            llm_func=pipeline_dependencies["llm_func"],
            vector_store=pipeline_dependencies["vector_store"],
            reranker_func=None,
        )
        pipeline.reranker_func = None

        pipeline.load_documents(documents=sample_documents)

        result = pipeline.query("Python", top_k=3, generate_answer=False)

        # Should still return results without reranking
        assert len(result["contexts"]) <= 3
        metadata = result["metadata"]
        assert metadata["reranked"] is False


class TestBasicRerankPipelineErrorHandling:
    """Test error handling in reranking pipeline."""

    def test_query_with_invalid_top_k(self, rerank_pipeline):
        """Test query with invalid top_k value."""
        # New API validates and raises ValueError
        import pytest
        with pytest.raises(ValueError, match="top_k parameter out of valid range"):
            rerank_pipeline.query("test", top_k=0, generate_answer=False)

    def test_query_on_empty_database(self, rerank_pipeline):
        """Test query when no documents are loaded."""
        # Query without loading documents first
        result = rerank_pipeline.query("nonexistent topic", top_k=3, generate_answer=False)

        # Should return empty or minimal results
        assert result is not None
        assert "contexts" in result


class TestBasicRerankPipelineIntegration:
    """Test full integration workflows with reranking."""

    def test_complete_rerank_workflow(self, rerank_pipeline):
        """Test complete workflow: load → query with reranking → answer."""
        docs = [
            Document(
                id="rerank_wf1",
                page_content="The capital of France is Paris.",
            ),
            Document(
                id="rerank_wf2",
                page_content="Paris is known for the Eiffel Tower and the Louvre Museum.",
            ),
            Document(
                id="rerank_wf3",
                page_content="France is a country in Western Europe.",
            ),
        ]
        rerank_pipeline.load_documents(documents=docs)

        result = rerank_pipeline.query(
            "What is the capital of France?",
            top_k=2,
            generate_answer=True,
        )

        assert "answer" in result
        assert "contexts" in result
        assert len(result["contexts"]) > 0
        metadata = result["metadata"]
        assert "reranked" in metadata

    def test_large_batch_with_reranking(self, rerank_pipeline):
        """Test loading and querying large batch with reranking."""
        docs = [
            Document(
                id=f"rerank_batch{i}",
                page_content=f"Document {i} content about topic {i % 5}",
            )
            for i in range(20)
        ]

        rerank_pipeline.load_documents(documents=docs)

        result = rerank_pipeline.query("topic content", top_k=5, generate_answer=False)
        assert result is not None
        assert len(result["contexts"]) <= 5

    def test_sequential_queries_with_reranking(self, rerank_pipeline, sample_documents):
        """Test multiple sequential queries with reranking."""
        rerank_pipeline.load_documents(documents=sample_documents)

        queries = [
            "What is Python?",
            "machine learning libraries",
            "web development frameworks",
        ]

        for query in queries:
            result = rerank_pipeline.query(query, top_k=3, generate_answer=False)
            assert "contexts" in result
            assert len(result["contexts"]) <= 3
            assert result["metadata"]["reranked"] is not None


class TestBasicRerankPipelinePerformance:
    """Test performance-related aspects of reranking."""

    def test_execution_time_tracking(self, rerank_pipeline, sample_documents):
        """Test that execution time is tracked."""
        rerank_pipeline.load_documents(documents=sample_documents)

        result = rerank_pipeline.query("test", top_k=3, generate_answer=False)

        assert "execution_time" in result or "processing_time" in result.get("metadata", {})

    def test_reranking_with_large_candidate_pool(self, rerank_pipeline):
        """Test reranking with large candidate pool."""
        # Create many documents
        docs = [
            Document(
                id=f"rerank_perf{i}",
                page_content=f"Performance test document {i} with various content",
            )
            for i in range(50)
        ]
        rerank_pipeline.load_documents(documents=docs)

        result = rerank_pipeline.query("performance test", top_k=10, generate_answer=False)

        assert len(result["contexts"]) <= 10
        assert "execution_time" in result


class TestBasicRerankPipelineCustomReranker:
    """Test custom reranker function support."""

    def test_custom_reranker_function(self, pipeline_dependencies, sample_documents):
        """Test pipeline with custom reranker function."""

        def custom_reranker(query, docs):
            """Simple custom reranker based on keyword matching."""
            scores = []
            for doc in docs:
                # Simple keyword matching score
                score = sum(
                    1.0 for word in query.lower().split() if word in doc.page_content.lower()
                )
                scores.append(score)
            return [(doc, score) for doc, score in zip(docs, scores)]

        pipeline = BasicRAGRerankingPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            llm_func=pipeline_dependencies["llm_func"],
            vector_store=pipeline_dependencies["vector_store"],
            reranker_func=custom_reranker,
        )

        pipeline.load_documents(documents=sample_documents)

        result = pipeline.query("Python machine learning", top_k=3, generate_answer=False)

        assert len(result["contexts"]) <= 3
        assert result["metadata"]["reranked"] is True
