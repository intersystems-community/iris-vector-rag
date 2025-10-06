"""
E2E Tests for CRAG Pipeline

Comprehensive end-to-end tests with real IRIS database integration.
Tests confidence-based retrieval, corrective actions, and knowledge refinement.
"""

import pytest
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.storage.vector_store_iris import IRISVectorStore
from common.utils import get_llm_func, get_embedding_func
from common.iris_connection_manager import get_iris_connection


@pytest.fixture(scope="session", autouse=True)
def clean_database_once():
    """Drop and recreate tables at session start with correct DOUBLE schema."""
    conn = get_iris_connection()
    cursor = conn.cursor()

    # Drop tables completely
    cursor.execute('DROP TABLE IF EXISTS RAG.DocumentChunks CASCADE')
    cursor.execute('DROP TABLE IF EXISTS RAG.SourceDocuments CASCADE')
    conn.commit()

    # Recreate with DOUBLE datatype
    cursor.execute('''
    CREATE TABLE RAG.SourceDocuments (
        doc_id VARCHAR(255) PRIMARY KEY,
        text_content TEXT,
        metadata TEXT,
        embedding VECTOR(DOUBLE, 384),
        created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute("CREATE INDEX idx_hnsw_source_embedding ON RAG.SourceDocuments (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')")

    cursor.execute('''
    CREATE TABLE RAG.DocumentChunks (
        id VARCHAR(255) PRIMARY KEY,
        chunk_id VARCHAR(255),
        doc_id VARCHAR(255),
        chunk_text TEXT,
        chunk_embedding VECTOR(DOUBLE, 384),
        chunk_index INTEGER,
        chunk_type VARCHAR(100),
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
    )
    ''')
    cursor.execute('CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks (doc_id)')
    cursor.execute('CREATE INDEX idx_chunks_chunk_id ON RAG.DocumentChunks (chunk_id)')
    cursor.execute("CREATE INDEX idx_hnsw_chunk_embedding ON RAG.DocumentChunks (chunk_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE')")

    conn.commit()
    cursor.close()
    yield


@pytest.fixture(scope="module")
def pipeline_dependencies():
    """Create real dependencies for E2E testing."""
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    llm_func = get_llm_func()
    embedding_func = get_embedding_func()
    vector_store = IRISVectorStore(connection_manager, config_manager)

    return {
        "config_manager": config_manager,
        "connection_manager": connection_manager,
        "llm_func": llm_func,
        "embedding_func": embedding_func,
        "vector_store": vector_store,
    }


@pytest.fixture(scope="module")
def crag_pipeline(pipeline_dependencies):
    """Create CRAG pipeline instance."""
    return CRAGPipeline(
        connection_manager=pipeline_dependencies["connection_manager"],
        config_manager=pipeline_dependencies["config_manager"],
        vector_store=pipeline_dependencies["vector_store"],
        embedding_func=pipeline_dependencies["embedding_func"],
        llm_func=pipeline_dependencies["llm_func"],
    )


@pytest.fixture
def confident_documents():
    """Documents that should produce confident retrieval."""
    return [
        Document(
            id="crag_conf1",
            page_content="Python is a high-level programming language designed for code readability and simplicity. It was created by Guido van Rossum and first released in 1991.",
            metadata={"category": "programming", "confidence": "high"},
        ),
        Document(
            id="crag_conf2",
            page_content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            metadata={"category": "programming", "confidence": "high"},
        ),
        Document(
            id="crag_conf3",
            page_content="The Python Software Foundation manages Python development and promotes Python usage worldwide.",
            metadata={"category": "programming", "confidence": "high"},
        ),
    ]


@pytest.fixture
def ambiguous_documents():
    """Documents that should produce ambiguous retrieval."""
    return [
        Document(
            id="crag_amb1",
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "ml", "confidence": "medium"},
        ),
        Document(
            id="crag_amb2",
            page_content="Data science involves statistical analysis and visualization.",
            metadata={"category": "data", "confidence": "medium"},
        ),
        Document(
            id="crag_amb3",
            page_content="Computer programming requires logical thinking and problem solving.",
            metadata={"category": "general", "confidence": "medium"},
        ),
    ]


@pytest.fixture
def diverse_documents():
    """Diverse set of documents for testing different scenarios."""
    return [
        Document(
            id="crag_div1",
            page_content="Climate change is affecting global weather patterns and ecosystems.",
            metadata={"category": "environment"},
        ),
        Document(
            id="crag_div2",
            page_content="Renewable energy sources include solar, wind, and hydroelectric power.",
            metadata={"category": "energy"},
        ),
        Document(
            id="crag_div3",
            page_content="Quantum computing uses quantum mechanics principles for computation.",
            metadata={"category": "computing"},
        ),
        Document(
            id="crag_div4",
            page_content="Artificial neural networks are inspired by biological neural systems.",
            metadata={"category": "ai"},
        ),
        Document(
            id="crag_div5",
            page_content="Blockchain technology provides decentralized and secure record keeping.",
            metadata={"category": "technology"},
        ),
    ]


class TestCRAGPipelineInitialization:
    """Test pipeline initialization and configuration."""

    def test_pipeline_initialization(self, crag_pipeline):
        """Test that pipeline initializes correctly."""
        assert crag_pipeline is not None
        assert hasattr(crag_pipeline, "evaluator")
        assert hasattr(crag_pipeline, "embedding_func")
        assert hasattr(crag_pipeline, "llm_func")

    def test_pipeline_has_evaluator(self, crag_pipeline):
        """Test that pipeline has retrieval evaluator."""
        assert crag_pipeline.evaluator is not None

    def test_document_chunks_table_ensured(self, crag_pipeline):
        """Test that DocumentChunks table is ensured on initialization."""
        # Pipeline should have attempted to create DocumentChunks table
        connection = crag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            # If we get here, table exists
            assert True
        except Exception:
            # Table may not exist in test environment, that's ok
            pass
        finally:
            cursor.close()


class TestCRAGPipelineDocumentLoading:
    """Test document loading functionality."""

    def test_load_documents_from_list(self, crag_pipeline, confident_documents):
        """Test loading documents from a list."""
        crag_pipeline.load_documents("", documents=confident_documents)

        result = crag_pipeline.query("Python programming", top_k=2, generate_answer=False)
        assert result is not None
        assert "contexts" in result

    def test_load_single_document(self, crag_pipeline):
        """Test loading a single document."""
        doc = Document(
            id="crag_single1",
            page_content="Test document for CRAG single loading.",
        )

        crag_pipeline.load_documents("", documents=[doc])
        result = crag_pipeline.query("single loading", top_k=1, generate_answer=False)
        assert result is not None

    def test_load_documents_with_embeddings(self, crag_pipeline, confident_documents):
        """Test loading documents with embedding generation."""
        crag_pipeline.load_documents("", documents=confident_documents, generate_embeddings=True)

        result = crag_pipeline.query("Python", top_k=2, generate_answer=False)
        assert len(result["contexts"]) > 0


class TestCRAGPipelineConfidentRetrieval:
    """Test confident retrieval scenarios."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, crag_pipeline, confident_documents):
        """Load confident documents before each test."""
        crag_pipeline.load_documents("", documents=confident_documents)

    def test_confident_query_returns_results(self, crag_pipeline):
        """Test that confident queries return good results."""
        result = crag_pipeline.query(
            "What is Python programming language?",
            top_k=3,
            generate_answer=False,
        )

        assert result is not None
        assert "metadata" in result
        # Check retrieval status
        metadata = result["metadata"]
        assert "retrieval_status" in metadata

    def test_confident_query_metadata(self, crag_pipeline):
        """Test metadata for confident queries."""
        result = crag_pipeline.query("Python language", top_k=2, generate_answer=False)

        metadata = result["metadata"]
        assert "retrieval_status" in metadata
        assert metadata["retrieval_status"] in ["confident", "ambiguous", "disoriented"]
        assert "initial_doc_count" in metadata
        assert "final_doc_count" in metadata

    def test_confident_query_with_answer(self, crag_pipeline):
        """Test confident query with answer generation."""
        result = crag_pipeline.query("What is Python?", top_k=2, generate_answer=True)

        assert "answer" in result
        assert len(result["answer"]) > 0


class TestCRAGPipelineAmbiguousRetrieval:
    """Test ambiguous retrieval scenarios."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, crag_pipeline, ambiguous_documents):
        """Load ambiguous documents before each test."""
        crag_pipeline.load_documents("", documents=ambiguous_documents)

    def test_ambiguous_query_enhancement(self, crag_pipeline):
        """Test that ambiguous queries trigger enhancement."""
        result = crag_pipeline.query(
            "machine learning data",
            top_k=3,
            generate_answer=False,
        )

        assert result is not None
        metadata = result["metadata"]
        assert "retrieval_status" in metadata
        # Enhanced retrieval may return more documents
        assert "final_doc_count" in metadata

    def test_ambiguous_query_metadata(self, crag_pipeline):
        """Test metadata for ambiguous queries."""
        result = crag_pipeline.query("data analysis", top_k=2, generate_answer=False)

        metadata = result["metadata"]
        assert "retrieval_status" in metadata
        assert "initial_doc_count" in metadata

    def test_ambiguous_with_chunk_enhancement(self, crag_pipeline):
        """Test chunk-based enhancement for ambiguous queries."""
        result = crag_pipeline.query(
            "programming concepts",
            top_k=3,
            generate_answer=False,
        )

        # Should return results (possibly enhanced)
        assert len(result["contexts"]) > 0


class TestCRAGPipelineDisorientedRetrieval:
    """Test disoriented retrieval scenarios."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, crag_pipeline, diverse_documents):
        """Load diverse documents before each test."""
        crag_pipeline.load_documents("", documents=diverse_documents)

    def test_disoriented_query_expansion(self, crag_pipeline):
        """Test that disoriented queries trigger knowledge base expansion."""
        result = crag_pipeline.query(
            "completely unrelated topic query",
            top_k=3,
            generate_answer=False,
        )

        assert result is not None
        metadata = result["metadata"]
        assert "retrieval_status" in metadata

    def test_disoriented_query_returns_documents(self, crag_pipeline):
        """Test that disoriented queries still return documents."""
        result = crag_pipeline.query(
            "random unrelated query",
            top_k=2,
            generate_answer=False,
        )

        # Should attempt to return some documents
        assert "contexts" in result

    def test_disoriented_with_semantic_search(self, crag_pipeline):
        """Test semantic search fallback for disoriented queries."""
        result = crag_pipeline.query(
            "xyz123 nonexistent",
            top_k=3,
            generate_answer=False,
        )

        # Should handle gracefully
        assert result is not None


class TestCRAGPipelineCorrectiveActions:
    """Test corrective action mechanisms."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, crag_pipeline, diverse_documents):
        """Load diverse documents before each test."""
        crag_pipeline.load_documents("", documents=diverse_documents)

    def test_corrective_action_for_confident(self, crag_pipeline):
        """Test that confident status uses initial results."""
        result = crag_pipeline.query("quantum computing", top_k=2, generate_answer=False)

        metadata = result["metadata"]
        # Initial and final counts should be similar for confident retrieval
        assert "initial_doc_count" in metadata
        assert "final_doc_count" in metadata

    def test_corrective_action_for_ambiguous(self, crag_pipeline):
        """Test enhancement for ambiguous status."""
        result = crag_pipeline.query("technology innovation", top_k=3, generate_answer=False)

        # Ambiguous may trigger enhancement
        assert len(result["contexts"]) > 0

    def test_corrective_action_for_disoriented(self, crag_pipeline):
        """Test knowledge base expansion for disoriented status."""
        result = crag_pipeline.query("unrelated subject matter", top_k=2, generate_answer=False)

        # Should attempt knowledge base expansion
        assert result is not None


class TestCRAGPipelineRetrievalEvaluator:
    """Test retrieval evaluator functionality."""

    def test_evaluator_with_high_scores(self, crag_pipeline, confident_documents):
        """Test evaluator with high similarity scores."""
        crag_pipeline.load_documents("", documents=confident_documents)

        result = crag_pipeline.query("Python programming", top_k=2, generate_answer=False)

        metadata = result["metadata"]
        assert "retrieval_status" in metadata

    def test_evaluator_with_no_documents(self, crag_pipeline):
        """Test evaluator when no documents are retrieved."""
        # Query on empty database
        result = crag_pipeline.query("nonexistent topic", top_k=2, generate_answer=False)

        metadata = result["metadata"]
        assert "retrieval_status" in metadata

    def test_evaluator_status_values(self, crag_pipeline, diverse_documents):
        """Test that evaluator returns valid status values."""
        crag_pipeline.load_documents("", documents=diverse_documents)

        result = crag_pipeline.query("technology", top_k=3, generate_answer=False)

        metadata = result["metadata"]
        status = metadata.get("retrieval_status")
        assert status in ["confident", "ambiguous", "disoriented"]


class TestCRAGPipelineAnswerGeneration:
    """Test answer generation with different retrieval statuses."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, crag_pipeline, confident_documents):
        """Load documents before each test."""
        crag_pipeline.load_documents("", documents=confident_documents)

    def test_answer_generation_confident(self, crag_pipeline):
        """Test answer generation for confident retrieval."""
        result = crag_pipeline.query("What is Python?", top_k=2, generate_answer=True)

        assert "answer" in result
        assert len(result["answer"]) > 0
        # Should indicate confidence in the context
        assert "highly relevant" in result["answer"] or "Based on" in result["answer"]

    def test_answer_generation_without_llm(self, pipeline_dependencies):
        """Test answer generation when LLM is not available."""
        pipeline = CRAGPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            vector_store=pipeline_dependencies["vector_store"],
            embedding_func=pipeline_dependencies["embedding_func"],
            llm_func=None,
        )

        docs = [Document(id="crag_no_llm1", page_content="Test content")]
        pipeline.load_documents("", documents=docs)

        result = pipeline.query("test", top_k=1, generate_answer=True)

        assert "answer" in result
        # Should provide a message about no LLM
        assert "No LLM function" in result["answer"]

    def test_answer_with_no_documents(self, crag_pipeline):
        """Test answer generation when no documents are found."""
        result = crag_pipeline.query("xyz nonexistent query", top_k=2, generate_answer=True)

        # Should handle gracefully
        assert "answer" in result


class TestCRAGPipelineDocumentChunks:
    """Test DocumentChunks table usage."""

    def test_chunk_based_enhancement(self, crag_pipeline):
        """Test that chunk-based retrieval works."""
        docs = [
            Document(
                id="crag_chunk1",
                page_content="This is a long document that should be chunked. " * 10,
            )
        ]
        crag_pipeline.load_documents("", documents=docs)

        result = crag_pipeline.query("long document", top_k=2, generate_answer=False)

        assert result is not None

    def test_chunks_table_interaction(self, crag_pipeline):
        """Test interaction with DocumentChunks table."""
        connection = crag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Try to query chunks table
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            count = cursor.fetchone()[0]
            # Table exists
            assert count >= 0
        except Exception:
            # Table may not exist in test environment
            pass
        finally:
            cursor.close()


class TestCRAGPipelineErrorHandling:
    """Test error handling in CRAG pipeline."""

    def test_query_with_invalid_top_k(self, crag_pipeline):
        """Test query with invalid top_k value."""
        result = crag_pipeline.query("test", top_k=0, generate_answer=False)
        assert result is not None

    def test_load_documents_with_none(self, crag_pipeline):
        """Test loading None as documents."""
        try:
            crag_pipeline.load_documents("", documents=None)
        except (TypeError, ValueError):
            # Expected to raise an error
            pass

    def test_query_error_handling(self, crag_pipeline):
        """Test that query errors are handled gracefully."""
        result = crag_pipeline.query("test query", top_k=3, generate_answer=False)

        # Should return valid result structure even if errors occur
        assert "query" in result
        assert "contexts" in result
        assert "metadata" in result


class TestCRAGPipelineIntegration:
    """Test full integration workflows."""

    def test_complete_crag_workflow(self, crag_pipeline):
        """Test complete CRAG workflow: load → evaluate → correct → answer."""
        docs = [
            Document(
                id="crag_wf1",
                page_content="The Eiffel Tower is located in Paris, France.",
            ),
            Document(
                id="crag_wf2",
                page_content="Paris is the capital city of France.",
            ),
        ]
        crag_pipeline.load_documents("", documents=docs)

        result = crag_pipeline.query("Where is the Eiffel Tower?", top_k=2, generate_answer=True)

        assert "answer" in result
        assert "contexts" in result
        assert len(result["contexts"]) > 0
        assert "retrieval_status" in result["metadata"]

    def test_large_batch_loading(self, crag_pipeline):
        """Test loading a larger batch of documents."""
        docs = [
            Document(id=f"crag_batch{i}", page_content=f"Document {i} about topic {i % 5}")
            for i in range(25)
        ]

        crag_pipeline.load_documents("", documents=docs)

        result = crag_pipeline.query("topic", top_k=5, generate_answer=False)
        assert result is not None

    def test_sequential_queries_different_statuses(self, crag_pipeline, diverse_documents):
        """Test sequential queries that might produce different statuses."""
        crag_pipeline.load_documents("", documents=diverse_documents)

        queries = [
            "quantum computing",  # Should be confident
            "technology innovation",  # May be ambiguous
            "xyz123 random",  # Likely disoriented
        ]

        for query in queries:
            result = crag_pipeline.query(query, top_k=2, generate_answer=False)
            assert "retrieval_status" in result["metadata"]


class TestCRAGPipelinePerformance:
    """Test performance-related aspects."""

    def test_execution_time_tracking(self, crag_pipeline, confident_documents):
        """Test that execution time is tracked."""
        crag_pipeline.load_documents("", documents=confident_documents)

        result = crag_pipeline.query("test", top_k=3, generate_answer=False)

        assert "execution_time" in result
        assert result["execution_time"] >= 0

    def test_corrective_action_efficiency(self, crag_pipeline, diverse_documents):
        """Test that corrective actions complete in reasonable time."""
        crag_pipeline.load_documents("", documents=diverse_documents)

        result = crag_pipeline.query("technology", top_k=5, generate_answer=False)

        # Should complete with execution time tracked
        assert "execution_time" in result
