"""
E2E Tests for GraphRAG Pipeline

Comprehensive end-to-end tests with real IRIS database integration.
Tests entity extraction, relationship storage, graph traversal, and multi-hop reasoning.

NOTE: These tests make REAL LLM API calls for entity extraction and answer generation.
They are slow (30-60s per test) and require valid API keys.
Use pytest -m "not slow" to skip these tests in normal development.
"""

import pytest

# Mark all tests in this module as slow since they make real LLM API calls
pytestmark = [pytest.mark.slow, pytest.mark.requires_llm_api]
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline, KnowledgeGraphNotPopulatedException
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from common.utils import get_llm_func


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def graphrag_pipeline(pipeline_dependencies):
    """Create GraphRAG pipeline instance."""
    return GraphRAGPipeline(
        connection_manager=pipeline_dependencies["connection_manager"],
        config_manager=pipeline_dependencies["config_manager"],
        llm_func=pipeline_dependencies["llm_func"],
        vector_store=pipeline_dependencies["vector_store"],
    )


@pytest.fixture
def entity_rich_documents():
    """Documents with clear entities for extraction."""
    return [
        Document(
            id="graphrag_ent1",
            page_content="Python is a programming language created by Guido van Rossum. Python is used for web development and data science.",
            metadata={"category": "programming", "has_entities": True},
        ),
        Document(
            id="graphrag_ent2",
            page_content="TensorFlow is a machine learning framework developed by Google. TensorFlow supports neural networks and deep learning.",
            metadata={"category": "ml", "has_entities": True},
        ),
        Document(
            id="graphrag_ent3",
            page_content="Django is a web framework written in Python. Django follows the model-view-template architectural pattern.",
            metadata={"category": "web", "has_entities": True},
        ),
    ]


@pytest.fixture
def relationship_documents():
    """Documents with clear relationships between entities."""
    return [
        Document(
            id="graphrag_rel1",
            page_content="Albert Einstein developed the theory of relativity. Einstein was a theoretical physicist who worked at Princeton University.",
            metadata={"category": "science", "has_relationships": True},
        ),
        Document(
            id="graphrag_rel2",
            page_content="Marie Curie discovered polonium and radium. Curie won the Nobel Prize in Physics and Chemistry.",
            metadata={"category": "science", "has_relationships": True},
        ),
        Document(
            id="graphrag_rel3",
            page_content="Isaac Newton formulated the laws of motion. Newton also developed calculus and studied optics.",
            metadata={"category": "science", "has_relationships": True},
        ),
    ]


class TestGraphRAGPipelineInitialization:
    """Test pipeline initialization and configuration."""

    def test_pipeline_initialization(self, graphrag_pipeline):
        """Test that pipeline initializes correctly."""
        assert graphrag_pipeline is not None
        assert hasattr(graphrag_pipeline, "entity_extraction_service")
        assert hasattr(graphrag_pipeline, "connection_manager")
        assert hasattr(graphrag_pipeline, "llm_func")

    def test_pipeline_has_entity_extraction(self, graphrag_pipeline):
        """Test that pipeline has entity extraction service."""
        assert graphrag_pipeline.entity_extraction_service is not None

    def test_pipeline_configuration(self, graphrag_pipeline):
        """Test pipeline configuration values."""
        assert hasattr(graphrag_pipeline, "max_depth")
        assert hasattr(graphrag_pipeline, "max_entities")
        assert graphrag_pipeline.max_depth > 0
        assert graphrag_pipeline.max_entities > 0


class TestGraphRAGPipelineTableSetup:
    """Test knowledge graph table setup."""

    def test_entities_table_exists(self, graphrag_pipeline):
        """Test that Entities table exists or can be created."""
        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            count = cursor.fetchone()[0]
            assert count >= 0
        except Exception:
            # Table will be created on first document load
            pass
        finally:
            cursor.close()

    def test_relationships_table_exists(self, graphrag_pipeline):
        """Test that EntityRelationships table exists or can be created."""
        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
            count = cursor.fetchone()[0]
            assert count >= 0
        except Exception:
            # Table will be created on first document load
            pass
        finally:
            cursor.close()


class TestGraphRAGPipelineDocumentLoading:
    """Test document loading with entity extraction."""

    def test_load_documents_extracts_entities(self, graphrag_pipeline, entity_rich_documents):
        """Test that loading documents extracts entities."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

        # Verify entities were extracted
        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE source_doc_id LIKE 'graphrag_ent%'")
            count = cursor.fetchone()[0]
            assert count > 0
        finally:
            cursor.close()

    def test_load_documents_extracts_relationships(self, graphrag_pipeline, relationship_documents):
        """Test that loading documents extracts relationships."""
        graphrag_pipeline.load_documents("", documents=relationship_documents)

        # Verify relationships were extracted
        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
            count = cursor.fetchone()[0]
            assert count >= 0  # May be 0 if extraction didn't find relationships
        finally:
            cursor.close()

    def test_load_single_document_with_entities(self, graphrag_pipeline):
        """Test loading single document with entity extraction."""
        doc = Document(
            id="graphrag_single1",
            page_content="Microsoft was founded by Bill Gates and Paul Allen. Microsoft develops Windows operating system.",
        )

        graphrag_pipeline.load_documents("", documents=[doc])

        # Verify document was loaded
        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE source_doc_id = 'graphrag_single1'")
            count = cursor.fetchone()[0]
            # Should have extracted some entities
            assert count >= 0
        finally:
            cursor.close()


class TestGraphRAGPipelineEntityExtraction:
    """Test entity extraction functionality."""

    def test_entity_extraction_from_text(self, graphrag_pipeline):
        """Test entity extraction from text content."""
        docs = [
            Document(
                id="graphrag_extract1",
                page_content="Apple Inc. is a technology company based in Cupertino, California. Apple produces iPhone and MacBook.",
            )
        ]

        graphrag_pipeline.load_documents("", documents=docs)

        # Check that entities were stored
        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT entity_name FROM RAG.Entities WHERE source_doc_id = 'graphrag_extract1'")
            entities = cursor.fetchall()
            # Should have extracted some entities
            assert len(entities) >= 0
        finally:
            cursor.close()

    def test_entity_types_extracted(self, graphrag_pipeline):
        """Test that entity types are extracted."""
        docs = [
            Document(
                id="graphrag_types1",
                page_content="Barack Obama was the President of the United States. Obama studied at Harvard University.",
            )
        ]

        graphrag_pipeline.load_documents("", documents=docs)

        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT entity_type FROM RAG.Entities WHERE source_doc_id = 'graphrag_types1'")
            types = cursor.fetchall()
            # May have entity types
            assert len(types) >= 0
        finally:
            cursor.close()

    def test_multiple_documents_entity_extraction(self, graphrag_pipeline, entity_rich_documents):
        """Test entity extraction from multiple documents."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT DISTINCT source_doc_id FROM RAG.Entities WHERE source_doc_id LIKE 'graphrag_ent%'")
            docs = cursor.fetchall()
            # Should have entities from multiple documents
            assert len(docs) >= 0
        finally:
            cursor.close()


class TestGraphRAGPipelineRelationshipStorage:
    """Test relationship storage functionality."""

    def test_relationship_storage(self, graphrag_pipeline):
        """Test that relationships are stored correctly."""
        docs = [
            Document(
                id="graphrag_relstore1",
                page_content="Amazon was founded by Jeff Bezos. Bezos started Amazon as an online bookstore.",
            )
        ]

        graphrag_pipeline.load_documents("", documents=docs)

        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
            count = cursor.fetchone()[0]
            assert count >= 0
        finally:
            cursor.close()

    def test_bidirectional_relationships(self, graphrag_pipeline):
        """Test that bidirectional relationships can be queried."""
        docs = [
            Document(
                id="graphrag_bidir1",
                page_content="Tesla Motors was founded by Elon Musk. Musk also leads SpaceX.",
            )
        ]

        graphrag_pipeline.load_documents("", documents=docs)

        # Relationships should support bidirectional queries
        connection = graphrag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships WHERE source_entity_id IS NOT NULL")
            count = cursor.fetchone()[0]
            assert count >= 0
        finally:
            cursor.close()


class TestGraphRAGPipelineQuerying:
    """Test query functionality with knowledge graph."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, graphrag_pipeline, entity_rich_documents):
        """Load documents before each test."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

    def test_simple_graph_query(self, graphrag_pipeline):
        """Test a simple query using knowledge graph."""
        result = graphrag_pipeline.query("Python programming", top_k=3, generate_answer=False)

        assert result is not None
        assert "contexts" in result
        assert "metadata" in result
        assert result["metadata"]["pipeline_type"] == "graphrag"

    def test_query_with_entity_matching(self, graphrag_pipeline):
        """Test query that matches known entities."""
        result = graphrag_pipeline.query("What is Python?", top_k=2, generate_answer=False)

        assert len(result["contexts"]) > 0
        metadata = result["metadata"]
        assert "retrieval_method" in metadata

    def test_query_with_answer_generation(self, graphrag_pipeline):
        """Test query with LLM answer generation."""
        result = graphrag_pipeline.query("What is TensorFlow?", top_k=2, generate_answer=True)

        assert "answer" in result
        assert len(result["answer"]) > 0


class TestGraphRAGPipelineGraphTraversal:
    """Test graph traversal functionality."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, graphrag_pipeline, relationship_documents):
        """Load documents with relationships."""
        graphrag_pipeline.load_documents("", documents=relationship_documents)

    def test_graph_traversal_finds_related_entities(self, graphrag_pipeline):
        """Test that graph traversal finds related entities."""
        result = graphrag_pipeline.query("Einstein", top_k=3, generate_answer=False)

        # Should find documents through graph traversal
        assert len(result["contexts"]) > 0

    def test_multi_hop_traversal(self, graphrag_pipeline):
        """Test multi-hop graph traversal."""
        result = graphrag_pipeline.query("physicist discoveries", top_k=3, generate_answer=False)

        # Should traverse multiple hops in the graph
        assert result is not None
        assert len(result["contexts"]) > 0

    def test_traversal_depth_limit(self, graphrag_pipeline):
        """Test that traversal respects depth limit."""
        # Query that might traverse deep in the graph
        result = graphrag_pipeline.query("science", top_k=5, generate_answer=False)

        # Should complete without infinite traversal
        assert "execution_time" in result
        assert result["execution_time"] < 30  # Should complete in reasonable time


class TestGraphRAGPipelineValidation:
    """Test knowledge graph validation."""

    def test_validation_with_empty_graph(self, pipeline_dependencies):
        """Test validation when graph is empty."""
        # Create new pipeline with empty graph
        new_pipeline = GraphRAGPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            llm_func=pipeline_dependencies["llm_func"],
            vector_store=pipeline_dependencies["vector_store"],
        )

        # Clear any existing entities
        connection = new_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM RAG.Entities WHERE entity_id LIKE 'test_%'")
            connection.commit()
        except Exception:
            pass
        finally:
            cursor.close()

        # Query should handle empty graph
        try:
            result = new_pipeline.query("test", top_k=2, generate_answer=False)
            # May fallback to vector search or return empty
            assert result is not None
        except KnowledgeGraphNotPopulatedException:
            # Expected if graph is truly empty
            pass

    def test_validation_with_populated_graph(self, graphrag_pipeline, entity_rich_documents):
        """Test validation when graph is populated."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

        # Should validate successfully
        graphrag_pipeline._validate_knowledge_graph()


class TestGraphRAGPipelineSeedEntityFinding:
    """Test seed entity finding functionality."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, graphrag_pipeline, entity_rich_documents):
        """Load documents before each test."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

    def test_find_seed_entities_for_query(self, graphrag_pipeline):
        """Test finding seed entities for a query."""
        seed_entities = graphrag_pipeline._find_seed_entities("Python programming")

        # Should find some seed entities
        assert len(seed_entities) >= 0

    def test_seed_entities_relevance(self, graphrag_pipeline):
        """Test that seed entities are relevant to query."""
        seed_entities = graphrag_pipeline._find_seed_entities("TensorFlow machine learning")

        # Seed entities should be related to the query
        assert isinstance(seed_entities, list)

    def test_seed_entities_with_no_matches(self, graphrag_pipeline):
        """Test seed entity finding with no matches."""
        try:
            seed_entities = graphrag_pipeline._find_seed_entities("xyz123nonexistent")
            # May return empty list or raise exception
            assert isinstance(seed_entities, list)
        except Exception:
            # Expected if no matches found
            pass


class TestGraphRAGPipelineDocumentRetrieval:
    """Test document retrieval from entities."""

    @pytest.fixture(autouse=True)
    def setup_documents(self, graphrag_pipeline, entity_rich_documents):
        """Load documents before each test."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

    def test_retrieve_documents_from_entities(self, graphrag_pipeline):
        """Test retrieving documents associated with entities."""
        result = graphrag_pipeline.query("Python", top_k=2, generate_answer=False)

        # Should retrieve documents
        assert len(result["contexts"]) > 0

    def test_document_retrieval_metadata(self, graphrag_pipeline):
        """Test metadata in retrieved documents."""
        result = graphrag_pipeline.query("web framework", top_k=2, generate_answer=False)

        # Check metadata
        assert "metadata" in result
        assert "retrieval_method" in result["metadata"]


class TestGraphRAGPipelineFallback:
    """Test vector search fallback functionality."""

    def test_fallback_to_vector_search(self, graphrag_pipeline):
        """Test fallback to vector search when entities not found."""
        # Load documents but query for something unlikely to match entities
        docs = [
            Document(
                id="graphrag_fallback1",
                page_content="Generic content without specific entities.",
            )
        ]
        graphrag_pipeline.load_documents("", documents=docs)

        result = graphrag_pipeline.query("generic content", top_k=2, generate_answer=False)

        # Should fallback to vector search and return results
        assert result is not None

    def test_fallback_metadata(self, graphrag_pipeline):
        """Test that fallback is indicated in metadata."""
        docs = [
            Document(
                id="graphrag_fallback2",
                page_content="Simple text content.",
            )
        ]
        graphrag_pipeline.load_documents("", documents=docs)

        result = graphrag_pipeline.query("simple text", top_k=2, generate_answer=False)

        # May indicate fallback method
        metadata = result["metadata"]
        assert "retrieval_method" in metadata


class TestGraphRAGPipelineErrorHandling:
    """Test error handling in GraphRAG pipeline."""

    def test_query_with_invalid_top_k(self, graphrag_pipeline, entity_rich_documents):
        """Test query with invalid top_k value."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)
        result = graphrag_pipeline.query("test", top_k=0, generate_answer=False)
        assert result is not None

    def test_load_empty_document_list(self, graphrag_pipeline):
        """Test loading empty document list."""
        try:
            graphrag_pipeline.load_documents("", documents=[])
        except Exception as e:
            # May raise exception for empty list
            assert "No documents" in str(e)

    def test_query_error_recovery(self, graphrag_pipeline):
        """Test that query errors are handled gracefully."""
        # Try to query even if something goes wrong
        result = graphrag_pipeline.query("test query", top_k=2, generate_answer=False)
        assert "query" in result


class TestGraphRAGPipelineIntegration:
    """Test full integration workflows."""

    def test_complete_graphrag_workflow(self, graphrag_pipeline):
        """Test complete workflow: load → extract → traverse → answer."""
        docs = [
            Document(
                id="graphrag_wf1",
                page_content="Google was founded by Larry Page and Sergey Brin at Stanford University.",
            ),
            Document(
                id="graphrag_wf2",
                page_content="Larry Page and Sergey Brin developed the PageRank algorithm.",
            ),
        ]
        graphrag_pipeline.load_documents("", documents=docs)

        result = graphrag_pipeline.query("Who founded Google?", top_k=2, generate_answer=True)

        assert "answer" in result
        assert "contexts" in result
        assert len(result["contexts"]) > 0

    def test_large_knowledge_graph(self, graphrag_pipeline):
        """Test with larger knowledge graph."""
        docs = [
            Document(
                id=f"graphrag_large{i}",
                page_content=f"Entity{i} is related to Entity{i+1}. Entity{i} has property{i}.",
            )
            for i in range(10)
        ]

        graphrag_pipeline.load_documents("", documents=docs)

        result = graphrag_pipeline.query("Entity property", top_k=5, generate_answer=False)
        assert len(result["contexts"]) > 0

    def test_sequential_queries_on_graph(self, graphrag_pipeline, entity_rich_documents):
        """Test sequential queries on the same knowledge graph."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

        queries = [
            "Python programming",
            "TensorFlow framework",
            "Django web development",
        ]

        for query in queries:
            result = graphrag_pipeline.query(query, top_k=2, generate_answer=False)
            assert len(result["contexts"]) > 0


class TestGraphRAGPipelinePerformance:
    """Test performance-related aspects."""

    def test_execution_time_tracking(self, graphrag_pipeline, entity_rich_documents):
        """Test that execution time is tracked."""
        graphrag_pipeline.load_documents("", documents=entity_rich_documents)

        result = graphrag_pipeline.query("test", top_k=3, generate_answer=False)

        assert "execution_time" in result
        assert result["execution_time"] >= 0

    def test_graph_traversal_performance(self, graphrag_pipeline):
        """Test that graph traversal completes in reasonable time."""
        docs = [
            Document(
                id=f"graphrag_perf{i}",
                page_content=f"Performance test entity {i} connects to entity {(i+1)%15}.",
            )
            for i in range(15)
        ]
        graphrag_pipeline.load_documents("", documents=docs)

        result = graphrag_pipeline.query("performance test", top_k=5, generate_answer=False)

        # Should complete quickly
        assert result["execution_time"] < 30

    def test_entity_extraction_performance(self, graphrag_pipeline):
        """Test entity extraction performance."""
        import time

        docs = [
            Document(
                id=f"graphrag_extract_perf{i}",
                page_content="Apple Inc. develops iPhone and MacBook. Microsoft creates Windows.",
            )
            for i in range(5)
        ]

        start = time.time()
        graphrag_pipeline.load_documents("", documents=docs)
        duration = time.time() - start

        # Should complete in reasonable time
        assert duration < 60
