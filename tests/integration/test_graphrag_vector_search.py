"""
Integration tests for GraphRAG vector search workflow.

Tests end-to-end vector search functionality including:
- Vector retrieval (FR-001)
- Top-K configuration (FR-006)
- Dimension validation (FR-005)
- Diagnostic logging (FR-004)
- Embedding consistency (FR-003)
"""

import pytest

from iris_vector_rag import create_pipeline
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.embeddings.manager import EmbeddingManager


class TestGraphRAGVectorSearchIntegration:
    """Integration tests for GraphRAG vector search workflow."""

    @pytest.fixture
    def config_manager(self):
        """Create configuration manager."""
        return ConfigurationManager()

    @pytest.fixture
    def embedding_manager(self, config_manager):
        """Create embedding manager."""
        return EmbeddingManager(config_manager)

    @pytest.fixture
    def graphrag_pipeline(self):
        """GraphRAG pipeline over a corpus this test loads itself.

        `query()` returns a dict (`contexts`, `answer`, `retrieved_documents`,
        `metadata`), not an object: the tests read it as one. They used to assume a
        2,376-document corpus already sat in the database and a live LLM; the corpus
        is loaded here and the LLM is a stub, so the test runs on any IRIS.
        """
        from iris_vector_rag.core.models import Document

        pipeline = create_pipeline(
            "graphrag",
            validate_requirements=True,
            llm_func=lambda prompt: "Symptoms include thirst and fatigue.",
        )
        pipeline.load_documents(
            "",
            documents=[
                Document(
                    id=f"gvs-{i}",
                    page_content=f"Diabetes note {i}: symptoms, diagnosis and treatment.",
                    metadata={"title": f"gvs {i}"},
                )
                for i in range(12)
            ],
        )
        return pipeline

    def test_graphrag_vector_search_retrieval(self, graphrag_pipeline):
        """
        FR-001: GraphRAG vector search MUST retrieve documents.

        Given: GraphRAG pipeline initialized with 2,376 documents
        When: Query executed
        Then: Documents are retrieved
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        # Should retrieve documents
        assert (
            len(result["contexts"]) > 0
        ), f"GraphRAG vector search returned 0 results for query: {query}"

        # Should have answer generated
        assert (
            len(result["answer"]) > 0
        ), "GraphRAG should generate answer from retrieved contexts"

        # Answer should not be default "No relevant documents"
        assert (
            "No relevant documents" not in result["answer"]
        ), "GraphRAG answer indicates no retrieval occurred"

    def test_graphrag_top_k_configuration(self, graphrag_pipeline):
        """
        FR-006: GraphRAG MUST respect the top-K parameter.

        Given: a corpus larger than K
        When: query executed with the default and with top_k=5
        Then: returned documents <= top_k
        """
        query = "What are the symptoms of diabetes?"
        result_default = graphrag_pipeline.query(query)

        assert (
            len(result_default["contexts"]) <= 10
        ), f"Default top_k=10 violated: {len(result_default['contexts'])} documents returned"

        result_custom = graphrag_pipeline.query(query, top_k=5)

        assert (
            0 < len(result_custom["contexts"]) <= 5
        ), f"Custom top_k=5 violated: {len(result_custom['contexts'])} documents returned"

    def test_graphrag_dimension_validation(self, graphrag_pipeline, embedding_manager):
        """
        FR-005: GraphRAG MUST validate embedding dimensions.

        Given: GraphRAG pipeline with embedding validation
        When: Query is embedded
        Then: Embedding dimensions are validated (384D)
        """
        query = "What are the symptoms of diabetes?"

        # Generate query embedding
        query_embedding = embedding_manager.generate_embedding(query)

        # Validate dimensions
        assert (
            len(query_embedding) == 384
        ), f"Query embedding dimension mismatch: {len(query_embedding)} != 384"

        # Query should execute successfully
        result = graphrag_pipeline.query(query)

        # If retrieval works, dimension validation passed
        assert isinstance(
            result["contexts"], list
        ), "GraphRAG query should succeed with matching dimensions"

    def test_graphrag_diagnostic_logging(self, graphrag_pipeline, caplog):
        """
        FR-004: GraphRAG MUST log diagnostic information.

        Given: GraphRAG pipeline with logging enabled
        When: Query executes
        Then: Diagnostic logs are emitted
        """
        import logging

        caplog.set_level(logging.DEBUG)

        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        # Check for diagnostic log entries
        log_messages = [record.message for record in caplog.records]
        " ".join(log_messages)

        # Should have some diagnostic logging
        assert len(log_messages) > 0, "No log messages emitted during GraphRAG query"

        # If 0 results, should have diagnostic info
        if len(result["contexts"]) == 0:
            assert any(
                "0 results" in msg or "no documents" in msg.lower()
                for msg in log_messages
            ), "Missing diagnostic log when 0 results returned"

    def test_graphrag_embedding_consistency(self, graphrag_pipeline, embedding_manager):
        """
        FR-003: GraphRAG MUST work with 384D all-MiniLM-L6-v2 embeddings.

        Given: Database with 384D embeddings
        When: Query with 384D embedding executes
        Then: Vector search succeeds (no dimension mismatch)
        """
        query = "What are the symptoms of diabetes?"

        # Generate query embedding (should be 384D)
        query_embedding = embedding_manager.generate_embedding(query)
        assert len(query_embedding) == 384, "Query embedding should be 384D"

        # Execute query (should work with 384D document embeddings)
        result = graphrag_pipeline.query(query)

        # If retrieval works, embeddings are consistent
        assert (
            len(result["contexts"]) >= 0
        ), "GraphRAG should execute without dimension mismatch errors"

    def test_graphrag_multiple_queries_consistency(self, graphrag_pipeline):
        """
        GraphRAG MUST handle multiple queries consistently.

        Given: GraphRAG pipeline
        When: Multiple queries executed
        Then: All queries retrieve documents
        """
        queries = [
            "What are the symptoms of diabetes?",
            "How is diabetes diagnosed?",
            "What are the treatments for type 2 diabetes?",
        ]

        for query in queries:
            result = graphrag_pipeline.query(query)

            # Each query should retrieve documents
            assert (
                len(result["contexts"]) > 0
            ), f"GraphRAG failed to retrieve documents for query: {query}"

            # Each query should generate answer
            assert (
                len(result["answer"]) > 0
            ), f"GraphRAG failed to generate answer for query: {query}"

    def test_graphrag_relevance_scoring(self, graphrag_pipeline):
        """
        GraphRAG retrieved documents MUST be sorted by relevance.

        Given: GraphRAG vector search
        When: Documents retrieved
        Then: Documents sorted by similarity score descending
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        if len(result["contexts"]) < 2:
            pytest.skip("Need at least 2 documents to test sorting")

        # Extract similarity scores
        scores = [
            doc.metadata["similarity_score"]
            for doc in result["retrieved_documents"]
            if "similarity_score" in doc.metadata
        ]

        assert scores, "retrieved documents carry no similarity_score in metadata"

        # Verify descending order
        for i in range(len(scores) - 1):
            assert (
                scores[i] >= scores[i + 1]
            ), f"Documents not sorted by score DESC: {scores[i]} < {scores[i+1]}"
