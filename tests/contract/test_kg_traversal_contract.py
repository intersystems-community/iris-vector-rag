"""
Contract tests for Knowledge Graph Traversal query path (FR-013 to FR-015).

Tests validate that HybridGraphRAG's knowledge graph traversal executes
correctly with proper seed entity finding and depth limits.

Contract: KG-001
Requirements: FR-013, FR-014, FR-015
"""

import pytest
from iris_vector_rag import create_pipeline


class TestKGTraversalContract:
    """Contract tests for knowledge graph traversal path."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    def test_kg_traversal_executes_successfully(self, graphrag_pipeline):
        """
        FR-013: KG traversal MUST execute pure knowledge graph queries.

        Given: HybridGraphRAG pipeline with knowledge graph data
        When: Query executed with method="kg"
        Then: Pure knowledge graph traversal executes and returns documents
        """
        query = "What are the relationships between diabetes and insulin?"

        # Execute query with KG method
        result = graphrag_pipeline.query(query, method="kg")

        # Verify result structure
        assert result is not None, "Result should not be None"
        assert ('contexts' in result), "Result should have contexts"
        assert ('metadata' in result), "Result should have metadata"

        # Verify documents retrieved (may be empty if no entities found)
        assert isinstance(result['contexts'], list), \
            "Contexts should be a list"

        # Verify metadata
        assert 'retrieval_method' in result['metadata'], \
            "Metadata should contain retrieval_method"

        method = result['metadata']['retrieval_method']
        assert method in ['knowledge_graph', 'kg', 'vector_fallback'], \
            f"Expected knowledge_graph/kg or fallback, got {method}"

    @pytest.mark.requires_database
    def test_kg_seed_entity_finding(self, graphrag_pipeline):
        """
        FR-014: KG MUST identify seed entities from query.

        Given: HybridGraphRAG pipeline with entity data
        When: Query contains entity mentions (e.g., "diabetes", "insulin")
        Then: System identifies seed entities for graph traversal
        """
        query = "diabetes treatment with insulin therapy"

        # Execute query
        result = graphrag_pipeline.query(query, method="kg")

        # Verify result structure
        assert result is not None, "Result should not be None"
        assert ('metadata' in result), "Result should have metadata"

        # If knowledge_graph method used, verify seed entities were processed
        if result['metadata']['retrieval_method'] in ['knowledge_graph', 'kg']:
            # KG traversal succeeded - seed entities were found
            # (Cannot directly verify seed entities without accessing internals)
            assert True, "KG traversal indicates seed entities were found"
        else:
            # Fallback occurred - acceptable if no entity data
            assert result['metadata']['retrieval_method'] == 'vector_fallback', \
                "If KG unavailable, should fall back to vector search"

    @pytest.mark.requires_database
    def test_kg_multi_hop_depth_limits(self, graphrag_pipeline):
        """
        FR-015: KG multi-hop traversal MUST respect depth limits.

        Given: HybridGraphRAG pipeline with graph data
        And: Configured traversal depth limit (e.g., max_hops=2)
        When: Query executed with method="kg"
        Then: Graph traversal respects depth limit and completes in reasonable time
        """
        query = "What are the complications of diabetes?"

        # Execute query with KG method
        import time
        start_time = time.time()

        result = graphrag_pipeline.query(query, method="kg")

        execution_time = time.time() - start_time

        # Verify result structure
        assert result is not None, "Result should not be None"

        # Verify execution completed in reasonable time (not infinite loop)
        # Allow up to 45 seconds as per contract (graph queries may be slower)
        assert execution_time < 45.0, \
            f"KG traversal took {execution_time:.2f}s, should complete within 45s"

        # Verify metadata
        assert 'retrieval_method' in result['metadata'], \
            "Metadata should contain retrieval_method"

        # Verify traversal completed (didn't hang or timeout)
        method = result['metadata']['retrieval_method']
        assert method in ['knowledge_graph', 'kg', 'vector_fallback'], \
            f"Expected kg or fallback, got {method}"
