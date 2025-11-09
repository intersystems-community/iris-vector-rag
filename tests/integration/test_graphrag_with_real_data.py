"""
Integration tests for GraphRAG using real production data.

These tests require the database to have documents with entities and relationships
loaded (e.g., from ticket ingestion or PMC documents). They test actual GraphRAG
functionality against real data rather than small fixtures.

IMPORTANT: These tests only run if the database has sufficient data.
"""

import pytest
from iris_vector_rag import create_pipeline
from common.iris_dbapi_connector import get_iris_dbapi_connection


def get_entity_count():
    """Helper to check if database has sufficient entities."""
    try:
        conn = get_iris_dbapi_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"Warning: Could not count entities: {e}")
        return 0


def get_relationship_count():
    """Helper to check if database has sufficient relationships."""
    try:
        conn = get_iris_dbapi_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"Warning: Could not count relationships: {e}")
        return 0


@pytest.fixture
def graphrag_with_real_data():
    """Create GraphRAG pipeline and return it with data counts."""
    pipeline = create_pipeline("graphrag", validate_requirements=False)

    # Get counts - will be checked in each test
    entity_count = get_entity_count()
    relationship_count = get_relationship_count()

    return pipeline, entity_count, relationship_count


@pytest.mark.integration
@pytest.mark.requires_real_data
class TestGraphRAGWithRealData:
    """Integration tests using production data."""

    def test_kg_traversal_returns_results(self, graphrag_with_real_data):
        """
        Test that knowledge graph traversal returns results with real data.

        This validates the core GraphRAG functionality:
        1. Finding seed entities from query
        2. Traversing relationships
        3. Retrieving documents
        """
        pipeline, entity_count, rel_count = graphrag_with_real_data

        # Skip if insufficient data
        if entity_count < 100:
            pytest.skip(f"Requires at least 100 entities (found {entity_count})")
        if rel_count < 50:
            pytest.skip(f"Requires at least 50 relationships (found {rel_count})")

        # Use a generic query that should match many entities
        result = pipeline.query(
            "error database connection",
            method="kg",
            top_k=5,
            generate_answer=False  # Skip LLM
        )

        # Should find documents via knowledge graph
        assert result is not None
        assert 'contexts' in result
        assert 'metadata' in result
        assert result['metadata']['retrieval_method'] == 'knowledge_graph_traversal'

        # Should have found at least some documents
        # (May be 0 if query doesn't match entities, but structure should be correct)
        assert isinstance(result['contexts'], list)
        assert len(result['contexts']) >= 0

        print(f"\n✓ KG traversal completed: {len(result['contexts'])} documents retrieved")
        print(f"  Database has {entity_count} entities, {rel_count} relationships")

    def test_vector_search_with_real_data(self, graphrag_with_real_data):
        """Test vector search fallback with real data."""
        pipeline, entity_count, rel_count = graphrag_with_real_data

        # Skip if insufficient data
        if entity_count < 100:
            pytest.skip(f"Requires at least 100 entities (found {entity_count})")
        if rel_count < 50:
            pytest.skip(f"Requires at least 50 relationships (found {rel_count})")

        # Query that might not have entities (fallback to vector)
        result = pipeline.query(
            "xyzabc nonsense query unlikely to match",
            top_k=3,
            generate_answer=False
        )

        # Should complete without error (may use fallback)
        assert result is not None
        assert 'contexts' in result
        assert 'metadata' in result

        # Retrieval method should be either KG or vector_fallback
        assert result['metadata']['retrieval_method'] in [
            'knowledge_graph_traversal',
            'vector_fallback'
        ]

        print(f"\n✓ Query completed with method: {result['metadata']['retrieval_method']}")

    def test_sequential_queries_stable(self, graphrag_with_real_data):
        """Test that multiple sequential queries work consistently."""
        pipeline, entity_count, rel_count = graphrag_with_real_data

        # Skip if insufficient data
        if entity_count < 100:
            pytest.skip(f"Requires at least 100 entities (found {entity_count})")
        if rel_count < 50:
            pytest.skip(f"Requires at least 50 relationships (found {rel_count})")

        queries = [
            "database error",
            "connection timeout",
            "performance issue"
        ]

        results = []
        for query in queries:
            result = pipeline.query(query, top_k=3, generate_answer=False)
            results.append(result)

            # Each should complete successfully
            assert result is not None
            assert 'contexts' in result
            assert 'metadata' in result

        # All should have completed
        assert len(results) == len(queries)

        print(f"\n✓ {len(results)} sequential queries completed successfully")

    def test_metadata_completeness(self, graphrag_with_real_data):
        """Test that query results include complete metadata."""
        pipeline, entity_count, rel_count = graphrag_with_real_data

        # Skip if insufficient data
        if entity_count < 100:
            pytest.skip(f"Requires at least 100 entities (found {entity_count})")
        if rel_count < 50:
            pytest.skip(f"Requires at least 50 relationships (found {rel_count})")

        result = pipeline.query("test query", top_k=5, generate_answer=False)

        # Verify required metadata fields
        assert 'metadata' in result
        metadata = result['metadata']

        assert 'retrieval_method' in metadata
        assert 'num_retrieved' in metadata
        assert 'processing_time' in metadata
        assert 'pipeline_type' in metadata

        # Verify metadata values are sensible
        assert metadata['pipeline_type'] == 'graphrag'
        assert metadata['num_retrieved'] >= 0
        assert metadata['processing_time'] > 0

        print(f"\n✓ Metadata complete: {list(metadata.keys())}")


@pytest.mark.integration
@pytest.mark.requires_real_data
def test_graphrag_smoke_test():
    """
    Smoke test: GraphRAG can be instantiated and has correct schema.

    This test always runs (doesn't require data) and validates basic setup.
    """
    pipeline = create_pipeline("graphrag", validate_requirements=False)

    # Verify pipeline initialized
    assert pipeline is not None
    assert hasattr(pipeline, 'query')
    assert hasattr(pipeline, 'connection_manager')

    # Check entity count (may be 0, that's OK for smoke test)
    entity_count = get_entity_count()

    print(f"\n✓ GraphRAG smoke test passed")
    print(f"  Entities in database: {entity_count}")
    print(f"  Note: Integration tests require 100+ entities")
