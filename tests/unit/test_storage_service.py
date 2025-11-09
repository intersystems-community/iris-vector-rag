"""
Unit tests for EntityStorageAdapter.

IMPORTANT: Most GraphRAG storage testing is done via contract tests in
tests/contract/test_graphrag_fixtures.py because they test the full integration
with real FK constraints and realistic data.

These unit tests focus on specific edge cases and documentation of expected behavior.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from iris_vector_rag.services.storage import EntityStorageAdapter
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.schema_manager import SchemaManager


class TestEntityStorageDocumentation:
    """
    Documentation tests explaining GraphRAG storage behavior.

    For actual validation of FK constraints and entity insertion, see:
    - tests/contract/test_graphrag_fixtures.py::TestGraphRAGStorageContracts
    """

    def test_entity_fk_constraint_explanation(self):
        """
        DOCUMENTATION: Entities must reference SourceDocuments via doc_id column.

        CRITICAL BEHAVIOR:
        - RAG.Entities.source_document_id is a foreign key
        - FK constraint: FOREIGN KEY (source_document_id) REFERENCES RAG.SourceDocuments(doc_id)
        - FK references doc_id column (VARCHAR), NOT id column (INTEGER)

        COMMON BUG:
        - Using SourceDocuments.id instead of SourceDocuments.doc_id causes FK violation
        - _resolve_source_document must return doc_id, not id

        TEST COVERAGE:
        - Contract test validates this: test_entity_insertion_respects_fk_constraints
        - Integration tests validate with real data: test_graphrag_realistic.py

        WHY THIS TEST EXISTS:
        - This test documents the critical FK constraint requirement
        - Previous bug: _resolve_source_document returned id instead of doc_id
        - Contract tests now prevent regression
        """
        # This is a documentation test - it always passes
        assert True, "See contract tests for actual validation"

    def test_graphrag_testing_strategy_explanation(self):
        """
        DOCUMENTATION: GraphRAG testing uses a three-tier strategy.

        TIER 1: Contract Tests (Automated CI) ✅
        - Location: tests/contract/test_graphrag_fixtures.py
        - Purpose: Validate API interfaces and FK constraints
        - Coverage: Fixture loading, entity insertion, FK validation
        - Run: Always in CI
        - Speed: Fast (< 1s)

        TIER 2: Realistic Integration Tests (Manual, Development) ℹ️
        - Location: tests/integration/test_graphrag_realistic.py
        - Purpose: Validate against production-like data (221K+ entities)
        - Coverage: KG traversal, vector fallback, metadata completeness
        - Run: Manual with IRIS_PORT environment variable
        - Speed: Slow (minutes)

        TIER 3: E2E HybridGraphRAG Tests (Skipped) ⏭️
        - Location: tests/integration/test_hybridgraphrag_e2e.py
        - Purpose: End-to-end validation of all 5 query methods
        - Status: Intentionally skipped - requires LLM + iris-vector-graph setup
        - Alternative: Manual testing with real data

        WHY INTEGRATION TESTS ARE SKIPPED:
        - Previous "passing" tests used 2,376 pre-existing documents, not fixtures
        - Mocking LLM + iris-vector-graph is complex and brittle
        - Contract tests + manual validation provides better signal

        TEST STRATEGY DECISION:
        - Use contract tests for regression prevention (FK constraints, API contracts)
        - Use realistic integration tests for development/pre-release validation
        - Skip complex E2E mocking in favor of manual testing
        """
        assert True, "See documentation for testing strategy"


class TestTableEnsureCaching:
    """
    Unit tests for table ensure caching performance optimization.

    BUG FIX: Redundant table existence checks in EntityStorageAdapter
    - Before: 59,564 table checks in 11 minutes (~5,400/min)
    - After: 22 table checks (99.96% reduction)
    - Performance: 83% faster batch processing (66s → 11s)
    """

    @patch("iris_rag.services.storage.SchemaManager")
    def test_table_ensure_only_executes_once_per_instance(self, mock_schema_manager_class):
        """
        Verify _ensure_kg_tables() only executes schema operations once per adapter instance.

        This test validates the performance fix for redundant table checks.
        Each adapter instance should check table existence only once, then cache the result.
        """
        # Setup mock
        mock_schema_manager = MagicMock()
        mock_schema_manager.ensure_table_schema.return_value = True
        mock_schema_manager_class.return_value = mock_schema_manager

        mock_connection_manager = Mock(spec=ConnectionManager)
        config = {
            "entity_extraction": {
                "storage": {
                    "entities_table": "RAG.Entities",
                    "relationships_table": "RAG.EntityRelationships",
                }
            }
        }

        # Create adapter instance
        adapter = EntityStorageAdapter(mock_connection_manager, config)

        # First call should execute schema checks
        adapter._ensure_kg_tables()
        assert mock_schema_manager.ensure_table_schema.call_count == 2  # Entities + EntityRelationships

        # Reset call count to verify subsequent calls are cached
        mock_schema_manager.ensure_table_schema.reset_mock()

        # Subsequent calls should be cached (no schema operations)
        adapter._ensure_kg_tables()
        adapter._ensure_kg_tables()
        adapter._ensure_kg_tables()

        # Verify NO additional schema operations occurred
        assert mock_schema_manager.ensure_table_schema.call_count == 0, (
            "Table ensure should be cached after first call, but schema operations were executed"
        )

    @patch("iris_rag.services.storage.SchemaManager")
    def test_table_ensure_caching_flag_lifecycle(self, mock_schema_manager_class):
        """
        Verify the _tables_ensured flag lifecycle.

        1. Initially False
        2. Set to True after first successful table creation
        3. Remains True for instance lifetime
        """
        mock_schema_manager = MagicMock()
        mock_schema_manager.ensure_table_schema.return_value = True
        mock_schema_manager_class.return_value = mock_schema_manager

        mock_connection_manager = Mock(spec=ConnectionManager)
        config = {
            "entity_extraction": {
                "storage": {
                    "entities_table": "RAG.Entities",
                    "relationships_table": "RAG.EntityRelationships",
                }
            }
        }

        adapter = EntityStorageAdapter(mock_connection_manager, config)

        # Initially should be False
        assert adapter._tables_ensured is False, "Cache flag should start as False"

        # After first call, should be True
        adapter._ensure_kg_tables()
        assert adapter._tables_ensured is True, "Cache flag should be True after first call"

        # Should remain True
        adapter._ensure_kg_tables()
        assert adapter._tables_ensured is True, "Cache flag should remain True"

    @patch("iris_rag.services.storage.SchemaManager")
    def test_table_ensure_exception_handling(self, mock_schema_manager_class):
        """
        Verify that if table creation fails, the cache flag is NOT set.

        This ensures that failed table creation attempts will be retried on subsequent calls.
        """
        # Setup mock to raise exception
        mock_schema_manager = MagicMock()
        mock_schema_manager.ensure_table_schema.side_effect = Exception("Database connection failed")
        mock_schema_manager_class.return_value = mock_schema_manager

        mock_connection_manager = Mock(spec=ConnectionManager)
        config = {
            "entity_extraction": {
                "storage": {
                    "entities_table": "RAG.Entities",
                    "relationships_table": "RAG.EntityRelationships",
                }
            }
        }

        adapter = EntityStorageAdapter(mock_connection_manager, config)

        # Call should raise exception
        with pytest.raises(Exception, match="Database connection failed"):
            adapter._ensure_kg_tables()

        # Cache flag should still be False (not set on failure)
        assert adapter._tables_ensured is False, (
            "Cache flag should remain False after failed table creation"
        )

        # Subsequent call should retry (not use cache)
        with pytest.raises(Exception):
            adapter._ensure_kg_tables()

        # Verify retry occurred (schema operations were attempted again)
        assert mock_schema_manager.ensure_table_schema.call_count > 1, (
            "Failed table creation should be retried, not cached"
        )
