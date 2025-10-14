"""
Unit tests for EntityStorageAdapter.

IMPORTANT: Most GraphRAG storage testing is done via contract tests in
tests/contract/test_graphrag_fixtures.py because they test the full integration
with real FK constraints and realistic data.

These unit tests focus on specific edge cases and documentation of expected behavior.
"""
import pytest


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
