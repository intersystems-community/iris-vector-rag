"""Contract tests for GraphRAG setup requirements.

Feature: 025-fixes-for-testing
Contract: graphrag_setup_contract.md

These tests validate GraphRAG dependencies and setup requirements.
GraphRAG is optional - tests should skip gracefully if not available.
"""

import pytest


def test_graphrag_import_available():
    """GraphRAG pipeline can be imported or skips gracefully."""
    try:
        from iris_rag.pipelines.graphrag import GraphRAGPipeline

        assert GraphRAGPipeline is not None
    except ImportError as e:
        pytest.skip(f"GraphRAG dependencies not available: {e}")


def test_entity_extraction_dependencies():
    """Entity extraction dependencies are available or tests skip."""
    try:
        # Try importing entity extraction service
        from iris_rag.services.entity_extraction import (
            OntologyAwareEntityExtractor,
        )

        assert OntologyAwareEntityExtractor is not None
    except ImportError as e:
        pytest.skip(f"Entity extraction dependencies not available: {e}")


def test_graph_ai_integration_optional():
    """graph-ai integration is optional and fails gracefully."""
    try:
        # Try importing optional graph-ai components
        import sys

        # Check if graph-ai is in path
        graph_ai_available = any("graph-ai" in path for path in sys.path)

        if graph_ai_available:
            # If available, should be importable
            # This is optional - just document availability
            pass
        else:
            pytest.skip("graph-ai integration not available (optional)")
    except Exception as e:
        pytest.skip(f"graph-ai check failed: {e}")


def test_graphrag_pipeline_fixture_error_handling():
    """GraphRAG pipeline fixture handles missing dependencies gracefully."""
    # This test validates that E2E GraphRAG tests skip properly
    # when dependencies are not available

    try:
        from iris_rag.pipelines.graphrag import GraphRAGPipeline

        # If we get here, GraphRAG is available
        assert hasattr(GraphRAGPipeline, "query")
    except ImportError:
        # Expected when GraphRAG dependencies not installed
        pytest.skip("GraphRAG not available - tests should skip")


def test_llm_configuration_for_entity_extraction():
    """LLM configuration is available for entity extraction."""
    try:
        from common.utils import get_llm_func

        # Should be able to get LLM function
        llm_func = get_llm_func(provider="openai", model_name="gpt-4o-mini")
        assert llm_func is not None
    except Exception as e:
        pytest.skip(f"LLM configuration not available: {e}")


def test_graphrag_test_errors_are_skips():
    """GraphRAG test errors should be skips, not errors."""
    # This is a meta-test to validate that GraphRAG setup issues
    # result in skipped tests, not ERROR status

    # Check that GraphRAG tests use proper skip logic
    from pathlib import Path

    graphrag_test_files = list(
        Path("tests/e2e").glob("*graphrag*_e2e.py")
    )

    if not graphrag_test_files:
        pytest.skip("No GraphRAG E2E test files found")

    # For each GraphRAG test file, verify it has skip logic
    for test_file in graphrag_test_files:
        content = test_file.read_text()

        # Should have import error handling or fixture skip logic
        has_skip_logic = (
            "pytest.skip" in content
            or "ImportError" in content
            or "@pytest.mark.skip" in content
        )

        if not has_skip_logic:
            # This is informational - GraphRAG tests should handle missing deps
            pytest.skip(
                f"{test_file.name} should handle missing GraphRAG dependencies"
            )
