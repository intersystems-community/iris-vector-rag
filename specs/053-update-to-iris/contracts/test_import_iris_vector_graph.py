"""
Contract Tests: iris-vector-graph 1.1.1 Import Compatibility

These tests validate FR-001 through FR-010 from the feature specification.
They enforce the new import structure and dependency version requirements.

Test Strategy (TDD):
- Write these tests FIRST (they should fail initially)
- Implement the import path changes
- Tests pass when implementation is complete

Constitutional Compliance:
- Section III: Test-Driven Development with Contract Tests
- Tests written before implementation
- Each requirement has corresponding test
"""

import pytest
import sys
from importlib.metadata import version, PackageNotFoundError


@pytest.fixture(autouse=True)
def clean_import_cache():
    """Clean up import cache before and after each test to prevent pollution between tests."""
    # Clean before test - remove any iris_vector_graph or iris_graph_core modules
    modules_to_remove = [k for k in list(sys.modules.keys()) if 'iris_vector_graph' in k or 'iris_graph_core' in k]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)

    # Also clear hybrid_graphrag_discovery to reset GraphCoreDiscovery's internal cache
    if 'iris_rag.pipelines.hybrid_graphrag_discovery' in sys.modules:
        sys.modules.pop('iris_rag.pipelines.hybrid_graphrag_discovery')

    yield

    # Clean after test
    modules_to_remove = [k for k in list(sys.modules.keys()) if 'iris_vector_graph' in k or 'iris_graph_core' in k]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)


class TestIrisVectorGraphImports:
    """
    Contract tests for iris-vector-graph 1.1.1 import compatibility.

    These tests validate that all required modules can be imported from the
    top-level iris_vector_graph package (not iris_vector_graph_core).
    """

    def test_import_iris_graph_engine(self):
        """
        FR-001: Package MUST import IRISGraphEngine from iris_vector_graph module.

        Validates that IRISGraphEngine is available via:
            from iris_vector_graph import IRISGraphEngine

        NOT from:
            from iris_vector_graph_core.engine import IRISGraphEngine
        """
        try:
            from iris_vector_graph import IRISGraphEngine
            assert IRISGraphEngine is not None, "IRISGraphEngine should not be None"
            assert hasattr(IRISGraphEngine, '__init__'), "IRISGraphEngine should be a class"
        except ImportError as e:
            pytest.fail(
                f"FR-001 FAILED: Cannot import IRISGraphEngine from iris_vector_graph. "
                f"Error: {e}. "
                f"Ensure iris-vector-graph >= 1.1.1 is installed."
            )

    def test_import_hybrid_search_fusion(self):
        """
        FR-002: Package MUST import HybridSearchFusion from iris_vector_graph module.

        Validates that HybridSearchFusion is available via:
            from iris_vector_graph import HybridSearchFusion

        NOT from:
            from iris_vector_graph_core.fusion import HybridSearchFusion
        """
        try:
            from iris_vector_graph import HybridSearchFusion
            assert HybridSearchFusion is not None, "HybridSearchFusion should not be None"
            assert hasattr(HybridSearchFusion, '__init__'), "HybridSearchFusion should be a class"
        except ImportError as e:
            pytest.fail(
                f"FR-002 FAILED: Cannot import HybridSearchFusion from iris_vector_graph. "
                f"Error: {e}. "
                f"Ensure iris-vector-graph >= 1.1.1 is installed."
            )

    def test_import_text_search_engine(self):
        """
        FR-003: Package MUST import TextSearchEngine from iris_vector_graph module.

        Validates that TextSearchEngine is available via:
            from iris_vector_graph import TextSearchEngine

        NOT from:
            from iris_vector_graph_core.text_search import TextSearchEngine
        """
        try:
            from iris_vector_graph import TextSearchEngine
            assert TextSearchEngine is not None, "TextSearchEngine should not be None"
            assert hasattr(TextSearchEngine, '__init__'), "TextSearchEngine should be a class"
        except ImportError as e:
            pytest.fail(
                f"FR-003 FAILED: Cannot import TextSearchEngine from iris_vector_graph. "
                f"Error: {e}. "
                f"Ensure iris-vector-graph >= 1.1.1 is installed."
            )

    def test_import_vector_optimizer(self):
        """
        FR-004: Package MUST import VectorOptimizer from iris_vector_graph module.

        Validates that VectorOptimizer is available via:
            from iris_vector_graph import VectorOptimizer

        NOT from:
            from iris_vector_graph_core.vector_utils import VectorOptimizer
        """
        try:
            from iris_vector_graph import VectorOptimizer
            assert VectorOptimizer is not None, "VectorOptimizer should not be None"
            assert hasattr(VectorOptimizer, '__init__'), "VectorOptimizer should be a class"
        except ImportError as e:
            pytest.fail(
                f"FR-004 FAILED: Cannot import VectorOptimizer from iris_vector_graph. "
                f"Error: {e}. "
                f"Ensure iris-vector-graph >= 1.1.1 is installed."
            )

    def test_old_iris_vector_graph_core_import_fails(self):
        """
        FR-006: Package MUST NOT reference iris_vector_graph_core anywhere in import statements.

        Validates that the old iris_vector_graph_core submodule does NOT exist.
        This confirms we're using iris-vector-graph >= 1.1.1.

        Expected behavior: ImportError when trying to import from iris_vector_graph_core
        """
        with pytest.raises(
            ImportError,
            match="No module named 'iris_vector_graph_core'|cannot import name"
        ):
            # This import should FAIL because iris_vector_graph_core doesn't exist in 1.1.1+
            from iris_vector_graph_core.engine import IRISGraphEngine  # noqa: F401

    def test_iris_vector_graph_version_requirement(self):
        """
        FR-005 & FR-009: Package MUST specify iris-vector-graph >= 1.1.1 as dependency.

        Validates that iris-vector-graph is installed and version is >= 1.1.1.
        This ensures the package version constraint in pyproject.toml is enforced.
        """
        try:
            pkg_version = version("iris-vector-graph")
        except PackageNotFoundError:
            pytest.fail(
                "FR-005 FAILED: iris-vector-graph is not installed. "
                "Install with: pip install rag-templates[hybrid-graphrag]"
            )

        # Parse version string (e.g., "1.1.1" or "2.0.0")
        major, minor, patch = map(int, pkg_version.split(".")[:3])

        # Check version >= 1.1.1
        if major < 1 or (major == 1 and minor < 1) or (major == 1 and minor == 1 and patch < 1):
            pytest.fail(
                f"FR-009 FAILED: iris-vector-graph version {pkg_version} is too old. "
                f"Requires >= 1.1.1. "
                f"Upgrade with: pip install --upgrade iris-vector-graph"
            )

        # Success - version is >= 1.1.1
        print(f"✅ iris-vector-graph version {pkg_version} meets requirement (>= 1.1.1)")


class TestHybridGraphRAGDiscoveryImports:
    """
    Integration tests for hybrid_graphrag_discovery.py import behavior.

    These tests validate that the GraphCoreDiscovery class successfully imports
    the required modules using the new import paths.
    """

    def test_graph_core_discovery_imports_successfully(self):
        """
        FR-008: All existing HybridGraphRAG functionality MUST continue to work.

        Validates that GraphCoreDiscovery.import_graph_core_modules() succeeds
        and returns the correct module classes.
        """
        from iris_vector_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

        discovery = GraphCoreDiscovery()
        success, modules = discovery.import_graph_core_modules()

        assert success, "GraphCoreDiscovery should successfully import modules"
        assert "IRISGraphEngine" in modules, "IRISGraphEngine should be in imported modules"
        assert "HybridSearchFusion" in modules, "HybridSearchFusion should be in imported modules"
        assert "TextSearchEngine" in modules, "TextSearchEngine should be in imported modules"
        assert "VectorOptimizer" in modules, "VectorOptimizer should be in imported modules"

        # Validate that the imported classes are the same as direct imports
        from iris_vector_graph import (
            IRISGraphEngine,
            HybridSearchFusion,
            TextSearchEngine,
            VectorOptimizer
        )

        assert modules["IRISGraphEngine"] is IRISGraphEngine
        assert modules["HybridSearchFusion"] is HybridSearchFusion
        assert modules["TextSearchEngine"] is TextSearchEngine
        assert modules["VectorOptimizer"] is VectorOptimizer

    def test_graph_core_discovery_caches_imports(self):
        """
        Performance validation: Verify imports are cached after first load.

        This is not a functional requirement but validates expected behavior.
        """
        from iris_vector_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

        discovery = GraphCoreDiscovery()

        # First import
        success1, modules1 = discovery.import_graph_core_modules()
        assert success1, "First import should succeed"

        # Second import (should use cache)
        success2, modules2 = discovery.import_graph_core_modules()
        assert success2, "Cached import should succeed"

        # Validate cache was used (same object references)
        assert modules1 is modules2, "Second call should return cached modules"


class TestBackwardCompatibilityGuarantees:
    """
    Tests validating that the update doesn't break existing functionality.

    These tests ensure FR-008 (existing functionality continues to work) and
    FR-010 (all existing tests pass).
    """

    def test_hybrid_graphrag_pipeline_imports(self):
        """
        FR-008: Validate HybridGraphRAGPipeline can still be imported.

        This is a smoke test ensuring the main user-facing API still works.
        """
        try:
            from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
            assert HybridGraphRAGPipeline is not None
        except ImportError as e:
            pytest.fail(
                f"FR-008 FAILED: HybridGraphRAGPipeline import broken. Error: {e}"
            )

    def test_create_pipeline_graphrag_type(self):
        """
        FR-008: Validate create_pipeline("graphrag") still works.

        This validates the pipeline factory integration.
        """
        # This test requires mock LLM and database connection, so we only
        # validate that the import doesn't raise an error.
        # Full integration test is in test_hybridgraphrag_e2e.py
        try:
            from iris_vector_rag import create_pipeline
            # Don't actually create the pipeline (requires DB), just validate import
            assert create_pipeline is not None
        except ImportError as e:
            pytest.fail(
                f"FR-008 FAILED: create_pipeline import broken. Error: {e}"
            )


class TestErrorMessages:
    """
    Tests validating clear error messages when requirements are not met.

    Validates FR-009 (clear error messages for version issues).
    """

    @pytest.mark.skipif(
        sys.version_info < (3, 10),
        reason="This test requires Python 3.10+ for version checking"
    )
    def test_helpful_error_when_package_missing(self):
        """
        FR-009: Package MUST provide clear error messages if iris-vector-graph is missing.

        This test validates that GraphCoreDiscovery logs helpful messages
        when the package is not installed.
        """
        from iris_vector_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

        # We can't actually uninstall the package during tests, so we test
        # the error message logging logic by checking the _log_dependency_help method
        discovery = GraphCoreDiscovery()

        # Validate the helper method exists and is callable
        assert hasattr(discovery, '_log_dependency_help')
        assert callable(discovery._log_dependency_help)

        # The actual error message is logged, we just validate the method exists
        # Full integration test would require mocking the import


if __name__ == "__main__":
    """
    Run contract tests directly with:
        python specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py

    Or with pytest:
        pytest specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py -v
    """
    pytest.main([__file__, "-v", "--tb=short"])
