"""
Integration tests demonstrating @pytest.mark.dat_fixture decorator usage.

This test file serves as both a test and documentation for how to use
the .DAT fixture decorator in real tests.

Reference: specs/047-create-a-unified/tasks.md (T086)
"""

import pytest


# ==============================================================================
# FUNCTION-SCOPED FIXTURE EXAMPLES (default)
# ==============================================================================


@pytest.mark.dat_fixture("medical-graphrag-20")
def test_with_default_scope(fixture_metadata):
    """
    Example: Load fixture with default function scope.

    Fixture is loaded before this test and cleaned up after.
    Each test function gets a fresh database state.
    """
    # fixture_metadata provides info about loaded fixture
    if fixture_metadata:
        assert fixture_metadata.name == "medical-graphrag-20"
        assert fixture_metadata.checksum_valid
        print(f"Loaded {fixture_metadata.rows_loaded} rows in {fixture_metadata.load_time_seconds:.2f}s")
    else:
        pytest.skip("Fixture not available (no .DAT fixture found)")


@pytest.mark.dat_fixture("medical-graphrag-20", version="1.0.0")
def test_with_specific_version(fixture_metadata):
    """
    Example: Load specific fixture version.

    Useful when you need to test against a known fixture version
    rather than always using the latest.
    """
    if fixture_metadata:
        assert fixture_metadata.version == "1.0.0"
    else:
        pytest.skip("Fixture not available")


@pytest.mark.dat_fixture("medical-graphrag-20", cleanup_first=False)
def test_without_cleanup(fixture_metadata):
    """
    Example: Skip cleanup before loading.

    Useful when you want to test incremental updates or
    when previous test's data should remain.

    WARNING: This can cause data conflicts - use with caution!
    """
    if fixture_metadata:
        # Data from previous tests may still be present
        pass
    else:
        pytest.skip("Fixture not available")


# ==============================================================================
# CLASS-SCOPED FIXTURE EXAMPLES
# ==============================================================================


@pytest.mark.dat_fixture("medical-graphrag-20", scope="class")
class TestWithClassScopedFixture:
    """
    Example: Load fixture once for entire test class.

    Fixture is loaded before first test in class and cleaned up
    after last test in class. All tests share the same database state.

    Benefits:
    - Faster test execution (fixture loaded once)
    - Tests can modify data and subsequent tests see changes

    Drawbacks:
    - Tests are not isolated from each other
    - Test order matters
    """

    def test_first(self, fixture_metadata):
        """First test in class - fixture should be loaded."""
        if fixture_metadata:
            assert fixture_metadata.name == "medical-graphrag-20"
        else:
            pytest.skip("Fixture not available")

    def test_second(self, fixture_metadata):
        """Second test - same fixture, no reload."""
        if fixture_metadata:
            # Same fixture as test_first
            assert fixture_metadata.name == "medical-graphrag-20"
        else:
            pytest.skip("Fixture not available")

    def test_third(self, fixture_metadata):
        """Third test - still same fixture."""
        if fixture_metadata:
            assert fixture_metadata.name == "medical-graphrag-20"
        else:
            pytest.skip("Fixture not available")


# ==============================================================================
# REAL-WORLD USAGE EXAMPLE
# ==============================================================================


@pytest.mark.dat_fixture("medical-graphrag-20")
def test_graphrag_pipeline_with_fixture(fixture_metadata):
    """
    Real example: Test GraphRAG pipeline with .DAT fixture.

    This demonstrates the intended usage - fixture is automatically
    loaded, test runs, fixture is automatically cleaned up.
    """
    if not fixture_metadata:
        pytest.skip("Fixture not available")

    from iris_rag import create_pipeline

    # Create pipeline (database already has fixture data)
    pipeline = create_pipeline("graphrag", validate_requirements=False)

    # Query the pipeline
    result = pipeline.query(
        query="What are cancer treatment targets?",
        top_k=5,
    )

    # Verify results
    assert "answer" in result
    assert "retrieved_documents" in result
    assert len(result["retrieved_documents"]) > 0

    # Verify metadata
    assert result["metadata"]["retrieval_method"] in ["kg", "vector", "hybrid", "rrf", "text"]

    print(f"✓ Pipeline returned {len(result['retrieved_documents'])} documents")
    print(f"✓ Retrieval method: {result['metadata']['retrieval_method']}")


# ==============================================================================
# ERROR HANDLING EXAMPLES
# ==============================================================================


@pytest.mark.dat_fixture("non-existent-fixture")
def test_missing_fixture_fails_gracefully(fixture_metadata):
    """
    Example: Missing fixture fails with clear error.

    When fixture doesn't exist, pytest should fail with
    a clear error message (not crash).
    """
    # This test will fail during fixture setup
    # The error should clearly indicate which fixture is missing
    pytest.fail("This test should not reach here - fixture should fail to load")


# ==============================================================================
# PERFORMANCE VALIDATION
# ==============================================================================


@pytest.mark.dat_fixture("medical-graphrag-20")
def test_fixture_loads_quickly(fixture_metadata):
    """
    Validate: .DAT fixtures load in < 10 seconds.

    Per constitution and Feature 047 requirements, .DAT fixtures
    should load 100-200x faster than JSON fixtures.
    """
    if not fixture_metadata:
        pytest.skip("Fixture not available")

    # Verify load time is reasonable
    assert fixture_metadata.load_time_seconds < 10.0, \
        f"Fixture took {fixture_metadata.load_time_seconds:.2f}s (>10s threshold)"

    print(f"✓ Fixture loaded in {fixture_metadata.load_time_seconds:.2f}s")


# ==============================================================================
# CHECKSUM VALIDATION EXAMPLES
# ==============================================================================


@pytest.mark.dat_fixture("medical-graphrag-20")
def test_checksum_validation_passes(fixture_metadata):
    """
    Validate: Checksum validation ensures fixture integrity.

    Every fixture load should validate checksum to ensure
    the .DAT file hasn't been corrupted.
    """
    if not fixture_metadata:
        pytest.skip("Fixture not available")

    assert fixture_metadata.checksum_valid, \
        "Checksum validation should pass for valid fixture"

    print(f"✓ Checksum validation passed")


# ==============================================================================
# DOCUMENTATION EXAMPLES
# ==============================================================================


def test_without_fixture_decorator():
    """
    Example: Test without decorator runs normally.

    Tests that don't need fixtures work as usual.
    The dat_fixture_loader fixture returns None when no marker present.
    """
    # This test has no fixture - runs normally
    assert True


@pytest.mark.skip(reason="Example - not a real test")
@pytest.mark.dat_fixture("example-fixture")
def test_decorator_documentation_example(fixture_metadata):
    """
    Documentation: Full example with all parameters.

    @pytest.mark.dat_fixture(
        "fixture-name",           # Required: fixture name
        version="1.0.0",          # Optional: specific version (default: latest)
        cleanup_first=True,       # Optional: cleanup before load (default: True)
        scope="function"          # Optional: function/class/module/session (default: function)
    )
    """
    pass
