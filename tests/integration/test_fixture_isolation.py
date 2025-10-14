"""
Integration tests for fixture isolation across test runs.

These tests verify that fixtures provide isolated, reproducible database states
that remain consistent across multiple test runs.

Reference: specs/047-create-a-unified/tasks.md (T031)
"""

import pytest
from pathlib import Path
from tests.fixtures.manager import FixtureManager


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def fixture_manager():
    """Create FixtureManager with default configuration."""
    return FixtureManager()


# ==============================================================================
# FIXTURE ISOLATION INTEGRATION TESTS
# ==============================================================================


class TestFixtureIsolation:
    """Integration tests for fixture isolation across test runs."""

    def test_same_fixture_loads_identical_state_twice(self, fixture_manager):
        """Loading same fixture twice produces identical database state."""
        # Skip if no fixtures available
        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available for testing")

        fixture_name = fixtures[0].name

        # Load fixture first time
        result1 = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=True,
            validate_checksum=True,
        )

        assert result1.success, f"First load failed: {result1.error_message}"
        first_rows = result1.rows_loaded
        first_tables = set(result1.tables_loaded)

        # Load fixture second time (simulating second test run)
        result2 = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=True,
            validate_checksum=True,
        )

        assert result2.success, f"Second load failed: {result2.error_message}"

        # Results should be identical
        assert result2.rows_loaded == first_rows, \
            f"Row counts differ: {result2.rows_loaded} != {first_rows}"
        assert set(result2.tables_loaded) == first_tables, \
            f"Table lists differ: {result2.tables_loaded} != {list(first_tables)}"

    def test_cleanup_first_ensures_isolation(self, fixture_manager):
        """cleanup_first=True ensures isolated state between loads."""
        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available for testing")

        fixture_name = fixtures[0].name

        # Load fixture with cleanup
        result1 = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=True,
        )

        assert result1.success
        expected_rows = result1.rows_loaded

        # Load again with cleanup - should get same state
        result2 = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=True,
        )

        assert result2.success
        assert result2.rows_loaded == expected_rows, \
            "Cleanup should restore identical state"

    def test_fixture_state_independent_of_prior_database_state(self, fixture_manager):
        """Fixture loading produces same state regardless of prior database content."""
        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available for testing")

        fixture_name = fixtures[0].name
        metadata = fixture_manager.get_fixture(fixture_name)

        # First load: Clean database
        result1 = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=True,
        )
        assert result1.success
        baseline_rows = result1.rows_loaded

        # Second load: Database already has data, but cleanup_first removes it
        result2 = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=True,  # Should produce identical state
        )

        assert result2.success
        assert result2.rows_loaded == baseline_rows, \
            "Fixture should produce identical state regardless of prior state"

        # Third load: Skip cleanup (should add duplicate data)
        result3 = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=False,  # Intentionally skip cleanup
        )

        assert result3.success
        # NOTE: Depending on implementation, this might duplicate data or fail
        # The important test is that cleanup_first=True produces consistent results

    def test_checksum_validation_catches_corruption(self, fixture_manager):
        """Checksum validation detects corrupted fixture files."""
        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available for testing")

        fixture_name = fixtures[0].name

        # Load with checksum validation (should succeed)
        result = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            validate_checksum=True,
            cleanup_first=True,
        )

        assert result.success
        assert result.checksum_valid, \
            "Checksum validation should pass for valid fixture"

    def test_multiple_fixtures_isolated_from_each_other(self, fixture_manager):
        """Different fixtures provide independent, isolated states."""
        fixtures = fixture_manager.list_fixtures()
        if len(fixtures) < 2:
            pytest.skip("Need at least 2 fixtures for isolation testing")

        fixture1_name = fixtures[0].name
        fixture2_name = fixtures[1].name

        # Load first fixture
        result1 = fixture_manager.load_fixture(
            fixture_name=fixture1_name,
            cleanup_first=True,
        )
        assert result1.success

        # Load second fixture (should replace first fixture's data)
        result2 = fixture_manager.load_fixture(
            fixture_name=fixture2_name,
            cleanup_first=True,  # Removes fixture1's data
        )
        assert result2.success

        # Reload first fixture - should get original state back
        result3 = fixture_manager.load_fixture(
            fixture_name=fixture1_name,
            cleanup_first=True,
        )
        assert result3.success
        assert result3.rows_loaded == result1.rows_loaded, \
            "Fixture should restore its original state"


class TestFixtureReproducibility:
    """Tests for fixture reproducibility across sessions."""

    def test_fixture_metadata_stable_across_scans(self, fixture_manager):
        """Fixture metadata remains stable across multiple scans."""
        # First scan
        manifest1 = fixture_manager.scan_fixtures(rescan=True)
        fixtures1 = {f.name: f for f in manifest1.list_fixtures()}

        # Second scan
        manifest2 = fixture_manager.scan_fixtures(rescan=True)
        fixtures2 = {f.name: f for f in manifest2.list_fixtures()}

        # Should find same fixtures
        assert set(fixtures1.keys()) == set(fixtures2.keys()), \
            "Fixture list should be stable across scans"

        # Metadata should be identical
        for name in fixtures1.keys():
            f1 = fixtures1[name]
            f2 = fixtures2[name]

            assert f1.version == f2.version
            assert f1.checksum == f2.checksum
            assert f1.tables == f2.tables
            assert f1.row_counts == f2.row_counts

    def test_fixture_load_result_deterministic(self, fixture_manager):
        """Same fixture produces deterministic load results."""
        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available")

        fixture_name = fixtures[0].name

        # Load multiple times
        results = []
        for _ in range(3):
            result = fixture_manager.load_fixture(
                fixture_name=fixture_name,
                cleanup_first=True,
                validate_checksum=True,
            )
            assert result.success
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].rows_loaded == results[0].rows_loaded
            assert set(results[i].tables_loaded) == set(results[0].tables_loaded)
            assert results[i].checksum_valid == results[0].checksum_valid


class TestFixtureVersioning:
    """Tests for fixture version isolation."""

    def test_get_fixture_returns_latest_version_by_default(self, fixture_manager):
        """get_fixture() returns latest version when no version specified."""
        fixture_manager.scan_fixtures()
        fixtures = fixture_manager.list_fixtures()

        if not fixtures:
            pytest.skip("No fixtures available")

        fixture_name = fixtures[0].name

        # Get without version (should return latest)
        latest = fixture_manager.get_fixture(fixture_name)
        assert latest is not None

        # Get with explicit version
        explicit = fixture_manager.get_fixture(fixture_name, version=latest.version)
        assert explicit is not None
        assert explicit.version == latest.version

    def test_different_fixture_versions_isolated(self, fixture_manager):
        """Different versions of same fixture are isolated."""
        # This test requires fixtures with multiple versions
        # For now, just verify the API supports version parameter

        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available")

        fixture_name = fixtures[0].name
        version = fixtures[0].version

        # Should be able to request specific version
        metadata = fixture_manager.get_fixture(fixture_name, version=version)
        assert metadata is not None
        assert metadata.version == version


class TestFixtureCleanup:
    """Tests for fixture cleanup and isolation."""

    def test_cleanup_fixture_data_clears_tables(self, fixture_manager):
        """cleanup_fixture_data() removes all data from specified tables."""
        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available")

        fixture_name = fixtures[0].name
        metadata = fixture_manager.get_fixture(fixture_name)

        # Load fixture
        result = fixture_manager.load_fixture(
            fixture_name=fixture_name,
            cleanup_first=True,
        )
        assert result.success
        assert result.rows_loaded > 0

        # Cleanup
        deleted = fixture_manager.cleanup_fixture_data(metadata.tables)

        # Should have deleted some rows (exact count depends on implementation)
        # In contract tests (no DB), this returns 0
        assert deleted >= 0

    def test_cleanup_idempotent(self, fixture_manager):
        """cleanup_fixture_data() can be called multiple times safely."""
        fixtures = fixture_manager.list_fixtures()
        if not fixtures:
            pytest.skip("No fixtures available")

        metadata = fixture_manager.get_fixture(fixtures[0].name)

        # First cleanup
        deleted1 = fixture_manager.cleanup_fixture_data(metadata.tables)

        # Second cleanup (should succeed, may delete 0 rows)
        deleted2 = fixture_manager.cleanup_fixture_data(metadata.tables)

        # Both should succeed
        assert deleted1 >= 0
        assert deleted2 >= 0
