"""
Contract tests for FixtureManager.

These tests define the expected behavior of FixtureManager and MUST fail
initially (no implementation exists yet). They serve as executable
specifications for TDD.

Contract Reference: specs/047-create-a-unified/contracts/fixture_manager_contract.md
"""

import pytest
from pathlib import Path
import time


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def fixtures_root(tmp_path):
    """Temporary fixtures directory for testing."""
    root = tmp_path / "fixtures"
    root.mkdir()
    return root


@pytest.fixture
def fixture_manager(fixtures_root):
    """Create FixtureManager instance for testing."""
    from tests.fixtures.manager import FixtureManager

    return FixtureManager(fixtures_root=fixtures_root)


@pytest.fixture
def sample_manifest_data():
    """Sample manifest data for testing."""
    return {
        "name": "test-fixture",
        "version": "1.0.0",
        "description": "Test fixture for contract tests",
        "created_at": "2025-01-14T12:00:00Z",
        "created_by": "iris-devtools",
        "source_type": "dat",
        "tables": ["RAG.SourceDocuments", "RAG.Entities"],
        "row_counts": {"RAG.SourceDocuments": 3, "RAG.Entities": 21},
        "checksum": "sha256:abc123",
        "requires_embeddings": True,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "namespace": "USER",
    }


@pytest.fixture
def create_test_fixture(fixtures_root, sample_manifest_data):
    """Helper to create a test fixture directory with manifest."""

    def _create(name="test-fixture", with_dat_file=True):
        import json
        import hashlib

        fixture_dir = fixtures_root / "dat" / name
        fixture_dir.mkdir(parents=True)

        # Create .DAT file first if needed
        dat_data = b"mock dat data"
        if with_dat_file:
            (fixture_dir / "data.dat").write_bytes(dat_data)

            # Compute checksum for the .DAT file
            sha256 = hashlib.sha256()
            sha256.update(dat_data)
            checksum = f"sha256:{sha256.hexdigest()}"
        else:
            checksum = "sha256:abc123"  # Placeholder

        # Create manifest.json with correct checksum
        manifest_data = sample_manifest_data.copy()
        manifest_data["name"] = name
        manifest_data["checksum"] = checksum
        with open(fixture_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        return fixture_dir

    return _create


# ==============================================================================
# CONSTRUCTOR TESTS
# ==============================================================================


@pytest.mark.contract
class TestFixtureManagerConstructor:
    """Contract tests for FixtureManager.__init__()."""

    def test_accepts_custom_fixtures_root(self, tmp_path):
        """✅ Accepts custom fixtures_root path."""
        from tests.fixtures.manager import FixtureManager

        custom_root = tmp_path / "custom_fixtures"
        manager = FixtureManager(fixtures_root=custom_root)

        assert manager.fixtures_root == custom_root

    def test_creates_fixtures_root_if_missing(self, tmp_path):
        """✅ Creates fixtures_root directory if missing."""
        from tests.fixtures.manager import FixtureManager

        non_existent = tmp_path / "does_not_exist" / "fixtures"
        manager = FixtureManager(fixtures_root=non_existent)

        assert non_existent.exists()
        assert non_existent.is_dir()

    def test_loads_existing_manifest(self, fixtures_root):
        """✅ Loads existing manifest.json."""
        from tests.fixtures.manager import FixtureManager
        import json

        # Create manifest.json
        manifest_path = fixtures_root / "manifest.json"
        manifest_data = {"version": "1.0.0", "fixtures": {}}
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        manager = FixtureManager(fixtures_root=fixtures_root)

        # Manager should have loaded the manifest
        assert hasattr(manager, "_manifest")

    def test_initializes_empty_registry_if_manifest_missing(self, fixtures_root):
        """✅ Initializes empty registry if manifest missing."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager(fixtures_root=fixtures_root)

        # Should initialize empty registry, not fail
        assert hasattr(manager, "_registry")

    def test_accepts_backend_mode_override(self, fixtures_root):
        """✅ Accepts backend_mode override."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager(
            fixtures_root=fixtures_root, backend_mode="enterprise"
        )

        assert manager.backend_mode == "enterprise"


# ==============================================================================
# LOAD_FIXTURE TESTS
# ==============================================================================


@pytest.mark.contract
class TestLoadFixture:
    """Contract tests for FixtureManager.load_fixture()."""

    def test_loads_dat_fixture_successfully(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Loads .DAT fixture successfully."""
        create_test_fixture("test-fixture")

        result = fixture_manager.load_fixture("test-fixture")

        assert result.success
        assert result.fixture_name == "test-fixture"
        assert result.load_time_seconds > 0

    def test_resolves_latest_version_when_none_specified(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Resolves 'latest' version when version=None."""
        create_test_fixture("test-fixture")

        result = fixture_manager.load_fixture("test-fixture", version=None)

        assert result.success
        assert result.fixture_version == "1.0.0"  # Latest available

    def test_validates_checksum_and_fails_on_mismatch(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Validates checksum and fails on mismatch."""
        fixture_dir = create_test_fixture("test-fixture")

        # Corrupt the .DAT file to cause checksum mismatch
        (fixture_dir / "data.dat").write_bytes(b"corrupted data")

        from tests.fixtures.manager import ChecksumMismatchError

        with pytest.raises(ChecksumMismatchError):
            fixture_manager.load_fixture("test-fixture", validate_checksum=True)

    def test_skips_checksum_validation_when_disabled(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Skips checksum validation when validate_checksum=False."""
        fixture_dir = create_test_fixture("test-fixture")

        # Corrupt the .DAT file
        (fixture_dir / "data.dat").write_bytes(b"corrupted data")

        # Should succeed with validation disabled
        result = fixture_manager.load_fixture("test-fixture", validate_checksum=False)

        assert result.success  # No checksum error

    def test_returns_accurate_timing_statistics(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Returns accurate timing statistics."""
        create_test_fixture("test-fixture")

        start = time.time()
        result = fixture_manager.load_fixture("test-fixture")
        elapsed = time.time() - start

        # Load time should be within reasonable range of actual elapsed time
        assert result.load_time_seconds > 0
        assert result.load_time_seconds <= elapsed + 0.1  # Small overhead allowance

    def test_raises_fixture_not_found_error(self, fixture_manager):
        """✅ Raises FixtureNotFoundError for missing fixtures."""
        from tests.fixtures.manager import FixtureNotFoundError

        with pytest.raises(FixtureNotFoundError) as exc_info:
            fixture_manager.load_fixture("non-existent-fixture")

        assert "non-existent-fixture" in str(exc_info.value)


# ==============================================================================
# SCAN_FIXTURES TESTS
# ==============================================================================


@pytest.mark.contract
class TestScanFixtures:
    """Contract tests for FixtureManager.scan_fixtures()."""

    def test_discovers_dat_fixtures(self, fixture_manager, create_test_fixture):
        """✅ Discovers .DAT fixtures in tests/fixtures/dat/."""
        create_test_fixture("fixture-a")
        create_test_fixture("fixture-b")

        manifest = fixture_manager.scan_fixtures()

        assert "fixture-a" in manifest.fixtures
        assert "fixture-b" in manifest.fixtures

    def test_loads_manifest_json_for_each_fixture(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Loads manifest.json for each fixture."""
        create_test_fixture("test-fixture")

        manifest = fixture_manager.scan_fixtures()

        fixture_metadata = manifest.fixtures["test-fixture"]
        assert fixture_metadata.name == "test-fixture"
        assert fixture_metadata.version == "1.0.0"
        assert fixture_metadata.description == "Test fixture for contract tests"

    def test_caches_results_and_skips_rescan(self, fixture_manager, create_test_fixture):
        """✅ Caches results and skips rescan when rescan=False."""
        create_test_fixture("test-fixture")

        # First scan
        manifest1 = fixture_manager.scan_fixtures()

        # Add another fixture
        create_test_fixture("new-fixture")

        # Second scan with rescan=False should NOT see new fixture
        manifest2 = fixture_manager.scan_fixtures(rescan=False)

        assert "test-fixture" in manifest2.fixtures
        assert "new-fixture" not in manifest2.fixtures  # Not rescanned

    def test_forces_rescan_when_rescan_true(self, fixture_manager, create_test_fixture):
        """✅ Forces rescan when rescan=True."""
        create_test_fixture("test-fixture")

        # First scan
        fixture_manager.scan_fixtures()

        # Add another fixture
        create_test_fixture("new-fixture")

        # Force rescan
        manifest = fixture_manager.scan_fixtures(rescan=True)

        assert "test-fixture" in manifest.fixtures
        assert "new-fixture" in manifest.fixtures  # Should be discovered


# ==============================================================================
# LIST_FIXTURES TESTS
# ==============================================================================


@pytest.mark.contract
class TestListFixtures:
    """Contract tests for FixtureManager.list_fixtures()."""

    def test_returns_all_fixtures_when_no_filter(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Returns all fixtures when filter_by=None."""
        create_test_fixture("fixture-a")
        create_test_fixture("fixture-b")

        fixture_manager.scan_fixtures()
        fixtures = fixture_manager.list_fixtures()

        assert len(fixtures) == 2

    def test_filters_by_source_type(self, fixture_manager, create_test_fixture):
        """✅ Filters by source_type='dat'."""
        create_test_fixture("dat-fixture")

        fixture_manager.scan_fixtures()
        fixtures = fixture_manager.list_fixtures(filter_by={"source_type": "dat"})

        assert len(fixtures) == 1
        assert fixtures[0].name == "dat-fixture"

    def test_filters_by_requires_embeddings(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Filters by requires_embeddings=True."""
        create_test_fixture("embedded-fixture")

        fixture_manager.scan_fixtures()
        fixtures = fixture_manager.list_fixtures(
            filter_by={"requires_embeddings": True}
        )

        assert len(fixtures) == 1
        assert fixtures[0].requires_embeddings is True

    def test_returns_empty_list_for_no_matches(self, fixture_manager):
        """✅ Returns empty list for no matches."""
        fixtures = fixture_manager.list_fixtures(filter_by={"source_type": "json"})

        assert fixtures == []


# ==============================================================================
# GET_FIXTURE TESTS
# ==============================================================================


@pytest.mark.contract
class TestGetFixture:
    """Contract tests for FixtureManager.get_fixture()."""

    def test_returns_latest_version_when_none_specified(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Returns latest version when version=None."""
        create_test_fixture("test-fixture")

        fixture_manager.scan_fixtures()
        metadata = fixture_manager.get_fixture("test-fixture", version=None)

        assert metadata is not None
        assert metadata.version == "1.0.0"

    def test_returns_exact_version_when_specified(
        self, fixture_manager, create_test_fixture
    ):
        """✅ Returns exact version when specified."""
        create_test_fixture("test-fixture")

        fixture_manager.scan_fixtures()
        metadata = fixture_manager.get_fixture("test-fixture", version="1.0.0")

        assert metadata is not None
        assert metadata.version == "1.0.0"

    def test_returns_none_for_nonexistent_fixture(self, fixture_manager):
        """✅ Returns None for non-existent fixture."""
        metadata = fixture_manager.get_fixture("does-not-exist")

        assert metadata is None


# ==============================================================================
# CLEANUP_FIXTURE_DATA TESTS
# ==============================================================================


@pytest.mark.contract
class TestCleanupFixtureData:
    """Contract tests for FixtureManager.cleanup_fixture_data()."""

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_deletes_data_from_single_table(self, fixture_manager):
        """✅ Deletes data from single table."""
        # This test requires actual IRIS database
        # Implementation will be in integration tests
        pass

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_returns_accurate_row_count(self, fixture_manager):
        """✅ Returns accurate row count."""
        # This test requires actual IRIS database
        # Implementation will be in integration tests
        pass


# ==============================================================================
# EXCEPTION HIERARCHY TESTS
# ==============================================================================


@pytest.mark.contract
class TestExceptionHierarchy:
    """Contract tests for fixture exceptions."""

    def test_fixture_not_found_error_inherits_from_fixture_error(self):
        """✅ FixtureNotFoundError inherits from FixtureError."""
        from tests.fixtures.manager import FixtureError, FixtureNotFoundError

        assert issubclass(FixtureNotFoundError, FixtureError)

    def test_checksum_mismatch_error_inherits_from_fixture_error(self):
        """✅ ChecksumMismatchError inherits from FixtureError."""
        from tests.fixtures.manager import FixtureError, ChecksumMismatchError

        assert issubclass(ChecksumMismatchError, FixtureError)

    def test_fixture_not_found_error_has_clear_message(self):
        """✅ FixtureNotFoundError has clear message."""
        from tests.fixtures.manager import FixtureNotFoundError

        error = FixtureNotFoundError("my-fixture")

        assert "my-fixture" in str(error)

    def test_checksum_mismatch_error_shows_expected_vs_actual(self):
        """✅ ChecksumMismatchError shows expected vs actual."""
        from tests.fixtures.manager import ChecksumMismatchError

        error = ChecksumMismatchError(
            fixture_name="test-fixture",
            expected="sha256:abc123",
            actual="sha256:def456",
        )

        assert "test-fixture" in str(error)
        assert "sha256:abc123" in str(error)
        assert "sha256:def456" in str(error)


# ==============================================================================
# RETURN VALUE CONTRACT TESTS
# ==============================================================================


@pytest.mark.contract
class TestFixtureLoadResult:
    """Contract tests for FixtureLoadResult return value."""

    def test_contains_all_required_fields(self, fixture_manager, create_test_fixture):
        """✅ Contains all required fields."""
        create_test_fixture("test-fixture")

        result = fixture_manager.load_fixture("test-fixture")

        # All required fields
        assert hasattr(result, "fixture_name")
        assert hasattr(result, "fixture_version")
        assert hasattr(result, "load_time_seconds")
        assert hasattr(result, "rows_loaded")
        assert hasattr(result, "checksum_valid")
        assert hasattr(result, "tables_loaded")
        assert hasattr(result, "tables_failed")
        assert hasattr(result, "success")
        assert hasattr(result, "error_message")

    def test_success_true_when_load_succeeds(
        self, fixture_manager, create_test_fixture
    ):
        """✅ success=True when load succeeds."""
        create_test_fixture("test-fixture")

        result = fixture_manager.load_fixture("test-fixture")

        assert result.success is True

    def test_error_message_none_on_success(self, fixture_manager, create_test_fixture):
        """✅ error_message=None on success."""
        create_test_fixture("test-fixture")

        result = fixture_manager.load_fixture("test-fixture")

        assert result.error_message is None

    def test_load_time_greater_than_zero(self, fixture_manager, create_test_fixture):
        """✅ load_time_seconds > 0."""
        create_test_fixture("test-fixture")

        result = fixture_manager.load_fixture("test-fixture")

        assert result.load_time_seconds > 0


# ==============================================================================
# PERFORMANCE CONTRACT TESTS
# ==============================================================================


@pytest.mark.contract
@pytest.mark.performance
class TestPerformanceContracts:
    """Contract tests for performance requirements."""

    @pytest.mark.skip(reason="Requires actual .DAT fixture loading")
    def test_dat_loads_faster_than_json(self):
        """✅ .DAT loads faster than JSON (same data)."""
        # Will be implemented in integration tests with real fixtures
        pass

    @pytest.mark.skip(reason="Requires actual .DAT fixture")
    def test_small_fixtures_load_in_under_2_seconds(self):
        """✅ Small fixtures (< 100 rows) load in < 2 seconds."""
        # Will be implemented in integration tests
        pass


# ==============================================================================
# BACKWARD COMPATIBILITY TESTS
# ==============================================================================


@pytest.mark.contract
class TestBackwardCompatibility:
    """Contract tests for backward compatibility."""

    @pytest.mark.skip(reason="Requires existing JSON fixture support")
    def test_loads_existing_json_fixtures(self):
        """✅ Loads existing JSON fixtures."""
        # Will be implemented when JSON fixture support is added
        pass
