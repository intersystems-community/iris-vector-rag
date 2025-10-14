"""
Unit tests for FixtureManager methods.

These tests focus on testing individual FixtureManager methods in isolation,
using mocks where appropriate to avoid database dependencies.

Reference: specs/047-create-a-unified/tasks.md (T098)
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import json


@pytest.fixture
def temp_fixtures_dir():
    """Create temporary fixtures directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def fixture_manager(temp_fixtures_dir):
    """Create FixtureManager instance with temporary directory."""
    from tests.fixtures.manager import FixtureManager
    return FixtureManager(fixtures_root=temp_fixtures_dir)


@pytest.mark.unit
class TestFixtureManagerInit:
    """Unit tests for FixtureManager initialization."""

    def test_creates_fixtures_root_if_missing(self, temp_fixtures_dir):
        """FixtureManager creates fixtures root directory if it doesn't exist."""
        from tests.fixtures.manager import FixtureManager

        # Delete the directory
        shutil.rmtree(temp_fixtures_dir)
        assert not temp_fixtures_dir.exists()

        # Initialize manager
        manager = FixtureManager(fixtures_root=temp_fixtures_dir)

        # Directory should be created
        assert temp_fixtures_dir.exists()

    def test_uses_default_fixtures_root(self):
        """FixtureManager uses default fixtures root when none provided."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Should default to tests/fixtures
        assert manager.fixtures_root.name == "fixtures"
        assert manager.fixtures_root.parent.name == "tests"

    def test_accepts_custom_backend_mode(self, temp_fixtures_dir):
        """FixtureManager accepts custom backend mode."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager(fixtures_root=temp_fixtures_dir, backend_mode="enterprise")

        assert manager.backend_mode == "enterprise"

    def test_defaults_to_community_backend_mode(self, temp_fixtures_dir):
        """FixtureManager defaults to community backend mode."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager(fixtures_root=temp_fixtures_dir)

        assert manager.backend_mode == "community"


@pytest.mark.unit
class TestScanFixtures:
    """Unit tests for scan_fixtures method."""

    def test_scans_dat_directory(self, fixture_manager, temp_fixtures_dir):
        """scan_fixtures discovers fixtures in dat/ directory."""
        from tests.fixtures.models import FixtureMetadata

        # Create a fixture directory with manifest
        dat_dir = temp_fixtures_dir / "dat"
        dat_dir.mkdir(parents=True, exist_ok=True)

        fixture_dir = dat_dir / "test-fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)

        metadata = FixtureMetadata(
            name="test-fixture",
            version="1.0.0",
            description="Test fixture",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=["RAG.SourceDocuments"],
            row_counts={"RAG.SourceDocuments": 10},
            checksum="sha256:abc123",
            schema_version="1.0",
            migration_history=[],
        )

        manifest_file = fixture_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Scan fixtures
        manifest = fixture_manager.scan_fixtures()

        # Should find the fixture
        assert len(manifest.fixtures) == 1
        assert "test-fixture" in manifest.fixtures

    def test_caches_scan_results(self, fixture_manager, temp_fixtures_dir):
        """scan_fixtures caches results and doesn't rescan unless requested."""
        # First scan
        manifest1 = fixture_manager.scan_fixtures()

        # Create a new fixture after first scan
        dat_dir = temp_fixtures_dir / "dat"
        dat_dir.mkdir(parents=True, exist_ok=True)

        fixture_dir = dat_dir / "new-fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)

        from tests.fixtures.models import FixtureMetadata
        metadata = FixtureMetadata(
            name="new-fixture",
            version="1.0.0",
            description="New fixture",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum="sha256:def456",
            schema_version="1.0",
            migration_history=[],
        )

        manifest_file = fixture_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Second scan without rescan=True should return cached results
        manifest2 = fixture_manager.scan_fixtures()
        assert len(manifest2.fixtures) == len(manifest1.fixtures)

        # Third scan with rescan=True should find new fixture
        manifest3 = fixture_manager.scan_fixtures(rescan=True)
        assert len(manifest3.fixtures) == len(manifest1.fixtures) + 1

    def test_skips_corrupted_manifests(self, fixture_manager, temp_fixtures_dir):
        """scan_fixtures skips fixtures with corrupted manifest.json."""
        dat_dir = temp_fixtures_dir / "dat"
        dat_dir.mkdir(parents=True, exist_ok=True)

        # Create fixture with corrupted manifest
        fixture_dir = dat_dir / "corrupted"
        fixture_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = fixture_dir / "manifest.json"
        manifest_file.write_text("{ invalid json")

        # Scan should not crash
        manifest = fixture_manager.scan_fixtures()

        # Should not include corrupted fixture
        assert "corrupted" not in manifest.fixtures


@pytest.mark.unit
class TestGetFixture:
    """Unit tests for get_fixture method."""

    def test_returns_none_for_nonexistent_fixture(self, fixture_manager):
        """get_fixture returns None for non-existent fixture."""
        result = fixture_manager.get_fixture("nonexistent")

        assert result is None

    def test_returns_fixture_when_exists(self, fixture_manager, temp_fixtures_dir):
        """get_fixture returns fixture metadata when it exists."""
        from tests.fixtures.models import FixtureMetadata

        # Create a fixture
        dat_dir = temp_fixtures_dir / "dat"
        dat_dir.mkdir(parents=True, exist_ok=True)

        fixture_dir = dat_dir / "test-fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)

        metadata = FixtureMetadata(
            name="test-fixture",
            version="1.0.0",
            description="Test",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum="sha256:abc",
            schema_version="1.0",
            migration_history=[],
        )

        manifest_file = fixture_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Get fixture
        result = fixture_manager.get_fixture("test-fixture")

        assert result is not None
        assert result.name == "test-fixture"
        assert result.version == "1.0.0"


@pytest.mark.unit
class TestListFixtures:
    """Unit tests for list_fixtures method."""

    def test_returns_all_fixtures_without_filter(self, fixture_manager, temp_fixtures_dir):
        """list_fixtures returns all fixtures when no filter provided."""
        from tests.fixtures.models import FixtureMetadata

        # Create two fixtures
        dat_dir = temp_fixtures_dir / "dat"
        dat_dir.mkdir(parents=True, exist_ok=True)

        for name in ["fixture1", "fixture2"]:
            fixture_dir = dat_dir / name
            fixture_dir.mkdir(parents=True, exist_ok=True)

            metadata = FixtureMetadata(
                name=name,
                version="1.0.0",
                description="Test",
                created_at="2025-01-14T00:00:00Z",
                created_by="test",
                source_type="dat",
                tables=[],
                row_counts={},
                checksum=f"sha256:{name}",
                schema_version="1.0",
                migration_history=[],
            )

            manifest_file = fixture_dir / "manifest.json"
            with open(manifest_file, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

        # List all fixtures
        fixtures = fixture_manager.list_fixtures()

        assert len(fixtures) == 2
        assert {f.name for f in fixtures} == {"fixture1", "fixture2"}

    def test_filters_by_source_type(self, fixture_manager, temp_fixtures_dir):
        """list_fixtures filters by source_type."""
        from tests.fixtures.models import FixtureMetadata

        # Create fixtures with different source types
        dat_dir = temp_fixtures_dir / "dat"
        dat_dir.mkdir(parents=True, exist_ok=True)

        fixture_dir = dat_dir / "dat-fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)

        metadata = FixtureMetadata(
            name="dat-fixture",
            version="1.0.0",
            description="DAT fixture",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum="sha256:dat",
            schema_version="1.0",
            migration_history=[],
        )

        manifest_file = fixture_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Filter by source_type
        fixtures = fixture_manager.list_fixtures(filter_by={"source_type": "dat"})

        assert len(fixtures) == 1
        assert fixtures[0].source_type == "dat"


@pytest.mark.unit
class TestComputeChecksum:
    """Unit tests for _compute_checksum method."""

    def test_computes_sha256_checksum(self, fixture_manager, temp_fixtures_dir):
        """_compute_checksum computes SHA256 checksum correctly."""
        # Create a test file
        test_file = temp_fixtures_dir / "test.txt"
        test_file.write_text("test content")

        # Compute checksum
        checksum = fixture_manager._compute_checksum(test_file)

        # Should be in format "sha256:hexdigest"
        assert checksum.startswith("sha256:")
        assert len(checksum) > 10

    def test_checksum_is_deterministic(self, fixture_manager, temp_fixtures_dir):
        """_compute_checksum returns same result for same file."""
        # Create a test file
        test_file = temp_fixtures_dir / "test.txt"
        test_file.write_text("test content")

        # Compute checksum twice
        checksum1 = fixture_manager._compute_checksum(test_file)
        checksum2 = fixture_manager._compute_checksum(test_file)

        assert checksum1 == checksum2

    def test_different_content_produces_different_checksum(self, fixture_manager, temp_fixtures_dir):
        """_compute_checksum produces different checksums for different content."""
        # Create two test files with different content
        file1 = temp_fixtures_dir / "test1.txt"
        file1.write_text("content 1")

        file2 = temp_fixtures_dir / "test2.txt"
        file2.write_text("content 2")

        # Compute checksums
        checksum1 = fixture_manager._compute_checksum(file1)
        checksum2 = fixture_manager._compute_checksum(file2)

        assert checksum1 != checksum2


@pytest.mark.unit
class TestGetFixturePath:
    """Unit tests for _get_fixture_path method."""

    def test_returns_dat_path_for_dat_fixture(self, fixture_manager):
        """_get_fixture_path returns correct path for DAT fixture."""
        from tests.fixtures.models import FixtureMetadata

        metadata = FixtureMetadata(
            name="test-dat",
            version="1.0.0",
            description="Test",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum="sha256:abc",
            schema_version="1.0",
            migration_history=[],
        )

        path = fixture_manager._get_fixture_path(metadata)

        assert path.name == "test-dat"
        assert path.parent.name == "dat"

    def test_returns_json_path_for_json_fixture(self, fixture_manager):
        """_get_fixture_path returns correct path for JSON fixture."""
        from tests.fixtures.models import FixtureMetadata

        metadata = FixtureMetadata(
            name="test-json",
            version="1.0.0",
            description="Test",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="json",
            tables=[],
            row_counts={},
            checksum="sha256:abc",
            schema_version="1.0",
            migration_history=[],
        )

        path = fixture_manager._get_fixture_path(metadata)

        assert path.name == "test-json"
        assert path.parent.name == "graphrag"
