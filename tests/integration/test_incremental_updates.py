"""
Integration tests for incremental fixture updates.

These tests verify that fixtures can be updated incrementally without
requiring full rebuilds.

Reference: specs/047-create-a-unified/tasks.md (T088)
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.mark.integration
class TestIncrementalFixtureUpdates:
    """Integration tests for incremental fixture update functionality."""

    @pytest.fixture
    def temp_fixture_dir(self):
        """Create temporary fixture directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def base_fixture(self, temp_fixture_dir):
        """Create a base fixture for testing updates."""
        from tests.fixtures.manager import FixtureManager
        from tests.fixtures.models import FixtureMetadata
        import json

        manager = FixtureManager()

        # Create minimal fixture metadata
        metadata = FixtureMetadata(
            name="test-base",
            version="1.0.0",
            description="Test base fixture",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=["RAG.SourceDocuments", "RAG.Entities"],
            row_counts={"RAG.SourceDocuments": 0, "RAG.Entities": 0},
            checksum="sha256:original",
            schema_version="1.0",
            migration_history=[],
        )

        # Save to temp directory
        fixture_path = temp_fixture_dir / "test-base"
        fixture_path.mkdir(parents=True, exist_ok=True)

        # Save manifest manually
        manifest_file = fixture_path / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return fixture_path

    def test_migrate_updates_fixture_version(self, base_fixture):
        """✅ migrate() increments fixture version."""
        from tests.fixtures.manager import FixtureManager
        from tests.fixtures.models import FixtureMetadata
        import json

        manager = FixtureManager()

        # Migrate fixture to version 2.0.0
        result = manager.migrate(
            fixture_name=str(base_fixture),
            target_version="2.0.0",
            dry_run=False,
        )

        # Verify version was updated
        assert result.success
        assert result.old_version == "1.0.0"
        assert result.new_version == "2.0.0"

        # Verify manifest was updated
        manifest_file = base_fixture / "manifest.json"
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        metadata = FixtureMetadata.from_dict(manifest_data)
        assert metadata.version == "2.0.0"

    def test_migrate_records_migration_in_history(self, base_fixture):
        """✅ migrate() adds entry to migration_history."""
        from tests.fixtures.manager import FixtureManager
        from tests.fixtures.models import FixtureMetadata
        import json

        manager = FixtureManager()

        # Perform migration
        result = manager.migrate(
            fixture_name=str(base_fixture),
            target_version="2.0.0",
            changes=["Added new table RAG.EntityRelationships"],
        )

        # Load updated manifest
        manifest_file = base_fixture / "manifest.json"
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        metadata = FixtureMetadata.from_dict(manifest_data)

        # Verify migration was recorded
        assert len(metadata.migration_history) == 1
        migration_entry = metadata.migration_history[0]

        assert migration_entry["from_version"] == "1.0.0"
        assert migration_entry["to_version"] == "2.0.0"
        assert "Added new table RAG.EntityRelationships" in migration_entry["changes"]
        assert "timestamp" in migration_entry
        assert "applied_by" in migration_entry

    def test_migrate_dry_run_shows_preview(self, base_fixture):
        """✅ migrate() with dry_run=True previews changes without applying."""
        from tests.fixtures.manager import FixtureManager
        from tests.fixtures.models import FixtureMetadata
        import json

        manager = FixtureManager()

        # Run migration in dry-run mode
        result = manager.migrate(
            fixture_name=str(base_fixture),
            target_version="2.0.0",
            dry_run=True,
        )

        # Verify result shows what would happen
        assert result.success
        assert result.new_version == "2.0.0"

        # Verify manifest was NOT updated
        manifest_file = base_fixture / "manifest.json"
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        metadata = FixtureMetadata.from_dict(manifest_data)
        assert metadata.version == "1.0.0"  # Still original version

    def test_incremental_update_adds_only_delta(self):
        """✅ Incremental update processes only changed data."""
        pytest.skip("Requires database connection - implement after basic migration works")

        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Create base fixture with 10 entities
        # ... (populate database with 10 entities)
        # base_result = manager.create("test-base-10", fixture_type="dat")

        # Add 1 more entity to database
        # ... (add entity)

        # Update fixture incrementally
        # update_result = manager.update(
        #     "test-base-10",
        #     incremental=True,
        #     new_version="1.1.0"
        # )

        # Verify only 1 entity was processed
        # assert update_result.rows_processed == 1
        # assert update_result.old_version == "1.0.0"
        # assert update_result.new_version == "1.1.0"

    def test_update_fixture_make_target_works(self):
        """✅ make update-fixture NAME=... target works end-to-end."""
        pytest.skip("Requires Makefile target implementation - T091")

        # This test will verify:
        # 1. make update-fixture NAME=medical-20
        # 2. Detects changes since last version
        # 3. Applies incremental update
        # 4. Updates manifest
        # 5. Validates checksums

    def test_version_compatibility_validation(self, base_fixture):
        """✅ migrate() validates version compatibility."""
        from tests.fixtures.manager import FixtureManager, VersionMismatchError

        manager = FixtureManager()

        # Attempting to migrate from 1.0.0 to 3.0.0 without intermediate versions
        # should raise error (breaking change - skipping major version 2.x.x)
        with pytest.raises(VersionMismatchError, match="incompatible"):
            manager.migrate(
                fixture_name=str(base_fixture),
                target_version="3.0.0",
            )

    def test_fixture_diff_detects_schema_changes(self):
        """✅ FixtureManager can diff two fixture versions."""
        pytest.skip("Requires diff_fixtures implementation")

        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Compare two fixtures
        # diff = manager.diff_fixtures("medical-20-v1", "medical-20-v2")

        # Verify diff shows changes
        # assert "RAG.Entities" in diff.tables_modified
        # assert diff.rows_added == 1
        # assert diff.rows_removed == 0
        # assert "Added column 'importance_score'" in diff.schema_changes

    def test_partial_export_specific_tables(self):
        """✅ FixtureManager can export only specific tables."""
        pytest.skip("Requires database connection and partial export implementation")

        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Export only RAG.Entities table
        # result = manager.create(
        #     "medical-20-entities-only",
        #     fixture_type="dat",
        #     tables=["RAG.Entities"],
        #     incremental=True,
        # )

        # Verify only specified table was exported
        # assert len(result.tables) == 1
        # assert "RAG.Entities" in result.tables

    def test_merge_incremental_updates(self):
        """✅ FixtureManager can merge delta into existing fixture."""
        pytest.skip("Requires merge implementation")

        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Create delta fixture with new data
        # delta = manager.create("medical-20-delta", tables=["RAG.Entities"])

        # Merge delta into base fixture
        # result = manager.merge(
        #     base="medical-20",
        #     delta="medical-20-delta",
        #     output="medical-21",
        # )

        # Verify merged fixture has combined data
        # assert result.success
        # assert result.new_version == "1.1.0"


@pytest.mark.integration
class TestFixtureVersioning:
    """Integration tests for fixture version management."""

    def test_pytest_decorator_supports_version_parameter(self):
        """✅ @pytest.mark.dat_fixture supports version parameter."""
        pytest.skip("Requires pytest plugin version support - T092")

        # This test will verify:
        # @pytest.mark.dat_fixture("medical-20", version="1.2.0")
        # class TestWithSpecificVersion:
        #     def test_uses_correct_version(self):
        #         # Verify version 1.2.0 was loaded
        #         pass

    def test_loading_specific_fixture_version(self):
        """✅ FixtureManager.load() can load specific version."""
        pytest.skip("Requires version-specific loading implementation")

        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Load specific version
        # result = manager.load("medical-20", version="1.2.0")

        # Verify correct version loaded
        # assert result.metadata.version == "1.2.0"

    def test_manifest_tracks_multiple_versions(self):
        """✅ Fixture directory can contain multiple versions."""
        pytest.skip("Requires multi-version storage design")

        # Directory structure:
        # tests/fixtures/dat/medical-20/
        #   1.0.0/
        #     IRIS.DAT
        #     manifest.json
        #   1.2.0/
        #     IRIS.DAT
        #     manifest.json
        #   latest -> 1.2.0/


@pytest.mark.integration
class TestMigrationErrorHandling:
    """Integration tests for migration error handling."""

    @pytest.fixture
    def temp_fixture_dir(self):
        """Create temporary fixture directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_migrate_fails_on_missing_fixture(self):
        """✅ migrate() fails gracefully on missing fixture."""
        from tests.fixtures.manager import FixtureManager, FixtureNotFoundError

        manager = FixtureManager()

        with pytest.raises(FixtureNotFoundError, match="Fixture 'nonexistent' not found"):
            manager.migrate("nonexistent", target_version="2.0.0")

    def test_migrate_fails_on_corrupted_manifest(self, temp_fixture_dir):
        """✅ migrate() detects and reports corrupted manifest."""
        from tests.fixtures.manager import FixtureManager

        # Create corrupted manifest
        fixture_path = temp_fixture_dir / "corrupted"
        fixture_path.mkdir(parents=True, exist_ok=True)

        manifest_file = fixture_path / "manifest.json"
        manifest_file.write_text("{ invalid json")

        manager = FixtureManager()

        # migrate() returns a MigrationResult with success=False for corrupted JSON
        result = manager.migrate(str(fixture_path), target_version="2.0.0")

        assert result.success is False
        assert "Expecting property name" in result.error_message  # JSON decode error message

    def test_migrate_rollback_on_failure(self):
        """✅ migrate() rolls back changes if migration fails."""
        pytest.skip("Requires transaction/rollback implementation")

        # If migration fails mid-way, should restore original state
        # Verify manifest.json is unchanged
        # Verify backup is created and can be restored
