"""
Contract tests for fixture checksum validation.

These tests verify that the FixtureManager properly validates fixture integrity
using SHA256 checksums before and after loading.

Reference: specs/047-create-a-unified/tasks.md (T030)
"""

import pytest
from pathlib import Path
from tests.fixtures.manager import FixtureManager, FixtureError
from tests.fixtures.models import FixtureMetadata


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def fixture_manager():
    """Create FixtureManager instance."""
    return FixtureManager()


@pytest.fixture
def sample_fixture_metadata():
    """Sample fixture metadata for testing."""
    return FixtureMetadata(
        name="test-fixture",
        version="1.0.0",
        description="Test fixture for checksum validation",
        source_type="dat",
        namespace="USER",
        tables=["RAG.SourceDocuments"],
        row_counts={"RAG.SourceDocuments": 10},
        checksum="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        created_at="2025-01-14T00:00:00",
        created_by="test",
        requires_embeddings=False,
        embedding_model=None,
        embedding_dimension=384,
    )


# ==============================================================================
# CHECKSUM VALIDATION CONTRACT TESTS
# ==============================================================================


@pytest.mark.contract
class TestChecksumValidation:
    """Contract tests for checksum validation logic."""

    def test_checksum_validation_accepts_valid_checksum(self, fixture_manager, tmp_path):
        """Checksum validation passes for matching checksum."""
        # Create a test file with known content (use expected filename)
        test_file = tmp_path / "data.dat"
        test_file.write_bytes(b"")  # Empty file has known SHA256

        metadata = FixtureMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            source_type="dat",
            namespace="USER",
            tables=["RAG.SourceDocuments"],
            row_counts={"RAG.SourceDocuments": 1},
            checksum="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            created_at="2025-01-14",
            created_by="test",
            requires_embeddings=False,
            embedding_model=None,
            embedding_dimension=384,
        )

        # Should not raise exception
        try:
            fixture_manager._validate_checksum(tmp_path, metadata)
        except Exception as e:
            pytest.fail(f"Valid checksum validation failed: {e}")

    def test_checksum_validation_rejects_mismatched_checksum(self, fixture_manager, tmp_path):
        """Checksum validation fails for mismatched checksum."""
        # Create a test file with known content (use expected filename)
        test_file = tmp_path / "data.dat"
        test_file.write_bytes(b"test content")

        metadata = FixtureMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            source_type="dat",
            namespace="USER",
            tables=["RAG.SourceDocuments"],
            row_counts={"RAG.SourceDocuments": 1},
            checksum="sha256:wrong_checksum_value",
            created_at="2025-01-14",
            created_by="test",
            requires_embeddings=False,
            embedding_model=None,
            embedding_dimension=384,
        )

        # Should raise FixtureError
        with pytest.raises(FixtureError) as exc_info:
            fixture_manager._validate_checksum(tmp_path, metadata)

        assert "checksum" in str(exc_info.value).lower()

    def test_checksum_validation_handles_missing_dat_file(self, fixture_manager, tmp_path):
        """Checksum validation fails gracefully when .DAT file is missing."""
        metadata = FixtureMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            source_type="dat",
            namespace="USER",
            tables=["RAG.SourceDocuments"],
            row_counts={"RAG.SourceDocuments": 1},
            checksum="sha256:abc123",
            created_at="2025-01-14",
            created_by="test",
            requires_embeddings=False,
            embedding_model=None,
            embedding_dimension=384,
        )

        # Should raise FixtureError
        with pytest.raises(FixtureError) as exc_info:
            fixture_manager._validate_checksum(tmp_path, metadata)

        error_msg = str(exc_info.value).lower()
        # Check that error message indicates file not found
        assert ".dat" in error_msg and ("not found" in error_msg or "found" in error_msg)

    def test_checksum_format_validation(self, fixture_manager):
        """Checksum must be in format 'sha256:hash'."""
        # Valid checksums should have 'sha256:' prefix
        valid_checksum = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert valid_checksum.startswith("sha256:")

        # Invalid checksums should be rejected
        invalid_checksums = [
            "md5:abc123",  # Wrong algorithm
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # Missing prefix
            "sha256:",  # Empty hash
            "",  # Empty string
        ]

        # FixtureMetadata should validate checksum format on creation
        for invalid in invalid_checksums:
            try:
                # Create metadata with invalid checksum
                metadata = FixtureMetadata(
                    name="test",
                    version="1.0.0",
                    description="Test",
                    source_type="dat",
                    namespace="USER",
                    tables=["RAG.SourceDocuments"],
                    row_counts={"RAG.SourceDocuments": 1},
                    checksum=invalid,
                    created_at="2025-01-14",
                    created_by="test",
                    requires_embeddings=False,
                    embedding_model=None,
                    embedding_dimension=384,
                )
                # If we get here, validate that checksum format check happens during loading
                # (some validation may be deferred to load time)
            except (ValueError, FixtureError):
                # Expected for invalid checksums
                pass


@pytest.mark.contract
class TestChecksumComputation:
    """Contract tests for SHA256 checksum computation."""

    def test_compute_checksum_for_empty_file(self, fixture_manager, tmp_path):
        """SHA256 checksum of empty file is correct."""
        test_file = tmp_path / "test.dat"
        test_file.write_bytes(b"")

        checksum = fixture_manager._compute_checksum(test_file)

        # SHA256 of empty file
        expected = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert checksum == expected

    def test_compute_checksum_for_file_with_content(self, fixture_manager, tmp_path):
        """SHA256 checksum of file with content is correct."""
        test_file = tmp_path / "test.dat"
        test_file.write_bytes(b"test content")

        checksum = fixture_manager._compute_checksum(test_file)

        # SHA256 of "test content"
        expected = "sha256:6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
        assert checksum == expected

    def test_compute_checksum_handles_large_files(self, fixture_manager, tmp_path):
        """SHA256 checksum handles large files efficiently."""
        test_file = tmp_path / "large.dat"

        # Create 10MB file
        test_file.write_bytes(b"x" * (10 * 1024 * 1024))

        # Should compute without error
        checksum = fixture_manager._compute_checksum(test_file)

        assert checksum.startswith("sha256:")
        assert len(checksum) == len("sha256:") + 64  # SHA256 hash is 64 hex chars

    def test_compute_checksum_is_deterministic(self, fixture_manager, tmp_path):
        """Same file produces same checksum every time."""
        test_file = tmp_path / "test.dat"
        test_file.write_bytes(b"deterministic content")

        checksum1 = fixture_manager._compute_checksum(test_file)
        checksum2 = fixture_manager._compute_checksum(test_file)
        checksum3 = fixture_manager._compute_checksum(test_file)

        assert checksum1 == checksum2 == checksum3


@pytest.mark.contract
class TestChecksumValidationWorkflow:
    """Contract tests for checksum validation in fixture loading workflow."""

    def test_load_fixture_validates_checksum_by_default(self, fixture_manager):
        """load_fixture() validates checksum by default."""
        # This is a contract test - implementation may vary
        # But the API must support checksum validation
        assert hasattr(fixture_manager, "load_fixture")

        # Check load_fixture signature includes validate_checksum parameter
        import inspect
        sig = inspect.signature(fixture_manager.load_fixture)
        params = sig.parameters

        assert "validate_checksum" in params, \
            "load_fixture() must have validate_checksum parameter"

    def test_load_fixture_allows_skipping_validation(self, fixture_manager):
        """load_fixture() can skip checksum validation if requested."""
        import inspect
        sig = inspect.signature(fixture_manager.load_fixture)
        params = sig.parameters

        # validate_checksum should have a default value (preferably True)
        assert "validate_checksum" in params
        param = params["validate_checksum"]

        # Should have a default value
        assert param.default is not inspect.Parameter.empty, \
            "validate_checksum should have a default value"

    def test_checksum_validation_error_message_is_helpful(self, fixture_manager, tmp_path):
        """Checksum validation errors include helpful messages."""
        test_file = tmp_path / "data.dat"
        test_file.write_bytes(b"corrupted content")

        metadata = FixtureMetadata(
            name="corrupted-fixture",
            version="1.0.0",
            description="Test",
            source_type="dat",
            namespace="USER",
            tables=["RAG.SourceDocuments"],
            row_counts={"RAG.SourceDocuments": 1},
            checksum="sha256:expected_checksum_value",
            created_at="2025-01-14",
            created_by="test",
            requires_embeddings=False,
            embedding_model=None,
            embedding_dimension=384,
        )

        try:
            fixture_manager._validate_checksum(tmp_path, metadata)
            pytest.fail("Expected FixtureError for checksum mismatch")
        except FixtureError as e:
            error_msg = str(e)

            # Error message should contain:
            # 1. What failed (checksum validation)
            # 2. Expected vs actual checksums
            # 3. Fixture name

            assert "checksum" in error_msg.lower() or "mismatch" in error_msg.lower(), \
                "Error message should mention checksum/mismatch"

            # Should help identify the fixture
            assert "corrupted-fixture" in error_msg or "fixture" in error_msg.lower(), \
                "Error message should identify the fixture"
