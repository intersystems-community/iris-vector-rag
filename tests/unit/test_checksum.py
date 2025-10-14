"""
Unit tests for checksum validation.

These tests verify checksum computation, validation, and error handling
in the fixture infrastructure.

Reference: specs/047-create-a-unified/tasks.md (T100)
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import hashlib


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.mark.unit
class TestChecksumComputation:
    """Unit tests for checksum computation."""

    def test_computes_sha256_for_file(self, temp_dir):
        """Checksum computation uses SHA256 algorithm."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Compute checksum
        checksum = manager._compute_checksum(test_file)

        # Should use sha256 prefix
        assert checksum.startswith("sha256:")

        # Verify actual SHA256
        expected_hash = hashlib.sha256(b"test content").hexdigest()
        assert checksum == f"sha256:{expected_hash}"

    def test_checksum_is_deterministic(self, temp_dir):
        """Checksum computation is deterministic."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        test_file = temp_dir / "test.txt"
        test_file.write_text("same content every time")

        # Compute multiple times
        checksum1 = manager._compute_checksum(test_file)
        checksum2 = manager._compute_checksum(test_file)
        checksum3 = manager._compute_checksum(test_file)

        assert checksum1 == checksum2 == checksum3

    def test_different_content_produces_different_checksum(self, temp_dir):
        """Different file content produces different checksums."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        file1 = temp_dir / "file1.txt"
        file1.write_text("content A")

        file2 = temp_dir / "file2.txt"
        file2.write_text("content B")

        checksum1 = manager._compute_checksum(file1)
        checksum2 = manager._compute_checksum(file2)

        assert checksum1 != checksum2

    def test_handles_empty_file(self, temp_dir):
        """Checksum computation handles empty files."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        checksum = manager._compute_checksum(empty_file)

        # Should compute checksum for empty file
        assert checksum.startswith("sha256:")
        empty_hash = hashlib.sha256(b"").hexdigest()
        assert checksum == f"sha256:{empty_hash}"

    def test_handles_large_file_efficiently(self, temp_dir):
        """Checksum computation handles large files efficiently (chunks)."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        # Create a large file (1 MB)
        large_file = temp_dir / "large.dat"
        content = b"x" * (1024 * 1024)  # 1 MB of 'x'
        large_file.write_bytes(content)

        # Should compute without loading entire file into memory
        checksum = manager._compute_checksum(large_file)

        # Verify correctness
        expected_hash = hashlib.sha256(content).hexdigest()
        assert checksum == f"sha256:{expected_hash}"

    def test_handles_binary_content(self, temp_dir):
        """Checksum computation handles binary files."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        binary_file = temp_dir / "binary.dat"
        binary_content = bytes(range(256))
        binary_file.write_bytes(binary_content)

        checksum = manager._compute_checksum(binary_file)

        # Verify correctness for binary data
        expected_hash = hashlib.sha256(binary_content).hexdigest()
        assert checksum == f"sha256:{expected_hash}"


@pytest.mark.unit
class TestChecksumValidation:
    """Unit tests for checksum validation."""

    def test_validate_checksum_accepts_matching_checksum(self, temp_dir):
        """_validate_checksum accepts file with matching checksum."""
        from tests.fixtures.manager import FixtureManager
        from tests.fixtures.models import FixtureMetadata

        manager = FixtureManager()

        # Create .DAT file
        fixture_dir = temp_dir / "fixture"
        fixture_dir.mkdir()

        dat_file = fixture_dir / "data.dat"
        dat_file.write_text("test data")

        # Compute correct checksum
        correct_checksum = manager._compute_checksum(dat_file)

        metadata = FixtureMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum=correct_checksum,
            schema_version="1.0",
            migration_history=[],
        )

        # Should not raise exception
        manager._validate_checksum(fixture_dir, metadata)

    def test_validate_checksum_rejects_mismatched_checksum(self, temp_dir):
        """_validate_checksum rejects file with mismatched checksum."""
        from tests.fixtures.manager import FixtureManager, ChecksumMismatchError
        from tests.fixtures.models import FixtureMetadata

        manager = FixtureManager()

        fixture_dir = temp_dir / "fixture"
        fixture_dir.mkdir()

        dat_file = fixture_dir / "data.dat"
        dat_file.write_text("actual data")

        metadata = FixtureMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum="sha256:wrongchecksum",
            schema_version="1.0",
            migration_history=[],
        )

        with pytest.raises(ChecksumMismatchError) as exc_info:
            manager._validate_checksum(fixture_dir, metadata)

        # Error should include expected and actual checksums
        assert "sha256:wrongchecksum" in str(exc_info.value)

    def test_validate_checksum_handles_missing_dat_file(self, temp_dir):
        """_validate_checksum handles missing .DAT file."""
        from tests.fixtures.manager import FixtureManager, FixtureLoadError
        from tests.fixtures.models import FixtureMetadata

        manager = FixtureManager()

        # Empty fixture directory - no .DAT file
        fixture_dir = temp_dir / "fixture"
        fixture_dir.mkdir()

        metadata = FixtureMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum="sha256:anything",
            schema_version="1.0",
            migration_history=[],
        )

        with pytest.raises(FixtureLoadError, match="No .DAT file found"):
            manager._validate_checksum(fixture_dir, metadata)

    def test_validate_checksum_finds_dat_files_with_different_names(self, temp_dir):
        """_validate_checksum finds .dat files even if not named data.dat."""
        from tests.fixtures.manager import FixtureManager
        from tests.fixtures.models import FixtureMetadata

        manager = FixtureManager()

        fixture_dir = temp_dir / "fixture"
        fixture_dir.mkdir()

        # Create .dat file with non-standard name (lowercase)
        dat_file = fixture_dir / "IRIS.dat"
        dat_file.write_text("test data")

        correct_checksum = manager._compute_checksum(dat_file)

        metadata = FixtureMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            created_at="2025-01-14T00:00:00Z",
            created_by="test",
            source_type="dat",
            tables=[],
            row_counts={},
            checksum=correct_checksum,
            schema_version="1.0",
            migration_history=[],
        )

        # Should find IRIS.dat and validate successfully
        manager._validate_checksum(fixture_dir, metadata)

    def test_checksum_mismatch_error_has_helpful_message(self, temp_dir):
        """ChecksumMismatchError includes helpful error message."""
        from tests.fixtures.manager import ChecksumMismatchError

        error = ChecksumMismatchError(
            fixture_name="test-fixture",
            expected="sha256:expected123",
            actual="sha256:actual456"
        )

        error_msg = str(error)

        # Should include fixture name
        assert "test-fixture" in error_msg

        # Should include both checksums
        assert "sha256:expected123" in error_msg
        assert "sha256:actual456" in error_msg

        # Should indicate mismatch
        assert "mismatch" in error_msg.lower()


@pytest.mark.unit
class TestChecksumSkipping:
    """Unit tests for skipping checksum validation."""

    def test_load_fixture_can_skip_checksum_validation(self):
        """load_fixture allows skipping checksum validation."""
        from tests.fixtures.manager import FixtureManager
        import inspect

        manager = FixtureManager()

        # Check method signature
        sig = inspect.signature(manager.load_fixture)

        assert 'validate_checksum' in sig.parameters
        assert sig.parameters['validate_checksum'].default is True


@pytest.mark.unit
class TestChecksumFormat:
    """Unit tests for checksum format validation."""

    def test_checksum_format_is_sha256_prefix_plus_hex(self, temp_dir):
        """Checksum format is 'sha256:' followed by hexadecimal digest."""
        from tests.fixtures.manager import FixtureManager

        manager = FixtureManager()

        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        checksum = manager._compute_checksum(test_file)

        # Should have sha256 prefix
        assert checksum.startswith("sha256:")

        # Remaining part should be hex
        hex_part = checksum.split(":", 1)[1]
        assert all(c in "0123456789abcdef" for c in hex_part)

        # SHA256 produces 64 hex characters
        assert len(hex_part) == 64
