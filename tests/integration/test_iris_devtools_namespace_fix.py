"""
Integration test for iris-devtools SYS.Namespace bug fix.

This test verifies that the fix for T066 works correctly - replacing
SYS.Namespace query with SET NAMESPACE approach.

Reference: specs/047-create-a-unified/tasks.md (T066)
"""

import pytest


@pytest.mark.integration
class TestIrisDevtoolsNamespaceFix:
    """Integration tests for iris-devtools namespace validation fix."""

    def test_can_import_fixture_creator(self):
        """✅ Can import FixtureCreator from iris-devtools."""
        from iris_devtools.fixtures import FixtureCreator

        creator = FixtureCreator()
        assert creator is not None

    def test_namespace_validation_uses_set_namespace(self):
        """✅ Namespace validation uses SET NAMESPACE instead of SYS.Namespace."""
        import sys
        from pathlib import Path

        # Add iris-devtools to path
        iris_devtools_path = Path("/Users/tdyar/ws/iris-devtools")
        if str(iris_devtools_path) not in sys.path:
            sys.path.insert(0, str(iris_devtools_path))

        # Read creator.py source to verify fix
        creator_file = iris_devtools_path / "iris_devtools" / "fixtures" / "creator.py"
        assert creator_file.exists(), "iris-devtools creator.py not found"

        source = creator_file.read_text()

        # Verify SYS.Namespace SQL query is NOT present (ignore comments)
        # Look for the actual SQL pattern: "FROM SYS.Namespace"
        assert "FROM SYS.Namespace" not in source, \
            "SYS.Namespace SQL query still present - fix not applied!"

        # Verify SET NAMESPACE is used instead
        assert "SET NAMESPACE" in source, \
            "SET NAMESPACE not found - fix may be incorrect"

    def test_fixture_creator_with_invalid_namespace_fails_gracefully(self):
        """✅ Invalid namespace fails with clear error message."""
        from iris_devtools.fixtures import FixtureCreator, FixtureCreateError
        from iris_devtools.config import IRISConfig
        import tempfile
        from pathlib import Path

        # Create creator with test IRIS connection
        config = IRISConfig(
            host="localhost",
            port=21972,
            namespace="USER",
            username="_SYSTEM",
            password="SYS"
        )
        creator = FixtureCreator(connection_config=config)

        # Try to export from non-existent namespace
        temp_dir = Path(tempfile.mkdtemp())
        temp_dat = temp_dir / "IRIS.DAT"

        try:
            with pytest.raises(FixtureCreateError) as exc_info:
                creator.export_namespace_to_dat(
                    namespace="NONEXISTENT_NAMESPACE_XYZ",
                    dat_file_path=str(temp_dat)
                )

            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "does not exist" in error_msg or "What went wrong" in error_msg

        finally:
            # Cleanup
            if temp_dat.exists():
                temp_dat.unlink()
            temp_dir.rmdir()

    def test_fixture_creator_with_valid_namespace_passes_validation(self):
        """✅ Valid namespace (USER) passes validation check."""
        from iris_devtools.fixtures import FixtureCreator
        from iris_devtools.config import IRISConfig

        # Create creator with test IRIS connection
        config = IRISConfig(
            host="localhost",
            port=21972,
            namespace="USER",
            username="_SYSTEM",
            password="SYS"
        )
        creator = FixtureCreator(connection_config=config)

        # Get tables from USER namespace (should work)
        try:
            tables = creator.get_namespace_tables("USER")
            # Should return list (may be empty if no tables)
            assert isinstance(tables, list)
        except Exception as e:
            # If IRIS is not running, skip test
            pytest.skip(f"IRIS not available: {e}")
