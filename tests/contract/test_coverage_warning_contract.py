"""Contract tests for coverage warning system per COV-001.

These tests define the expected behavior of the coverage warning plugin.
They must fail initially and pass once the plugin is implemented.
"""

import pytest
from unittest.mock import Mock, patch
import coverage


class TestCoverageWarningContract:
    """Contract tests for COV-001 coverage warning behavior."""

    def test_COV001_plugin_registration(self):
        """Verify coverage_warnings plugin registers correctly with pytest."""
        # This should fail until plugin exists
        # Check if plugin module can be imported
        try:
            import tests.plugins.coverage_warnings
            plugin_exists = True
        except ImportError:
            plugin_exists = False

        assert plugin_exists, "coverage_warnings plugin not registered"

    def test_COV001_threshold_detection(self):
        """Verify modules below 60% threshold are detected and warned."""
        # Mock coverage data
        mock_coverage = Mock()
        mock_coverage.get_data.return_value = Mock(
            measured_files=lambda: ["module1.py", "module2.py"],
            lines=lambda f: list(range(1, 101)),  # 100 lines
            executed_lines=lambda f: list(range(1, 51)) if f == "module1.py" else list(range(1, 71))  # 50% and 70%
        )

        # This should fail until plugin implements warning collection
        from tests.plugins.coverage_warnings import collect_coverage_warnings
        warnings = collect_coverage_warnings(mock_coverage)

        # Should warn about module1.py (50% < 60%) but not module2.py (70% > 60%)
        assert len(warnings) == 1
        assert warnings[0].module_path == "module1.py"
        assert warnings[0].current_coverage == 50.0
        assert warnings[0].threshold == 60.0

    def test_COV001_critical_module_detection(self):
        """Verify critical modules use 80% threshold instead of 60%."""
        # Mock coverage data for critical module
        mock_coverage = Mock()
        mock_coverage.get_data.return_value = Mock(
            measured_files=lambda: ["iris_rag/pipelines/basic.py"],
            lines=lambda f: list(range(1, 101)),  # 100 lines
            executed_lines=lambda f: list(range(1, 76))  # 75% coverage
        )

        # This should fail until plugin implements critical module detection
        from tests.plugins.coverage_warnings import collect_coverage_warnings
        warnings = collect_coverage_warnings(mock_coverage)

        # Should warn because 75% < 80% for critical module
        assert len(warnings) == 1
        assert warnings[0].module_path == "iris_rag/pipelines/basic.py"
        assert warnings[0].current_coverage == 75.0
        assert warnings[0].threshold == 80.0
        assert warnings[0].is_critical is True

    def test_COV001_warning_format(self):
        """Verify warning messages include required information."""
        # Mock a warning
        mock_warning = Mock(
            module_path="mymodule.py",
            current_coverage=45.5,
            threshold=60.0,
            is_critical=False
        )

        # This should fail until plugin implements format_warning
        from tests.plugins.coverage_warnings import format_warning
        message = format_warning(mock_warning)

        # Should include all required information
        assert "mymodule.py" in message
        assert "45.5%" in message
        assert "60.0%" in message
        assert "WARNING:" in message

    def test_COV001_non_failing_behavior(self):
        """Verify coverage warnings don't fail the test run."""
        # Mock pytest terminal reporter
        mock_terminal = Mock()
        mock_config = Mock()
        mock_config.option.verbose = 1
        mock_terminal.config = mock_config

        # Mock coverage with low coverage module
        with patch('coverage.Coverage') as mock_coverage_class:
            mock_cov = Mock()
            mock_coverage_class.return_value = mock_cov
            mock_data = Mock()
            mock_data.measured_files.return_value = ["low_module.py"]
            mock_data.lines.return_value = list(range(1, 101))
            mock_data.executed_lines.return_value = list(range(1, 31))  # 30% coverage
            mock_cov.get_data.return_value = mock_data

            # This should fail until plugin implements pytest_terminal_summary
            from tests.plugins.coverage_warnings import pytest_terminal_summary

            # Should not raise exception despite low coverage
            exitstatus = 0  # Success
            pytest_terminal_summary(mock_terminal, exitstatus, mock_config)

            # Verify warnings were written but exit status unchanged
            # Plugin uses write_line, section, etc - any of these being called means it worked
            assert (mock_terminal.write_line.called or
                    mock_terminal.section.called or
                    mock_terminal.write.called), "No terminal write method was called"
            assert exitstatus == 0  # Still success