"""Pytest plugin for coverage threshold warnings.

This plugin displays warnings when code coverage falls below configured thresholds
without failing the test run. Critical modules have higher thresholds.
"""

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import coverage


@dataclass
class CoverageWarning:
    """Represents a coverage threshold violation."""

    module_path: str
    current_coverage: float
    threshold: float
    is_critical: bool = False

    @property
    def severity(self) -> str:
        """Return severity level based on type."""
        return "CRITICAL" if self.is_critical else "WARNING"


#: Modules held to the higher threshold when nothing else says otherwise.
DEFAULT_CRITICAL_PATTERNS = [
    "iris_vector_rag/pipelines/",
    "iris_vector_rag/storage/",
    "iris_vector_rag/validation/",
]


def load_critical_patterns() -> List[str]:
    """Load critical module patterns from .coveragerc, or fall back to defaults.

    The defaults have to apply when there is no .coveragerc: this repo keeps its
    coverage config in pyproject.toml, so returning an empty list for a missing
    file measured every module against the plain 60% threshold and the critical
    80% one never applied to anything.
    """
    config_path = Path(".coveragerc")

    if not config_path.exists():
        return list(DEFAULT_CRITICAL_PATTERNS)

    config = configparser.ConfigParser()
    config.read(config_path)

    # Look for custom section
    if config.has_section("coverage:critical_modules"):
        patterns_str = config.get("coverage:critical_modules", "patterns", fallback="")
        # Parse multiline patterns
        patterns = [p.strip() for p in patterns_str.strip().split("\n") if p.strip()]
        if patterns:
            return patterns

    return list(DEFAULT_CRITICAL_PATTERNS)


def is_critical_module(module_path: str, patterns: List[str]) -> bool:
    """Check if module matches critical patterns."""
    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            if module_path.startswith(pattern):
                return True
        else:
            # Exact match or glob pattern
            if Path(module_path).match(pattern):
                return True
    return False


def calculate_coverage_percentage(cov_data: Any, module_path: str) -> float:
    """Calculate coverage percentage for a module."""
    try:
        # Get line numbers
        statements = cov_data.lines(module_path)
        if not statements:
            return 100.0  # No statements = 100% coverage

        # Get executed lines
        executed = cov_data.executed_lines(module_path)

        # Calculate percentage
        coverage_pct = (len(executed) / len(statements)) * 100
        return round(coverage_pct, 1)
    except Exception:
        # If we can't calculate, assume it's okay
        return 100.0


def collect_coverage_warnings(cov: coverage.Coverage) -> List[CoverageWarning]:
    """Collect all coverage warnings from coverage data."""
    warnings = []
    critical_patterns = load_critical_patterns()

    # Get coverage data
    cov_data = cov.get_data()

    # Process each measured file
    for module_path in cov_data.measured_files():
        # Skip test files and external packages
        if module_path.startswith("test") or "site-packages" in module_path:
            continue

        # Calculate coverage
        coverage_pct = calculate_coverage_percentage(cov_data, module_path)

        # Determine threshold
        is_critical = is_critical_module(module_path, critical_patterns)
        threshold = 80.0 if is_critical else 60.0

        # Check if below threshold
        if coverage_pct < threshold:
            warnings.append(
                CoverageWarning(
                    module_path=module_path,
                    current_coverage=coverage_pct,
                    threshold=threshold,
                    is_critical=is_critical,
                )
            )

    return warnings


def format_warning(warning: CoverageWarning) -> str:
    """Format a coverage warning for display."""
    return (
        f"WARNING: Coverage below threshold - {warning.module_path}: "
        f"{warning.current_coverage}% < {warning.threshold}% "
        f"[{warning.severity}]"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Display coverage warnings after test execution."""
    try:
        # Try to get coverage instance from pytest-cov
        cov = getattr(config, "_cov", None)
        if not cov:
            # Try to find coverage data file
            cov = coverage.Coverage()
            cov.load()

        # Collect warnings
        warnings = collect_coverage_warnings(cov)

        if warnings:
            # Add section header
            terminalreporter.section("Coverage Warnings")

            # Sort by severity (critical first) then by coverage
            warnings.sort(key=lambda w: (not w.is_critical, w.current_coverage))

            # Display each warning
            for warning in warnings:
                message = format_warning(warning)
                terminalreporter.write_line(message, yellow=True, bold=True)

            # Summary
            critical_count = sum(1 for w in warnings if w.is_critical)
            if critical_count > 0:
                terminalreporter.write_line(
                    f"\n{critical_count} critical modules below 80% coverage!",
                    red=True,
                    bold=True,
                )

            terminalreporter.write_line(
                f"Total modules with low coverage: {len(warnings)}", yellow=True
            )
    except Exception as e:
        # Don't fail tests if coverage warnings fail
        if config.option.verbose > 0:
            terminalreporter.write_line(f"Coverage warning plugin error: {e}", red=True)


def pytest_configure(config):
    """Register plugin with pytest."""
    config.addinivalue_line(
        "markers", "coverage_critical: mark test as critical for coverage purposes"
    )
