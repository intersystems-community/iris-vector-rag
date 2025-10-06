"""Contract tests for coverage reporting requirements.

Feature: 025-fixes-for-testing
Contract: coverage_reporting_contract.md
"""

import configparser
import subprocess

import pytest


def test_coverage_configuration():
    """REQ-1: Validate .coveragerc excludes test files."""
    config = configparser.ConfigParser()
    config.read(".coveragerc")

    # Check [coverage:run] section exists
    assert config.has_section("coverage:run"), "Missing [coverage:run] section"

    # Check source configuration
    source = config.get("coverage:run", "source")
    assert "iris_rag" in source, "iris_rag must be in coverage source"
    assert "common" in source, "common must be in coverage source"

    # Check omit patterns
    omit = config.get("coverage:run", "omit")
    assert "*/tests/*" in omit, "Must omit */tests/* from coverage"
    assert "*/test_*" in omit, "Must omit */test_* from coverage"
    assert "*/__pycache__/*" in omit, "Must omit */__pycache__/* from coverage"


def test_coverage_report_configuration():
    """REQ-3: Validate coverage reporting configuration."""
    config = configparser.ConfigParser()
    config.read(".coveragerc")

    # Check [coverage:report] section exists
    assert config.has_section("coverage:report"), "Missing [coverage:report] section"

    # Check precision
    precision = config.get("coverage:report", "precision")
    assert precision == "1", "Coverage precision must be 1 decimal place"

    # Check show_missing
    show_missing = config.get("coverage:report", "show_missing")
    assert show_missing.lower() == "true", "show_missing must be True"


def test_coverage_html_configuration():
    """REQ-2: Validate HTML coverage report configuration."""
    config = configparser.ConfigParser()
    config.read(".coveragerc")

    # Check [coverage:html] section exists
    assert config.has_section("coverage:html"), "Missing [coverage:html] section"

    # Check directory
    directory = config.get("coverage:html", "directory")
    assert directory == "htmlcov", "HTML coverage directory must be htmlcov"


def test_coverage_runs_successfully():
    """REQ-2: Coverage reports can be generated."""
    # Run a quick coverage check on a small subset
    result = subprocess.run(
        [
            "pytest",
            "tests/unit/",
            "--cov=iris_rag",
            "--cov=common",
            "--cov-report=term",
            "-q",
            "--tb=no",
            "-p",
            "no:randomly",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Check that coverage ran without errors
    assert result.returncode in [
        0,
        1,
    ], f"Coverage run failed with return code {result.returncode}"
    assert "TOTAL" in result.stdout, "Coverage report must include TOTAL line"


def test_coverage_targets_defined():
    """REQ-2: Coverage targets are documented."""
    # This is a documentation test - targets defined in contract
    # Overall: >= 60%
    # Critical modules (pipelines, storage, validation): >= 80%

    # Just verify the contract exists
    import pathlib

    contract_path = pathlib.Path(
        "specs/025-fixes-for-testing/contracts/coverage_reporting_contract.md"
    )
    assert contract_path.exists(), "Coverage reporting contract must exist"

    content = contract_path.read_text()
    assert (
        "60%" in content
    ), "Coverage contract must document 60% overall target"
    assert "80%" in content, "Coverage contract must document 80% critical module target"
