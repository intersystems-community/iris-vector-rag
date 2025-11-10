"""
Contract tests for root directory cleanup (Feature 052).

These tests validate that:
1. Root directory contains ≤15 files (excluding directories)
2. Essential files remain in root
3. Files are organized in correct subdirectories
4. No temporary/generated files exist in root
5. .gitignore properly excludes generated content

Run after cleanup to verify success:
    pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py -v
"""

import os
from pathlib import Path
from typing import List, Set

import pytest


# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent.parent


class TestRootDirectoryContract:
    """Contract tests for repository root directory organization."""

    def test_root_file_count_within_limit(self):
        """
        FR-001: Root directory contains ≤15 files (excluding directories).

        Target: 991 files → <15 files
        """
        root_files = [
            f for f in os.listdir(REPO_ROOT)
            if os.path.isfile(os.path.join(REPO_ROOT, f))
        ]

        # Filter out hidden files that are allowed
        visible_files = [f for f in root_files if not f.startswith('.')]

        assert len(visible_files) <= 15, (
            f"Root directory has {len(visible_files)} visible files, "
            f"expected ≤15. Files: {sorted(visible_files)}"
        )

    def test_essential_files_present_in_root(self):
        """
        Verify essential files remain in root directory.

        Required files:
        - README.md (GitHub landing page)
        - LICENSE (legal)
        - pyproject.toml (package config)
        - Makefile (build automation)
        - docker-compose.yml (default Docker setup)
        - .gitignore (version control)
        - .env.example (environment template)
        - .pre-commit-config.yaml (pre-commit hooks)
        """
        essential_files = {
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "Makefile",
            "docker-compose.yml",
            ".gitignore",
            ".env.example",
            ".pre-commit-config.yaml",
        }

        missing_files = []
        for file in essential_files:
            file_path = REPO_ROOT / file
            if not file_path.exists():
                missing_files.append(file)

        assert not missing_files, (
            f"Missing essential files in root: {missing_files}"
        )

    def test_no_log_files_in_root(self):
        """
        FR-002: All log files moved to logs/ directory.

        No .log files should exist in root directory.
        """
        log_files = [
            f for f in os.listdir(REPO_ROOT)
            if f.endswith('.log') and os.path.isfile(os.path.join(REPO_ROOT, f))
        ]

        assert not log_files, (
            f"Found log files in root directory: {log_files}. "
            f"Expected all logs in logs/ subdirectory."
        )

    def test_no_indexing_logs_in_root(self):
        """
        FR-002: 942 indexing logs removed or moved to logs/indexing/.

        No indexing_CONTINUOUS_RUN_*.log files in root.
        """
        indexing_logs = [
            f for f in os.listdir(REPO_ROOT)
            if f.startswith('indexing_CONTINUOUS_RUN_') and f.endswith('.log')
        ]

        assert not indexing_logs, (
            f"Found {len(indexing_logs)} indexing logs in root. "
            f"Expected 0 (should be deleted or in logs/indexing/)."
        )

    def test_scripts_directory_exists(self):
        """
        FR-003: Shell scripts consolidated in scripts/ directory.
        """
        scripts_dir = REPO_ROOT / "scripts"
        assert scripts_dir.exists(), "scripts/ directory not found"
        assert scripts_dir.is_dir(), "scripts/ is not a directory"

    def test_no_shell_scripts_in_root(self):
        """
        FR-003: No .sh files in root (should be in scripts/).

        Allowed exceptions:
        - None (all scripts should be in scripts/)
        """
        sh_files = [
            f for f in os.listdir(REPO_ROOT)
            if f.endswith('.sh') and os.path.isfile(os.path.join(REPO_ROOT, f))
        ]

        # Remove allowed exceptions (none expected after cleanup)
        allowed_scripts: Set[str] = set()
        unexpected_scripts = [f for f in sh_files if f not in allowed_scripts]

        assert not unexpected_scripts, (
            f"Found shell scripts in root: {unexpected_scripts}. "
            f"Expected all scripts in scripts/ subdirectory."
        )

    def test_test_artifacts_not_in_root(self):
        """
        FR-004: Test artifacts moved to tests/artifacts/.

        No htmlcov/, coverage_html/, .coverage, coverage.json in root.
        """
        artifacts = ["htmlcov", "coverage_html", ".coverage", "coverage.json"]

        found_artifacts = [
            artifact for artifact in artifacts
            if (REPO_ROOT / artifact).exists()
        ]

        assert not found_artifacts, (
            f"Found test artifacts in root: {found_artifacts}. "
            f"Expected in tests/artifacts/ or deleted."
        )

    def test_docs_directory_exists(self):
        """
        FR-005: Documentation files organized in docs/ directory.
        """
        docs_dir = REPO_ROOT / "docs"
        assert docs_dir.exists(), "docs/ directory not found"
        assert docs_dir.is_dir(), "docs/ is not a directory"

    def test_contributing_in_docs(self):
        """
        FR-005: CONTRIBUTING.md moved to docs/.
        """
        contributing = REPO_ROOT / "docs" / "CONTRIBUTING.md"
        assert contributing.exists(), (
            "docs/CONTRIBUTING.md not found. "
            "Expected CONTRIBUTING.md in docs/ directory."
        )

    def test_config_directory_exists(self):
        """
        FR-006: Configuration files organized in config/ directory.
        """
        config_dir = REPO_ROOT / "config"
        assert config_dir.exists(), "config/ directory not found"
        assert config_dir.is_dir(), "config/ is not a directory"

    def test_flake8_in_config(self):
        """
        FR-006: .flake8 configuration moved to config/.
        """
        flake8_config = REPO_ROOT / "config" / ".flake8"
        assert flake8_config.exists(), (
            "config/.flake8 not found. "
            "Expected .flake8 in config/ directory."
        )

    def test_coveragerc_in_config(self):
        """
        FR-006: .coveragerc moved to config/.
        """
        coveragerc = REPO_ROOT / "config" / ".coveragerc"
        assert coveragerc.exists(), (
            "config/.coveragerc not found. "
            "Expected .coveragerc in config/ directory."
        )

    def test_docker_compose_variants_in_config(self):
        """
        FR-007: Docker compose variants moved to config/docker/.

        Variants:
        - docker-compose.api.yml
        - docker-compose.full.yml
        - docker-compose.licensed.yml
        - docker-compose.mcp.yml
        - docker-compose.test.yml
        - docker-compose.iris-only.yml
        """
        docker_dir = REPO_ROOT / "config" / "docker"
        assert docker_dir.exists(), "config/docker/ directory not found"

        variants = [
            "docker-compose.api.yml",
            "docker-compose.full.yml",
            "docker-compose.licensed.yml",
            "docker-compose.mcp.yml",
            "docker-compose.test.yml",
            "docker-compose.iris-only.yml",
        ]

        missing_variants = []
        for variant in variants:
            variant_path = docker_dir / variant
            if not variant_path.exists():
                missing_variants.append(variant)

        assert not missing_variants, (
            f"Missing docker-compose variants in config/docker/: {missing_variants}"
        )

    def test_docker_compose_primary_in_root(self):
        """
        FR-007: Primary docker-compose.yml remains in root.
        """
        primary_compose = REPO_ROOT / "docker-compose.yml"
        assert primary_compose.exists(), (
            "docker-compose.yml not found in root. "
            "Primary Docker Compose file must remain in root."
        )

    def test_no_docker_compose_variants_in_root(self):
        """
        FR-007: No docker-compose variants in root (except primary).

        Only docker-compose.yml should remain in root.
        """
        compose_files = [
            f for f in os.listdir(REPO_ROOT)
            if f.startswith('docker-compose.') and f.endswith('.yml')
        ]

        # Remove primary from list
        variants_in_root = [f for f in compose_files if f != "docker-compose.yml"]

        assert not variants_in_root, (
            f"Found docker-compose variants in root: {variants_in_root}. "
            f"Expected only docker-compose.yml in root, variants in config/docker/."
        )

    def test_no_temporary_files_in_root(self):
        """
        FR-010: Temporary/generated files removed.

        No .DS_Store, .coverage, coverage.json, etc.
        """
        temp_files = [".DS_Store", ".coverage", "coverage.json", "Thumbs.db"]

        found_temp_files = [
            f for f in temp_files
            if (REPO_ROOT / f).exists()
        ]

        assert not found_temp_files, (
            f"Found temporary files in root: {found_temp_files}. "
            f"Expected these to be deleted."
        )

    def test_no_comprehensive_ragas_results_in_root(self):
        """
        FR-009, FR-011: Old evaluation result directories removed or archived.

        No comprehensive_ragas_results_* directories in root.
        """
        ragas_dirs = [
            f for f in os.listdir(REPO_ROOT)
            if f.startswith('comprehensive_ragas_results_')
            and os.path.isdir(os.path.join(REPO_ROOT, f))
        ]

        assert not ragas_dirs, (
            f"Found {len(ragas_dirs)} RAGAS result directories in root. "
            f"Expected these in archive/eval_results/ or deleted."
        )

    def test_gitignore_includes_logs(self):
        """
        FR-002: .gitignore properly excludes log directories.
        """
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore not found"

        content = gitignore.read_text()

        # Check for log-related entries
        log_patterns = ["logs/", "*.log"]
        missing_patterns = [
            pattern for pattern in log_patterns
            if pattern not in content
        ]

        assert not missing_patterns, (
            f".gitignore missing log patterns: {missing_patterns}. "
            f"Add these to prevent committing logs."
        )

    def test_gitignore_includes_test_artifacts(self):
        """
        FR-004: .gitignore properly excludes test artifact directories.
        """
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore not found"

        content = gitignore.read_text()

        # Check for artifact-related entries
        artifact_patterns = [
            "htmlcov/",
            "coverage_html/",
            ".coverage",
            "coverage.json",
            "tests/artifacts/",
        ]

        missing_patterns = [
            pattern for pattern in artifact_patterns
            if pattern not in content
        ]

        assert not missing_patterns, (
            f".gitignore missing artifact patterns: {missing_patterns}. "
            f"Add these to prevent committing test artifacts."
        )


class TestBackwardCompatibility:
    """Tests ensuring CI/CD and development workflows still work."""

    def test_makefile_exists(self):
        """
        FR-012: Makefile still in root and accessible.
        """
        makefile = REPO_ROOT / "Makefile"
        assert makefile.exists(), "Makefile not found in root"

    def test_pyproject_toml_exists(self):
        """
        FR-013: pyproject.toml configuration still accessible.
        """
        pyproject = REPO_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found in root"

    def test_readme_exists(self):
        """
        FR-015: README.md updated and present in root.
        """
        readme = REPO_ROOT / "README.md"
        assert readme.exists(), "README.md not found in root"

        # Verify it's not empty
        content = readme.read_text()
        assert len(content) > 100, "README.md appears to be empty or too short"

    def test_env_example_exists(self):
        """
        Environment template still accessible.
        """
        env_example = REPO_ROOT / ".env.example"
        assert env_example.exists(), ".env.example not found in root"


class TestDirectoryStructure:
    """Tests verifying proper directory structure after cleanup."""

    def test_expected_directories_exist(self):
        """
        Verify all expected subdirectories were created.
        """
        expected_dirs = [
            "config",
            "config/docker",
            "docs",
            "scripts",
            "tests",
        ]

        missing_dirs = []
        for dir_name in expected_dirs:
            dir_path = REPO_ROOT / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)

        assert not missing_dirs, (
            f"Missing expected directories: {missing_dirs}"
        )

    def test_archive_directory_structure(self):
        """
        If archive/ exists, verify it has proper structure.
        """
        archive_dir = REPO_ROOT / "archive"

        # Archive directory is optional (may not exist if all old content deleted)
        if archive_dir.exists():
            assert archive_dir.is_dir(), "archive/ is not a directory"

            # If it exists, it should have eval_results/
            # (other subdirs optional)
            # This is informational, not a hard requirement

    def test_logs_directory_structure(self):
        """
        If logs/ exists, verify it's properly structured.
        """
        logs_dir = REPO_ROOT / "logs"

        # logs/ directory is optional (may not exist if all logs deleted)
        if logs_dir.exists():
            assert logs_dir.is_dir(), "logs/ is not a directory"
