"""
Unified test-runner for the RAG-templates repository.

Usage
-----
poetry run python run_tests.py [suite] [additional pytest args]

`suite` can be one of:
    unit            – quick unit tests (default)
    integration     – integration tests
    e2e             – end-to-end (marked e2e) tests
    thousand        – 1000-doc tests (real PMC)
    all             – everything (lint + docs + tests)

Any extra arguments after the suite name are forwarded directly to pytest.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHONPATH = os.environ.get("PYTHONPATH", "")


def _ensure_pythonpath() -> None:
    """Add project root to PYTHONPATH so tests find local packages."""
    paths = [str(PROJECT_ROOT)]
    if DEFAULT_PYTHONPATH:
        paths.append(DEFAULT_PYTHONPATH)
    os.environ["PYTHONPATH"] = ":".join(paths)


def _load_dotenv() -> None:
    """Load .env file if present for local secrets."""
    dotenv_path = PROJECT_ROOT / ".env"
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:  # pragma: no cover
        print("python-dotenv not installed – skipping .env loading")
        return
    load_dotenv(dotenv_path)  # type: ignore


def _run(command: List[str]) -> int:
    """Run a subprocess, streaming output."""
    print(f"Running: {' '.join(command)}")
    return subprocess.call(command)


def run_pytest(py_args: List[str]) -> int:
    """Invoke pytest with provided args inside poetry env."""
    return _run(["pytest", "-xvs", *py_args])


def main() -> None:
    _ensure_pythonpath()
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Unified test runner")
    parser.add_argument(
        "suite",
        nargs="?",
        default="unit",
        choices=["unit", "integration", "e2e", "thousand", "all"],
        help="Select which predefined test suite to run",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest",
    )
    args = parser.parse_args()

    # Mapping of suite to pytest markers / paths
    suite_map = {
        "unit": ["-m", "unit"],
        "integration": ["-m", "integration"],
        "e2e": ["-m", "e2e"],
        "thousand": ["-m", "e2e", "tests/test_*_1000*.py"],
    }

    if args.suite == "all":
        # run Makefile target test-all
        sys.exit(_run(["make", "test-all"]))

    pytest_base = suite_map.get(args.suite, [])
    full_args = [*pytest_base, *args.pytest_args]
    sys.exit(run_pytest(full_args))


if __name__ == "__main__":
    main()
