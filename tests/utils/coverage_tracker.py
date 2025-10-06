"""Coverage trend tracking utility.

Feature: 025-fixes-for-testing
Task: T014

Tracks coverage trends over time using CoverageReport entities.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import coverage


@dataclass
class CoverageReport:
    """CoverageReport entity from data-model.md.

    Attributes:
        module_name: Name of the module (e.g., iris_rag.pipelines.basic)
        total_lines: Total executable lines in module
        covered_lines: Number of lines covered by tests
        percentage: Coverage percentage (0-100)
        missing_lines: List of uncovered line numbers
        timestamp: When coverage was measured
        is_critical: Whether this is a critical module (80% target)
    """

    module_name: str
    total_lines: int
    covered_lines: int
    percentage: float
    missing_lines: List[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    is_critical: bool = False


@dataclass
class CoverageSummary:
    """Overall coverage summary.

    Attributes:
        overall_percentage: Overall coverage across all modules
        total_modules: Total number of modules
        critical_modules_percentage: Average coverage for critical modules
        below_target_modules: Modules below their target (60% or 80%)
        module_reports: Coverage reports per module
    """

    overall_percentage: float = 0.0
    total_modules: int = 0
    critical_modules_percentage: float = 0.0
    below_target_modules: List[CoverageReport] = field(default_factory=list)
    module_reports: List[CoverageReport] = field(default_factory=list)


class CoverageTracker:
    """Tracks coverage trends over time."""

    # Critical modules requiring 80% coverage
    CRITICAL_MODULES = [
        "iris_rag.pipelines",
        "iris_rag.storage",
        "iris_rag.validation",
        "iris_rag.core",
        "common.db_vector_utils",
        "common.iris_connection_manager",
    ]

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize coverage tracker.

        Args:
            db_path: Path to SQLite database for storing trends (optional)
        """
        self.db_path = db_path
        if db_path:
            self._init_db()

    def _init_db(self):
        """Initialize SQLite database for trend tracking."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS coverage_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT NOT NULL,
                total_lines INTEGER NOT NULL,
                covered_lines INTEGER NOT NULL,
                percentage REAL NOT NULL,
                timestamp TEXT NOT NULL,
                is_critical INTEGER NOT NULL,
                missing_lines TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def parse_coverage_file(self, coverage_file: Path = Path(".coverage")) -> CoverageSummary:
        """Parse .coverage file and create CoverageReport entities.

        Args:
            coverage_file: Path to .coverage file (default: .coverage)

        Returns:
            CoverageSummary with per-module coverage
        """
        if not coverage_file.exists():
            raise FileNotFoundError(f"Coverage file not found: {coverage_file}")

        # Load coverage data
        cov = coverage.Coverage(data_file=str(coverage_file))
        cov.load()

        summary = CoverageSummary()
        module_reports = []

        # Analyze each file
        analysis = cov.analysis2

        for filename in cov.get_data().measured_files():
            # Convert file path to module name
            module_name = self._file_to_module(filename)

            if not module_name:
                continue  # Skip files not in iris_rag or common

            # Get coverage analysis
            try:
                _, statements, _, missing, _ = cov.analysis2(filename)
            except Exception:
                continue  # Skip files that can't be analyzed

            total_lines = len(statements)
            covered_lines = total_lines - len(missing)
            percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

            # Check if critical module
            is_critical = any(
                module_name.startswith(crit) for crit in self.CRITICAL_MODULES
            )

            report = CoverageReport(
                module_name=module_name,
                total_lines=total_lines,
                covered_lines=covered_lines,
                percentage=percentage,
                missing_lines=sorted(missing),
                is_critical=is_critical,
            )

            module_reports.append(report)
            summary.module_reports.append(report)

            # Store in database if configured
            if self.db_path:
                self._store_report(report)

        # Calculate summary statistics
        if module_reports:
            total_statements = sum(r.total_lines for r in module_reports)
            total_covered = sum(r.covered_lines for r in module_reports)
            summary.overall_percentage = (
                (total_covered / total_statements * 100) if total_statements > 0 else 0.0
            )
            summary.total_modules = len(module_reports)

            # Calculate critical modules percentage
            critical_reports = [r for r in module_reports if r.is_critical]
            if critical_reports:
                crit_total = sum(r.total_lines for r in critical_reports)
                crit_covered = sum(r.covered_lines for r in critical_reports)
                summary.critical_modules_percentage = (
                    (crit_covered / crit_total * 100) if crit_total > 0 else 0.0
                )

            # Identify modules below target
            for report in module_reports:
                target = 80.0 if report.is_critical else 60.0
                if report.percentage < target:
                    summary.below_target_modules.append(report)

        return summary

    def _file_to_module(self, filepath: str) -> Optional[str]:
        """Convert file path to Python module name.

        Args:
            filepath: File path (e.g., /path/to/iris_rag/pipelines/basic.py)

        Returns:
            Module name (e.g., iris_rag.pipelines.basic) or None
        """
        path = Path(filepath)

        # Only track iris_rag and common modules
        parts = path.parts
        if "iris_rag" in parts:
            idx = parts.index("iris_rag")
            module_parts = parts[idx:]
        elif "common" in parts:
            idx = parts.index("common")
            module_parts = parts[idx:]
        else:
            return None

        # Remove .py extension and convert to module name
        module_name = ".".join(module_parts)
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        return module_name

    def _store_report(self, report: CoverageReport):
        """Store coverage report in database.

        Args:
            report: CoverageReport to store
        """
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO coverage_reports
            (module_name, total_lines, covered_lines, percentage, timestamp, is_critical, missing_lines)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                report.module_name,
                report.total_lines,
                report.covered_lines,
                report.percentage,
                report.timestamp.isoformat(),
                1 if report.is_critical else 0,
                json.dumps(report.missing_lines),
            ),
        )

        conn.commit()
        conn.close()

    def print_summary(self, summary: CoverageSummary):
        """Print coverage summary to console.

        Args:
            summary: CoverageSummary to print
        """
        print(f"\n{'=' * 60}")
        print("Coverage Summary")
        print(f"{'=' * 60}")
        print(f"Overall Coverage: {summary.overall_percentage:.1f}%")
        print(f"Total Modules: {summary.total_modules}")
        print(f"Critical Modules Coverage: {summary.critical_modules_percentage:.1f}%")
        print()

        if summary.below_target_modules:
            print(f"Modules Below Target ({len(summary.below_target_modules)}):")
            for report in sorted(
                summary.below_target_modules, key=lambda r: r.percentage
            ):
                target = 80.0 if report.is_critical else 60.0
                print(
                    f"  {report.module_name}: {report.percentage:.1f}% "
                    f"(target: {target:.0f}%, {report.covered_lines}/{report.total_lines} lines)"
                )

        print(f"{'=' * 60}\n")
