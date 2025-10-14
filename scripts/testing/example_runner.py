"""
Example Test Runner for rag-templates.

This module provides the core framework for executing and validating
example scripts in isolated environments with comprehensive monitoring.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class ExampleTestResult:
    """Results from executing an example script."""

    # Execution metadata
    script_path: str
    success: bool = False
    execution_time: float = 0.0
    exit_code: int = -1

    # Output capture
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""

    # Performance metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0

    # Validation results
    validation_results: Optional[Dict] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    environment: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "script_path": self.script_path,
            "success": self.success,
            "execution_time": self.execution_time,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "validation_score": (
                self.validation_results.get("score", 0.0)
                if self.validation_results
                else 0.0
            ),
            "validation_issues": (
                self.validation_results.get("issues", [])
                if self.validation_results
                else []
            ),
            "timestamp": self.timestamp,
        }


class ExampleTestRunner:
    """
    Framework for executing and validating example scripts.

    Provides isolated execution environments, comprehensive monitoring,
    and integration with mock providers for predictable testing.
    """

    def __init__(self, project_root: Path = None, config: Dict = None):
        """
        Initialize the test runner.

        Args:
            project_root: Root directory of the project
            config: Configuration dictionary with test settings
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.config = config or {}

        # Ensure directories exist
        self.reports_dir = self.scripts_dir / "validation" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_example(
        self, script_path: str, timeout: int = 300, mode: str = "mock"
    ) -> ExampleTestResult:
        """
        Execute an example script with comprehensive monitoring.

        Args:
            script_path: Relative path to script from scripts directory
            timeout: Maximum execution time in seconds
            mode: Execution mode ("mock" or "real")

        Returns:
            ExampleTestResult with execution details and metrics
        """
        result = ExampleTestResult(script_path=script_path)
        full_path = self.scripts_dir / script_path

        # Validate script exists
        if not full_path.exists():
            result.error_message = f"Script not found: {full_path}"
            return result

        # Setup environment
        env = self._create_test_environment(mode)
        result.environment = {k: v for k, v in env.items() if not k.startswith("_")}

        try:
            # Start performance monitoring
            monitor = PerformanceMonitor()
            monitor.start()

            start_time = time.time()

            # Execute script
            process = subprocess.run(
                [sys.executable, str(full_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            # Capture results
            result.execution_time = time.time() - start_time
            result.exit_code = process.returncode
            result.stdout = process.stdout
            result.stderr = process.stderr
            result.success = process.returncode == 0

            # Get performance metrics
            metrics = monitor.get_metrics()
            result.peak_memory_mb = metrics.get("peak_memory_mb", 0.0)
            result.avg_cpu_percent = metrics.get("avg_cpu_percent", 0.0)

            if not result.success:
                result.error_message = (
                    f"Script failed with exit code {process.returncode}"
                )
                if process.stderr:
                    result.error_message += f": {process.stderr.strip()}"

        except subprocess.TimeoutExpired:
            result.error_message = f"Script timed out after {timeout} seconds"
        except Exception as e:
            result.error_message = f"Execution error: {str(e)}"

        return result

    def run_multiple_examples(
        self, patterns: List[str] = None, mode: str = "mock", timeout: int = 300
    ) -> List[ExampleTestResult]:
        """
        Execute multiple example scripts matching patterns.

        Args:
            patterns: List of patterns to match script names (None for all)
            mode: Execution mode ("mock" or "real")
            timeout: Maximum execution time per script

        Returns:
            List of ExampleTestResult objects
        """
        scripts = self._discover_example_scripts(patterns)
        results = []

        for script in scripts:
            print(f"Running: {script}")
            result = self.run_example(script, timeout=timeout, mode=mode)
            results.append(result)

            # Brief status update
            status = "✅" if result.success else "❌"
            print(f"  {status} {script} ({result.execution_time:.1f}s)")

        return results

    def generate_report(
        self, results: List[ExampleTestResult], output_file: str = None
    ) -> str:
        """
        Generate comprehensive test report.

        Args:
            results: List of test results
            output_file: Optional file path to save report

        Returns:
            Report content as string
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Calculate summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.success)
        avg_time = sum(r.execution_time for r in results) / total if total > 0 else 0
        avg_memory = sum(r.peak_memory_mb for r in results) / total if total > 0 else 0

        # Generate report content
        report_lines = [
            "# Example Test Report",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Examples**: {total}",
            f"**Passed**: {passed}",
            f"**Failed**: {total - passed}",
            f"**Success Rate**: {(passed/total*100):.1f}%" if total > 0 else "N/A",
            f"**Average Execution Time**: {avg_time:.2f}s",
            f"**Average Memory Usage**: {avg_memory:.1f}MB",
            "",
            "## Detailed Results",
            "",
        ]

        # Add individual results
        for result in results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report_lines.extend(
                [
                    f"### {result.script_path}",
                    f"**Status**: {status}",
                    f"**Execution Time**: {result.execution_time:.2f}s",
                    f"**Memory Usage**: {result.peak_memory_mb:.1f}MB",
                    "",
                ]
            )

            if not result.success:
                report_lines.extend([f"**Error**: {result.error_message}", ""])

            if result.validation_results:
                val = result.validation_results
                report_lines.extend(
                    [
                        f"**Validation Score**: {val.get('score', 0.0):.2f}",
                        f"**Validation Issues**: {len(val.get('issues', []))}",
                        "",
                    ]
                )

            report_lines.append("---")
            report_lines.append("")

        # Add failure analysis if any
        failed_results = [r for r in results if not r.success]
        if failed_results:
            report_lines.extend(["## Failure Analysis", ""])

            error_categories = {}
            for result in failed_results:
                error_type = self._categorize_error(result.error_message)
                if error_type not in error_categories:
                    error_categories[error_type] = []
                error_categories[error_type].append(result.script_path)

            for error_type, scripts in error_categories.items():
                report_lines.extend(
                    [
                        f"### {error_type}",
                        f"**Count**: {len(scripts)}",
                        f"**Scripts**: {', '.join(scripts)}",
                        "",
                    ]
                )

        report_content = "\n".join(report_lines)

        # Save report if requested
        if output_file:
            output_path = self.reports_dir / output_file
        else:
            output_path = self.reports_dir / f"example_test_report_{timestamp}.md"

        with open(output_path, "w") as f:
            f.write(report_content)

        # Also save JSON data for programmatic access
        json_data = {
            "timestamp": timestamp,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": passed / total if total > 0 else 0,
                "avg_execution_time": avg_time,
                "avg_memory_usage": avg_memory,
            },
            "results": [r.to_dict() for r in results],
        }

        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Report saved to: {output_path}")
        print(f"JSON data saved to: {json_path}")

        return report_content

    def _create_test_environment(self, mode: str) -> Dict[str, str]:
        """Create isolated environment for test execution."""
        env = os.environ.copy()

        # Set test-specific variables
        env["EXAMPLE_TEST_MODE"] = "true"
        env["USE_MOCK_LLM"] = "true" if mode == "mock" else "false"
        env["PYTHONPATH"] = str(self.project_root)

        # Disable external integrations in test mode
        if mode == "mock":
            env["DISABLE_EXTERNAL_APIS"] = "true"
            env["MOCK_VECTOR_SEARCH"] = "true"

        return env

    def _discover_example_scripts(self, patterns: List[str] = None) -> List[str]:
        """Discover example scripts matching patterns."""
        example_patterns = [
            "basic/try_*.py",
            "crag/try_*.py",
            "reranking/try_*.py",
            "demo_*.py",
        ]

        scripts = []
        for pattern in example_patterns:
            for script_path in self.scripts_dir.glob(pattern):
                relative_path = script_path.relative_to(self.scripts_dir)
                scripts.append(str(relative_path))

        # Filter by patterns if provided
        if patterns:
            filtered_scripts = []
            for script in scripts:
                for pattern in patterns:
                    if pattern in script:
                        filtered_scripts.append(script)
                        break
            scripts = filtered_scripts

        return sorted(scripts)

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message for analysis."""
        error_lower = error_message.lower()

        if "timeout" in error_lower:
            return "Timeout Errors"
        elif "import" in error_lower or "module" in error_lower:
            return "Import/Module Errors"
        elif "connection" in error_lower or "database" in error_lower:
            return "Database Connection Errors"
        elif "api" in error_lower or "key" in error_lower:
            return "API/Authentication Errors"
        elif "memory" in error_lower:
            return "Memory Errors"
        else:
            return "Other Errors"


class PerformanceMonitor:
    """Monitor performance metrics during example execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.memory_samples = []
        self.cpu_samples = []

    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.memory_samples = []
        self.cpu_samples = []

    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        current_memory = self.process.memory_info().rss
        current_time = time.time()

        # Sample current values
        self.memory_samples.append(current_memory / 1024 / 1024)  # MB
        try:
            cpu_percent = self.process.cpu_percent()
            self.cpu_samples.append(cpu_percent)
        except psutil.NoSuchProcess:
            cpu_percent = 0.0

        return {
            "execution_time": current_time - self.start_time if self.start_time else 0,
            "current_memory_mb": current_memory / 1024 / 1024,
            "peak_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "avg_memory_mb": (
                sum(self.memory_samples) / len(self.memory_samples)
                if self.memory_samples
                else 0
            ),
            "current_cpu_percent": cpu_percent,
            "avg_cpu_percent": (
                sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
            ),
        }
