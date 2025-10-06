#!/usr/bin/env python3
"""
Performance benchmark runner for RAG templates framework.

This script runs performance benchmarks and compares results against baseline.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psutil


def get_project_root() -> Path:
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent.parent


def load_baseline(baseline_file: Path) -> Dict[str, Any]:
    """Load baseline benchmark results."""
    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            return json.load(f)
    return {}


def save_results(results: Dict[str, Any], output_file: Path) -> None:
    """Save benchmark results to file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def compare_with_baseline(
    current: Dict[str, Any], baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare current results with baseline."""
    comparison = {
        "timestamp": current["timestamp"],
        "performance_changes": [],
        "summary": {"improved": 0, "degraded": 0, "unchanged": 0, "new_benchmarks": 0},
    }

    current_benchmarks = {b["name"]: b for b in current["benchmarks"]}
    baseline_benchmarks = {b["name"]: b for b in baseline.get("benchmarks", [])}

    for name, current_bench in current_benchmarks.items():
        if name in baseline_benchmarks:
            baseline_bench = baseline_benchmarks[name]
            current_time = current_bench["stats"]["mean"]
            baseline_time = baseline_bench["stats"]["mean"]

            # Calculate percentage change
            change_percent = ((current_time - baseline_time) / baseline_time) * 100

            status = "unchanged"
            if abs(change_percent) > 5:  # 5% threshold
                status = "degraded" if change_percent > 0 else "improved"

            comparison["performance_changes"].append(
                {
                    "name": name,
                    "current_time": current_time,
                    "baseline_time": baseline_time,
                    "change_percent": change_percent,
                    "status": status,
                }
            )

            comparison["summary"][status] += 1
        else:
            comparison["summary"]["new_benchmarks"] += 1

    return comparison


def run_pytest_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """Run pytest benchmarks."""
    import subprocess

    project_root = get_project_root()
    benchmark_dir = project_root / "benchmarks"
    results_dir = project_root / "benchmarks" / "results"

    # Ensure directories exist
    benchmark_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Build pytest command
    cmd = [
        "pytest",
        str(benchmark_dir),
        "--benchmark-only",
        "--benchmark-json",
        str(results_dir / "benchmark_results.json"),
        "--benchmark-sort=mean",
        "--benchmark-warmup=on",
        "--benchmark-warmup-iterations=3",
        "--benchmark-min-rounds=5",
    ]

    if args.verbose:
        cmd.append("-v")

    print(f"Running benchmarks: {' '.join(cmd)}")

    # Run benchmarks
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    if result.returncode != 0:
        print(f"Benchmark execution failed: {result.stderr}")
        sys.exit(1)

    # Load results
    results_file = results_dir / "benchmark_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)

    return {}


def run_custom_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """Run custom performance benchmarks."""
    project_root = get_project_root()

    # Add project to Python path
    sys.path.insert(0, str(project_root))

    benchmarks = []
    timestamp = datetime.now().isoformat()

    # System information
    system_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "timestamp": timestamp,
    }

    print("Running custom benchmarks...")

    try:
        # Import required modules
        from iris_rag.config.manager import ConfigManager
        from iris_rag.pipelines.factory import PipelineFactory

        # Initialize components
        config_manager = ConfigManager()

        # Benchmark 1: Pipeline factory initialization
        start_time = time.time()
        for _ in range(10):
            factory = PipelineFactory(config_manager)
        end_time = time.time()

        benchmarks.append(
            {
                "name": "pipeline_factory_init",
                "description": "Pipeline factory initialization",
                "iterations": 10,
                "total_time": end_time - start_time,
                "stats": {
                    "mean": (end_time - start_time) / 10,
                    "min": (end_time - start_time) / 10,
                    "max": (end_time - start_time) / 10,
                },
            }
        )

        # Benchmark 2: Configuration loading
        start_time = time.time()
        for _ in range(5):
            config = config_manager.get_config()
        end_time = time.time()

        benchmarks.append(
            {
                "name": "config_loading",
                "description": "Configuration loading",
                "iterations": 5,
                "total_time": end_time - start_time,
                "stats": {
                    "mean": (end_time - start_time) / 5,
                    "min": (end_time - start_time) / 5,
                    "max": (end_time - start_time) / 5,
                },
            }
        )

        print(f"Completed {len(benchmarks)} custom benchmarks")

    except ImportError as e:
        print(f"Warning: Could not import modules for benchmarks: {e}")

    return {
        "timestamp": timestamp,
        "system_info": system_info,
        "benchmarks": benchmarks,
    }


def generate_report(
    results: Dict[str, Any], comparison: Dict[str, Any], output_file: Path
) -> None:
    """Generate a markdown report."""
    report_lines = [
        "# Performance Benchmark Report",
        "",
        f"**Generated:** {results['timestamp']}",
        "",
        "## System Information",
        "",
        f"- **Python Version:** {results['system_info']['python_version']}",
        f"- **Platform:** {results['system_info']['platform']}",
        f"- **CPU Count:** {results['system_info']['cpu_count']}",
        f"- **Total Memory:** {results['system_info']['memory_total'] / (1024**3):.1f} GB",
        "",
        "## Benchmark Results",
        "",
        "| Benchmark | Mean Time (s) | Iterations |",
        "|-----------|---------------|------------|",
    ]

    for benchmark in results["benchmarks"]:
        mean_time = benchmark["stats"]["mean"]
        iterations = benchmark["iterations"]
        report_lines.append(f"| {benchmark['name']} | {mean_time:.6f} | {iterations} |")

    if comparison:
        report_lines.extend(
            [
                "",
                "## Performance Changes",
                "",
                f"- **Improved:** {comparison['summary']['improved']}",
                f"- **Degraded:** {comparison['summary']['degraded']}",
                f"- **Unchanged:** {comparison['summary']['unchanged']}",
                f"- **New Benchmarks:** {comparison['summary']['new_benchmarks']}",
                "",
                "### Detailed Changes",
                "",
                "| Benchmark | Current (s) | Baseline (s) | Change (%) | Status |",
                "|-----------|-------------|--------------|------------|--------|",
            ]
        )

        for change in comparison["performance_changes"]:
            status_emoji = {"improved": "✅", "degraded": "❌", "unchanged": "➖"}.get(
                change["status"], "❓"
            )

            report_lines.append(
                f"| {change['name']} | "
                f"{change['current_time']:.6f} | "
                f"{change['baseline_time']:.6f} | "
                f"{change['change_percent']:+.1f}% | "
                f"{status_emoji} {change['status']} |"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument(
        "--type",
        choices=["pytest", "custom", "all"],
        default="all",
        help="Type of benchmarks to run",
    )
    parser.add_argument(
        "--baseline", type=Path, help="Baseline results file for comparison"
    )
    parser.add_argument("--output", type=Path, help="Output directory for results")
    parser.add_argument(
        "--docker", action="store_true", help="Running in Docker container"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set default paths
    project_root = get_project_root()
    if not args.output:
        args.output = project_root / "benchmarks" / "results"
    if not args.baseline:
        args.baseline = args.output / "baseline.json"

    print("RAG Templates Performance Benchmarks")
    print("=" * 40)
    print(f"Type: {args.type}")
    print(f"Output: {args.output}")
    print(f"Baseline: {args.baseline}")
    print()

    # Load baseline results
    baseline = load_baseline(args.baseline)

    # Run benchmarks
    if args.type in ["custom", "all"]:
        results = run_custom_benchmarks(args)
    elif args.type == "pytest":
        results = run_pytest_benchmarks(args)
    else:
        results = run_custom_benchmarks(args)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.output / f"benchmark_results_{timestamp}.json"
    save_results(results, results_file)

    # Compare with baseline
    comparison = None
    if baseline:
        comparison = compare_with_baseline(results, baseline)
        comparison_file = args.output / f"benchmark_comparison_{timestamp}.json"
        save_results(comparison, comparison_file)

    # Generate report
    report_file = args.output / f"benchmark_report_{timestamp}.md"
    generate_report(results, comparison, report_file)

    print(f"\nResults saved to: {results_file}")
    print(f"Report saved to: {report_file}")

    # Check for performance regressions
    if comparison:
        degraded_count = comparison["summary"]["degraded"]
        if degraded_count > 0:
            print(
                f"\n⚠️  Warning: {degraded_count} benchmark(s) show performance degradation"
            )
            if not args.docker:  # Don't fail in CI unless explicitly configured
                sys.exit(1)
        else:
            print("\n✅ No performance regressions detected")


if __name__ == "__main__":
    main()
