#!/usr/bin/env python3
"""
Main script for executing example tests in rag-templates.

This script provides a comprehensive CLI interface for running, validating,
and reporting on example script execution with support for different modes,
filtering, and detailed reporting.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.testing.example_runner import ExampleTestResult, ExampleTestRunner
from scripts.testing.mock_providers import MockLLMProvider
from scripts.testing.validation_suite import ValidationSuite


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load testing configuration."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_examples_to_test(
    config: Dict, pattern: str = None, category: str = None
) -> List[str]:
    """Get list of examples to test based on filters."""
    all_examples = list(config.get("examples", {}).keys())

    # Filter by pattern
    if pattern:
        all_examples = [e for e in all_examples if pattern.lower() in e.lower()]

    # Filter by category
    if category:
        category_examples = []
        for example in all_examples:
            example_config = config.get("examples", {}).get(example, {})
            features = example_config.get("features", [])

            if category == "basic" and any(
                cat in example.lower() for cat in ["basic", "simple"]
            ):
                category_examples.append(example)
            elif category == "advanced" and any(
                cat in example.lower() for cat in ["crag", "hybrid", "rerank"]
            ):
                category_examples.append(example)
            elif category == "demo" and "demo" in example.lower():
                category_examples.append(example)
            elif category == "visualization" and any(
                feat in features for feat in ["visualization", "graph_export"]
            ):
                category_examples.append(example)

        all_examples = category_examples

    return sorted(all_examples)


def print_test_summary(examples: List[str], config: Dict):
    """Print summary of tests to be executed."""
    print(f"\n{'='*60}")
    print(f"Example Test Execution Plan")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")

    if not examples:
        print("No examples found matching criteria!")
        return

    # Group by category
    categories = {}
    for example in examples:
        if "basic" in example:
            category = "Basic RAG"
        elif "crag" in example:
            category = "CRAG (Advanced)"
        elif "rerank" in example:
            category = "Reranking"
        elif "hybrid" in example:
            category = "HybridGraphRAG"
        elif "demo" in example:
            category = "Demonstrations"
        else:
            category = "Other"

        if category not in categories:
            categories[category] = []
        categories[category].append(example)

    for category, category_examples in categories.items():
        print(f"\n{category}: {len(category_examples)} examples")
        for example in category_examples:
            timeout = config.get("examples", {}).get(example, {}).get("timeout", 300)
            print(f"  • {example} (timeout: {timeout}s)")


def print_result_summary(
    example: str, result: ExampleTestResult, verbose: bool = False
):
    """Print summary of individual test result."""
    status = "✅ PASS" if result.success else "❌ FAIL"
    print(
        f"  {status} {example} ({result.execution_time:.1f}s, {result.peak_memory_mb:.1f}MB)"
    )

    if not result.success:
        print(f"    Error: {result.error_message}")

    if hasattr(result, "validation_results") and result.validation_results:
        val = result.validation_results
        score_str = f"{val.get('score', 0.0):.2f}"
        issues_count = len(val.get("issues", []))
        print(f"    Validation: {score_str} score, {issues_count} issues")

        if verbose and issues_count > 0:
            for issue in val.get("issues", [])[:3]:  # Show first 3 issues
                print(f"      - {issue}")

    if verbose and result.stderr:
        print(f"    Stderr: {result.stderr[:200]}...")


def generate_final_report(results: List[tuple], config: Dict, args: argparse.Namespace):
    """Generate and display final test report."""
    total = len(results)
    passed = sum(1 for _, r in results if r.success)
    failed = total - passed

    # Calculate validation statistics
    validation_scores = []
    for _, result in results:
        if hasattr(result, "validation_results") and result.validation_results:
            validation_scores.append(result.validation_results.get("score", 0.0))

    avg_validation_score = (
        sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
    )

    # Performance statistics
    execution_times = [r.execution_time for _, r in results if r.success]
    memory_usage = [r.peak_memory_mb for _, r in results if r.success]

    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
    avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0.0

    print(f"\n{'='*60}")
    print(f"Final Test Results")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"Passed: {passed} (✅)")
    print(f"Failed: {failed} (❌)")
    print(f"Success rate: {(passed/total*100):.1f}%" if total > 0 else "N/A")
    print(f"Average validation score: {avg_validation_score:.2f}")
    print(f"Average execution time: {avg_time:.2f}s")
    print(f"Average memory usage: {avg_memory:.1f}MB")

    # Show failed examples
    if failed > 0:
        print(f"\nFailed Examples:")
        for example, result in results:
            if not result.success:
                print(f"  ❌ {example}: {result.error_message}")

    # CI/CD integration
    ci_config = config.get("ci_integration", {})
    if ci_config.get("fail_on_example_failure", True) and failed > 0:
        print(f"\n⚠️  CI/CD Integration: Failing due to {failed} failed examples")
        return 1

    min_validation_score = ci_config.get("fail_on_validation_score_below", 0.7)
    if avg_validation_score < min_validation_score:
        print(
            f"\n⚠️  CI/CD Integration: Failing due to low validation score ({avg_validation_score:.2f} < {min_validation_score})"
        )
        return 1

    print(f"\n🎉 All tests completed successfully!")
    return 0


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for rag-templates examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all examples in mock mode
  python run_example_tests.py

  # Run only basic examples
  python run_example_tests.py --category basic

  # Run examples matching pattern
  python run_example_tests.py --pattern "basic"

  # Run with real LLM calls (requires API keys)
  python run_example_tests.py --mode real --timeout 600

  # Run specific example with verbose output
  python run_example_tests.py --pattern "try_basic_rag" --verbose

  # Generate only reports without running tests
  python run_example_tests.py --report-only
        """,
    )

    # Test execution options
    parser.add_argument("--pattern", help="Pattern to match example names")
    parser.add_argument(
        "--category",
        choices=["basic", "advanced", "demo", "visualization"],
        help="Category of examples to run",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Execution mode (mock=fast, real=with APIs)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout per example in seconds"
    )
    parser.add_argument(
        "--clean-iris",
        action="store_true",
        help="Enable lenient validation for clean IRIS testing",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed results",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate reports from previous results only",
    )

    # Configuration options
    parser.add_argument("--config", type=Path, help="Path to custom configuration file")
    parser.add_argument("--output-dir", type=Path, help="Directory for output reports")

    # Development options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without execution",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue testing after failures",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose and not args.quiet)
    config = load_config(args.config)

    # Get examples to test
    examples = get_examples_to_test(config, args.pattern, args.category)

    if not examples:
        print("No examples found matching the specified criteria.")
        return 1

    # Show test plan
    if not args.quiet:
        print_test_summary(examples, config)

    if args.dry_run:
        print("\n✅ Dry run completed - no tests executed")
        return 0

    # Confirm execution for real mode (skip in CI environments)
    if (
        args.mode == "real"
        and not args.quiet
        and not os.getenv("CI")
        and not os.getenv("GITHUB_ACTIONS")
    ):
        response = input(
            f"\nExecute {len(examples)} examples with real LLM calls? [y/N]: "
        )
        if response.lower() != "y":
            print("Execution cancelled.")
            return 0

    # Initialize test components
    runner = ExampleTestRunner(project_root, config)
    validator = ValidationSuite(config, clean_iris_mode=args.clean_iris)

    print(f"\n🧪 Running {len(examples)} example tests in {args.mode} mode...")

    # Execute tests
    results = []
    for i, example in enumerate(examples, 1):
        if not args.quiet:
            print(f"\n[{i}/{len(examples)}] Testing: {example}")

        # Get timeout for this example
        example_config = config.get("examples", {}).get(example, {})
        timeout = args.timeout or example_config.get("timeout", 300)

        # Run the test
        result = runner.run_example(example, timeout=timeout, mode=args.mode)

        # Validate output if successful
        if result.success and result.stdout:
            try:
                validation = validator.validate_example_output(
                    example,
                    result.stdout,
                    performance_metrics={
                        "execution_time": result.execution_time,
                        "peak_memory_mb": result.peak_memory_mb,
                        "avg_cpu_percent": result.avg_cpu_percent,
                    },
                )
                result.validation_results = validation.to_dict()
            except Exception as e:
                logging.warning(f"Validation failed for {example}: {e}")

        results.append((example, result))

        # Print immediate feedback
        if not args.quiet:
            print_result_summary(example, result, args.verbose)

        # Handle failures
        if not result.success and not args.continue_on_failure:
            print(f"\n❌ Stopping due to failure in {example}")
            print(f"Use --continue-on-failure to continue testing after failures")
            break

    # Generate comprehensive report
    if not args.quiet:
        runner.generate_report([r for _, r in results])

    # Show final summary and determine exit code
    exit_code = generate_final_report(results, config, args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
