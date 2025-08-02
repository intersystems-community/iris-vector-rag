#!/usr/bin/env python3
"""
Test runner for Docker-compose integration tests.

This script provides a convenient way to run Docker-compose integration tests
with various options and configurations.
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path
from typing import List, Optional


def run_tests(
    test_pattern: Optional[str] = None,
    verbose: bool = False,
    coverage: bool = False,
    parallel: bool = False,
    profile: Optional[str] = None,
    fail_fast: bool = False,
    markers: Optional[str] = None
) -> int:
    """
    Run Docker-compose integration tests with specified options.
    
    Args:
        test_pattern: Specific test pattern to run
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        parallel: Run tests in parallel
        profile: Run tests for specific profile only
        fail_fast: Stop on first failure
        markers: Pytest markers to filter tests
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test file
    test_file = Path(__file__).parent / "test_docker_compose_integration.py"
    cmd.append(str(test_file))
    
    # Add specific test pattern if provided
    if test_pattern:
        cmd.append(f"::{test_pattern}")
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage options
    if coverage:
        cmd.extend([
            "--cov=quick_start.docker",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add fail fast
    if fail_fast:
        cmd.append("-x")
    
    # Add markers
    if markers:
        cmd.extend(["-m", markers])
    
    # Add profile-specific filtering
    if profile:
        cmd.extend(["-k", f"{profile}_profile"])
    
    # Set environment variables for testing
    env = os.environ.copy()
    env["TESTING"] = "true"
    env["DOCKER_COMPOSE_TEST_MODE"] = "true"
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print("-" * 50)
    
    # Execute tests
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_test_categories() -> int:
    """Run tests by category for better organization."""
    categories = [
        ("Docker-compose File Generation", "test_docker_compose_file_generation"),
        ("Container Configuration", "test_.*_container_configuration"),
        ("Service Dependencies", "test_service_dependencies"),
        ("Volume and Network", "test_volume_and_network"),
        ("Environment Variables", "test_environment_variable"),
        ("Health Checks", "test_health_checks"),
        ("Integration Tests", "test_integration_with"),
        ("Development Workflow", "test_development_mode or test_hot_reload or test_debug"),
        ("Production Deployment", "test_production_mode or test_ssl or test_backup"),
        ("Scaling and Resources", "test_scaling or test_load_balancer or test_auto_scaling"),
        ("Docker Operations", "test_docker_compose_up or test_docker_compose_down"),
        ("Error Handling", "test_invalid or test_missing or test_port_conflict"),
        ("End-to-End Workflows", "test_complete_docker_workflow")
    ]
    
    print("Running Docker-compose integration tests by category...")
    print("=" * 60)
    
    total_failures = 0
    
    for category_name, test_pattern in categories:
        print(f"\n🧪 Running {category_name} Tests...")
        print("-" * 40)
        
        result = run_tests(
            test_pattern=None,
            verbose=True,
            coverage=False,
            parallel=False,
            fail_fast=False,
            markers=None
        )
        
        if result != 0:
            print(f"❌ {category_name} tests failed")
            total_failures += 1
        else:
            print(f"✅ {category_name} tests passed")
    
    print("\n" + "=" * 60)
    if total_failures == 0:
        print("🎉 All test categories passed!")
        return 0
    else:
        print(f"❌ {total_failures} test categories failed")
        return 1


def run_profile_tests() -> int:
    """Run tests for each Docker profile."""
    profiles = ["minimal", "standard", "extended", "development", "production", "testing"]
    
    print("Running Docker-compose tests for each profile...")
    print("=" * 50)
    
    total_failures = 0
    
    for profile in profiles:
        print(f"\n🐳 Running {profile.title()} Profile Tests...")
        print("-" * 30)
        
        result = run_tests(
            profile=profile,
            verbose=True,
            coverage=False,
            parallel=False,
            fail_fast=False
        )
        
        if result != 0:
            print(f"❌ {profile} profile tests failed")
            total_failures += 1
        else:
            print(f"✅ {profile} profile tests passed")
    
    print("\n" + "=" * 50)
    if total_failures == 0:
        print("🎉 All profile tests passed!")
        return 0
    else:
        print(f"❌ {total_failures} profile tests failed")
        return 1


def check_prerequisites() -> bool:
    """Check if prerequisites for running tests are met."""
    print("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required")
        return False
    print("✅ Python version OK")
    
    # Check pytest installation
    try:
        import pytest
        print("✅ pytest available")
    except ImportError:
        print("❌ pytest not installed")
        return False
    
    # Check Docker availability (optional for mocked tests)
    try:
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Docker available")
        else:
            print("⚠️  Docker not available (tests will use mocks)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Docker not available (tests will use mocks)")
    
    # Check docker-compose availability (optional for mocked tests)
    try:
        result = subprocess.run(
            ["docker-compose", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            print("✅ docker-compose available")
        else:
            print("⚠️  docker-compose not available (tests will use mocks)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  docker-compose not available (tests will use mocks)")
    
    return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run Docker-compose integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_docker_compose_tests.py
  
  # Run with coverage
  python run_docker_compose_tests.py --coverage
  
  # Run specific test
  python run_docker_compose_tests.py --test test_docker_compose_file_generation_minimal
  
  # Run tests for specific profile
  python run_docker_compose_tests.py --profile minimal
  
  # Run by category
  python run_docker_compose_tests.py --by-category
  
  # Run profile tests
  python run_docker_compose_tests.py --by-profile
  
  # Run in parallel with verbose output
  python run_docker_compose_tests.py --parallel --verbose
        """
    )
    
    parser.add_argument(
        "--test", "-t",
        help="Specific test pattern to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--profile",
        choices=["minimal", "standard", "extended", "development", "production", "testing"],
        help="Run tests for specific profile only"
    )
    
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Pytest markers to filter tests"
    )
    
    parser.add_argument(
        "--by-category",
        action="store_true",
        help="Run tests organized by category"
    )
    
    parser.add_argument(
        "--by-profile",
        action="store_true",
        help="Run tests for each profile separately"
    )
    
    parser.add_argument(
        "--check-prereqs",
        action="store_true",
        help="Check prerequisites and exit"
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if args.check_prereqs:
        if check_prerequisites():
            print("\n✅ All prerequisites met")
            return 0
        else:
            print("\n❌ Prerequisites not met")
            return 1
    
    if not check_prerequisites():
        print("\n⚠️  Some prerequisites missing, but tests may still work with mocks")
    
    # Run tests based on arguments
    if args.by_category:
        return run_specific_test_categories()
    elif args.by_profile:
        return run_profile_tests()
    else:
        return run_tests(
            test_pattern=args.test,
            verbose=args.verbose,
            coverage=args.coverage,
            parallel=args.parallel,
            profile=args.profile,
            fail_fast=args.fail_fast,
            markers=args.markers
        )


if __name__ == "__main__":
    sys.exit(main())