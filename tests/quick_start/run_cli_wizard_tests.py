#!/usr/bin/env python3
"""
Test runner script for Quick Start CLI wizard tests.

This script provides a convenient way to run CLI wizard tests with various options,
following TDD principles and ensuring comprehensive test coverage.

Usage:
    python tests/quick_start/run_cli_wizard_tests.py [options]

Examples:
    # Run all CLI wizard tests
    python tests/quick_start/run_cli_wizard_tests.py

    # Run tests with coverage report
    python tests/quick_start/run_cli_wizard_tests.py --coverage

    # Run specific test category
    python tests/quick_start/run_cli_wizard_tests.py --category profile_selection

    # Run in TDD mode (expect failures)
    python tests/quick_start/run_cli_wizard_tests.py --tdd

    # Run with verbose output
    python tests/quick_start/run_cli_wizard_tests.py --verbose

    # Generate HTML coverage report
    python tests/quick_start/run_cli_wizard_tests.py --coverage --html
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional


class CLIWizardTestRunner:
    """Test runner for CLI wizard tests."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent
        self.test_files = {
            "main": "test_cli_wizard.py",
            "fixtures": "test_cli_wizard_fixtures.py"
        }
        
        self.test_categories = {
            "profile_selection": [
                "test_profile_selection_interactive_minimal",
                "test_profile_selection_interactive_standard", 
                "test_profile_selection_interactive_extended",
                "test_profile_selection_interactive_custom",
                "test_profile_selection_non_interactive_minimal",
                "test_profile_selection_non_interactive_with_overrides",
                "test_profile_selection_invalid_profile",
                "test_profile_characteristics_display"
            ],
            "environment_config": [
                "test_database_connection_prompts",
                "test_llm_provider_configuration",
                "test_embedding_model_selection",
                "test_environment_variable_generation",
                "test_environment_configuration_validation"
            ],
            "template_generation": [
                "test_configuration_file_generation",
                "test_env_file_creation",
                "test_docker_compose_generation",
                "test_sample_data_script_generation",
                "test_file_validation_and_error_handling"
            ],
            "validation_integration": [
                "test_database_connectivity_validation",
                "test_llm_provider_credential_validation",
                "test_embedding_model_availability_check",
                "test_system_health_check_integration",
                "test_error_reporting_and_recovery"
            ],
            "cli_interface": [
                "test_command_line_argument_parsing",
                "test_interactive_prompt_handling",
                "test_output_formatting_and_display",
                "test_progress_indicators_and_status_updates",
                "test_cli_error_handling_and_user_feedback"
            ],
            "integration": [
                "test_integration_with_template_engine",
                "test_integration_with_schema_validator",
                "test_integration_with_integration_factory",
                "test_integration_with_sample_manager",
                "test_end_to_end_integration_workflow"
            ],
            "error_handling": [
                "test_invalid_profile_name_handling",
                "test_missing_required_parameters",
                "test_network_connectivity_issues",
                "test_file_permission_errors",
                "test_disk_space_validation",
                "test_concurrent_wizard_instances",
                "test_interrupted_wizard_recovery"
            ],
            "end_to_end": [
                "test_complete_minimal_profile_workflow",
                "test_complete_standard_profile_workflow",
                "test_complete_extended_profile_workflow",
                "test_non_interactive_complete_workflow",
                "test_wizard_help_and_list_commands",
                "test_wizard_configuration_validation_workflow",
                "test_wizard_with_environment_variable_overrides",
                "test_wizard_cleanup_on_failure",
                "test_wizard_progress_tracking_and_cancellation"
            ],
            "utilities": [
                "test_profile_comparison_utility",
                "test_resource_estimation_utility",
                "test_configuration_diff_utility",
                "test_backup_and_restore_utilities"
            ],
            "scenarios": [
                "test_development_environment_setup",
                "test_production_environment_setup",
                "test_migration_from_existing_setup",
                "test_multi_tenant_setup"
            ]
        }
    
    def run_tests(self, 
                  category: Optional[str] = None,
                  coverage: bool = False,
                  html_coverage: bool = False,
                  verbose: bool = False,
                  tdd_mode: bool = False,
                  specific_test: Optional[str] = None,
                  fail_fast: bool = False,
                  markers: Optional[str] = None) -> int:
        """Run CLI wizard tests with specified options."""
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test files
        if specific_test:
            cmd.append(f"{self.test_dir}/{self.test_files['main']}::{specific_test}")
        elif category:
            if category not in self.test_categories:
                print(f"Error: Unknown test category '{category}'")
                print(f"Available categories: {', '.join(self.test_categories.keys())}")
                return 1
            
            # Add specific tests for category
            for test_name in self.test_categories[category]:
                cmd.append(f"{self.test_dir}/{self.test_files['main']}::TestQuickStartCLIWizard::{test_name}")
        else:
            # Run all CLI wizard tests
            cmd.append(str(self.test_dir / self.test_files['main']))
        
        # Add pytest options
        if verbose:
            cmd.append("-v")
        
        if fail_fast:
            cmd.append("-x")
        
        if tdd_mode:
            cmd.extend(["--tb=short", "-v"])
            print("Running in TDD mode - tests are expected to fail initially")
        
        if markers:
            cmd.extend(["-m", markers])
        
        # Add coverage options
        if coverage:
            cmd.extend([
                "--cov=quick_start.cli",
                "--cov-report=term-missing"
            ])
            
            if html_coverage:
                cmd.extend([
                    "--cov-report=html:htmlcov/cli_wizard",
                    "--cov-report=xml:coverage_cli_wizard.xml"
                ])
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        
        # Run tests
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {self.project_root}")
        print("-" * 80)
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, env=env)
            return result.returncode
        except KeyboardInterrupt:
            print("\nTest execution interrupted by user")
            return 130
        except Exception as e:
            print(f"Error running tests: {e}")
            return 1
    
    def list_categories(self):
        """List available test categories."""
        print("Available test categories:")
        print("-" * 40)
        for category, tests in self.test_categories.items():
            print(f"{category}:")
            for test in tests:
                print(f"  - {test}")
            print()
    
    def list_tests(self, category: Optional[str] = None):
        """List available tests."""
        if category:
            if category not in self.test_categories:
                print(f"Error: Unknown category '{category}'")
                return
            
            print(f"Tests in category '{category}':")
            for test in self.test_categories[category]:
                print(f"  - {test}")
        else:
            print("All available tests:")
            for category, tests in self.test_categories.items():
                print(f"\n{category}:")
                for test in tests:
                    print(f"  - {test}")
    
    def validate_environment(self) -> bool:
        """Validate test environment setup."""
        print("Validating test environment...")
        
        # Check if pytest is available
        try:
            subprocess.run(["python", "-m", "pytest", "--version"], 
                         capture_output=True, check=True)
            print("✓ pytest is available")
        except subprocess.CalledProcessError:
            print("✗ pytest is not available - install with: pip install pytest")
            return False
        
        # Check if test files exist
        main_test_file = self.test_dir / self.test_files['main']
        if not main_test_file.exists():
            print(f"✗ Main test file not found: {main_test_file}")
            return False
        print(f"✓ Main test file found: {main_test_file}")
        
        fixtures_file = self.test_dir / self.test_files['fixtures']
        if not fixtures_file.exists():
            print(f"✗ Fixtures file not found: {fixtures_file}")
            return False
        print(f"✓ Fixtures file found: {fixtures_file}")
        
        # Check if Quick Start modules are importable
        try:
            sys.path.insert(0, str(self.project_root))
            import quick_start.config.template_engine
            import quick_start.config.schema_validator
            import quick_start.config.integration_factory
            import quick_start.data.sample_manager
            print("✓ Quick Start modules are importable")
        except ImportError as e:
            print(f"✗ Quick Start modules not importable: {e}")
            return False
        
        print("✓ Test environment validation passed")
        return True
    
    def run_tdd_cycle(self, test_name: str):
        """Run a specific test in TDD cycle mode."""
        print(f"Running TDD cycle for test: {test_name}")
        print("=" * 60)
        
        # RED phase - run test (should fail)
        print("RED PHASE: Running test (expecting failure)...")
        red_result = self.run_tests(specific_test=test_name, tdd_mode=True)
        
        if red_result == 0:
            print("⚠️  Test passed in RED phase - this may indicate the test is not properly written")
        else:
            print("✓ Test failed as expected in RED phase")
        
        print("\nNow implement the minimal code to make this test pass (GREEN phase)")
        print("Then refactor the code while keeping the test passing (REFACTOR phase)")
        
        return red_result


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run Quick Start CLI wizard tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--category", "-c",
        choices=["profile_selection", "environment_config", "template_generation",
                "validation_integration", "cli_interface", "integration", 
                "error_handling", "end_to_end", "utilities", "scenarios"],
        help="Run tests for specific category"
    )
    
    parser.add_argument(
        "--test", "-t",
        help="Run specific test by name"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (requires --coverage)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--tdd",
        action="store_true",
        help="Run in TDD mode (expect failures)"
    )
    
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Run tests with specific markers"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories"
    )
    
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests"
    )
    
    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validate test environment"
    )
    
    parser.add_argument(
        "--tdd-cycle",
        help="Run specific test in TDD cycle mode"
    )
    
    args = parser.parse_args()
    
    runner = CLIWizardTestRunner()
    
    # Handle special commands
    if args.list_categories:
        runner.list_categories()
        return 0
    
    if args.list_tests:
        runner.list_tests(args.category)
        return 0
    
    if args.validate_env:
        if runner.validate_environment():
            return 0
        else:
            return 1
    
    if args.tdd_cycle:
        return runner.run_tdd_cycle(args.tdd_cycle)
    
    # Validate environment before running tests
    if not runner.validate_environment():
        return 1
    
    # Run tests
    return runner.run_tests(
        category=args.category,
        coverage=args.coverage,
        html_coverage=args.html,
        verbose=args.verbose,
        tdd_mode=args.tdd,
        specific_test=args.test,
        fail_fast=args.fail_fast,
        markers=args.markers
    )


if __name__ == "__main__":
    sys.exit(main())