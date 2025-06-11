#!/usr/bin/env python
"""
End-to-End Test Runner for RAG Templates

This script runs end-to-end tests with real PMC data, following TDD principles
outlined in the project's .clinerules file. It:

1. Checks if the IRIS Docker container is running and starts it if needed
2. Verifies the database has been initialized with real PMC data (at least 1000 documents)
3. Runs the verify_real_data_testing.py script to ensure real data is available
4. Runs the end-to-end tests with pytest
5. Generates test reports in both JSON and HTML formats
6. Logs detailed information about the test execution

The script supports command-line arguments for:
- Specific tests to run
- Number of documents to use
- Output directory for test reports
- Verbose mode for detailed logging
- LLM provider selection (openai, anthropic, etc.)

It also includes error handling and recovery mechanisms for:
- Database connection issues
- Missing data
- Test failures
- Real data verification failures

Usage:
    python scripts/run_e2e_tests.py [options]

Example:
    python scripts/run_e2e_tests.py --test test_basic_rag_with_real_data --min-docs 1500 --output-dir ./test_reports --verbose --llm-provider openai
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import importlib.util

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
try:
    from common.iris_connector import get_iris_connection, IRISConnectionError
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Configure logging
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging with appropriate level based on verbose flag."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger("run_e2e_tests")
    logger.setLevel(log_level)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

# Docker container management
def check_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_iris_container_running() -> bool:
    """Check if the IRIS container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=iris_db", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return "iris_db" in result.stdout
    except Exception:
        return False

def start_iris_container(logger: logging.Logger) -> bool:
    """Start the IRIS container using docker-compose."""
    logger.info("Starting IRIS container...")
    
    try:
        # Use the iris-only compose file as specified in the Makefile
        compose_file = "docker-compose.iris-only.yml"
        
        # Run docker-compose up
        result = subprocess.run(
            ["docker-compose", "-f", compose_file, "up", "-d", "--wait", "iris_db"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to start IRIS container: {result.stderr}")
            return False
        
        # Wait for container to be fully initialized
        logger.info("IRIS container started. Waiting for initialization...")
        time.sleep(15)  # Same wait time as in the Makefile
        
        # Verify container is running
        if not check_iris_container_running():
            logger.error("IRIS container failed to start properly.")
            return False
        
        logger.info("IRIS container is now running.")
        return True
    
    except Exception as e:
        logger.error(f"Error starting IRIS container: {e}")
        return False

# Database verification
def verify_database_initialized(logger: logging.Logger, min_docs: int = 1000) -> bool:
    """
    Verify that the database has been initialized with schema and contains
    at least the minimum number of documents.
    
    This is a basic check that will be followed by a more thorough verification
    using the verify_real_data_testing.py script.
    """
    logger.info(f"Performing basic verification that database has at least {min_docs} documents...")
    
    try:
        # Get connection to IRIS
        connection = get_iris_connection()
        if not connection:
            logger.error("Failed to connect to IRIS database.")
            return False
        
        try:
            # Check document count
            with connection.cursor() as cursor:
                try:
                    # Try with RAG schema qualification first (as in conftest_real_pmc.py)
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
                    count = cursor.fetchone()[0]
                    logger.info(f"Found {count} documents in RAG.SourceDocuments_V2.")
                except Exception:
                    try:
                        # Try without schema qualification
                        cursor.execute("SELECT COUNT(*) FROM SourceDocuments_V2")
                        count = cursor.fetchone()[0]
                        logger.info(f"Found {count} documents in SourceDocuments.")
                    except Exception as e:
                        logger.error(f"Error querying document count: {e}")
                        logger.error("Database schema may not be initialized.")
                        return False
            
            # Check if we have enough documents
            if count < min_docs:
                logger.error(f"Insufficient documents: found {count}, need at least {min_docs}.")
                return False
            
            logger.info(f"✅ Basic database verification passed: {count} documents available.")
            return True
        
        finally:
            # Close connection
            connection.close()
    
    except IRISConnectionError as e:
        logger.error(f"IRIS connection error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error verifying database: {e}")
        return False

def initialize_database_if_needed(logger: logging.Logger) -> bool:
    """Initialize the database schema if needed."""
    logger.info("Checking if database schema needs to be initialized...")
    
    try:
        # Try to connect and check if tables exist
        connection = get_iris_connection()
        if not connection:
            logger.error("Failed to connect to IRIS database.")
            return False
        
        try:
            with connection.cursor() as cursor:
                try:
                    # Check if SourceDocuments table exists
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
                    logger.info("Database schema already initialized.")
                    return True
                except Exception:
                    logger.info("Database schema not initialized. Initializing now...")
                    
                    # Run the database initialization script
                    result = subprocess.run(
                        ["python", "run_db_init_local.py", "--force-recreate"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"Failed to initialize database: {result.stderr}")
                        return False
                    
                    logger.info("Database schema initialized successfully.")
                    return True
        
        finally:
            connection.close()
    
    except Exception as e:
        logger.error(f"Error checking/initializing database: {e}")
        return False

def load_pmc_data_if_needed(logger: logging.Logger, min_docs: int = 1000) -> bool:
    """Load PMC data if the database doesn't have enough documents."""
    # First check if we already have enough documents
    try:
        connection = get_iris_connection()
        if not connection:
            logger.error("Failed to connect to IRIS database.")
            return False
        
        try:
            with connection.cursor() as cursor:
                try:
                    # Try with RAG schema qualification first
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
                    count = cursor.fetchone()[0]
                except Exception:
                    try:
                        # Try without schema qualification
                        cursor.execute("SELECT COUNT(*) FROM SourceDocuments_V2")
                        count = cursor.fetchone()[0]
                    except Exception:
                        logger.error("Error querying document count.")
                        return False
            
            # If we already have enough documents, we're done
            if count >= min_docs:
                logger.info(f"✅ Database already has {count} documents (>= {min_docs} required).")
                return True
            
            # Otherwise, load more data
            logger.info(f"Database has only {count} documents. Loading more data to reach {min_docs}...")
            
            # Run the data loading script
            # Use a slightly higher limit to ensure we meet the minimum
            limit = min_docs + 100
            
            result = subprocess.run(
                ["python", "scripts_to_review/load_pmc_data.py", "--limit", str(limit), "--load-colbert"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to load PMC data: {result.stderr}")
                return False
            
            # Verify we now have enough documents
            with connection.cursor() as cursor:
                try:
                    # Try with RAG schema qualification first
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
                    new_count = cursor.fetchone()[0]
                except Exception:
                    try:
                        # Try without schema qualification
                        cursor.execute("SELECT COUNT(*) FROM SourceDocuments_V2")
                        new_count = cursor.fetchone()[0]
                    except Exception:
                        logger.error("Error querying document count after loading.")
                        return False
            
            if new_count >= min_docs:
                logger.info(f"✅ Successfully loaded data. Now have {new_count} documents.")
                return True
            else:
                logger.error(f"Failed to load enough documents. Only have {new_count} (need {min_docs}).")
                return False
        
        finally:
            connection.close()
    
    except Exception as e:
        logger.error(f"Error loading PMC data: {e}")
        return False

# Test execution
def verify_real_data(logger: logging.Logger, min_docs: int = 1000, output_dir: str = "test_results", verbose: bool = False) -> bool:
    """
    Run the verify_real_data_testing.py script to perform thorough verification of real data.
    
    Args:
        logger: Logger instance
        min_docs: Minimum number of documents required
        output_dir: Directory for test reports
        verbose: Whether to use verbose output
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Running real data verification script...")
    
    # Build command
    cmd = [
        "python",
        "scripts/verify_real_data_testing.py",
        f"--min-docs={min_docs}",
        f"--output-dir={output_dir}"
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the script
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Log output
        if verbose:
            logger.debug(f"Verification script stdout:\n{result.stdout}")
        
        if result.stderr:
            logger.warning(f"Verification script stderr:\n{result.stderr}")
        
        # Check result
        if result.returncode == 0:
            logger.info("✅ Real data verification passed.")
            return True
        else:
            logger.error(f"❌ Real data verification failed with return code {result.returncode}.")
            return False
    
    except Exception as e:
        logger.error(f"Error running real data verification script: {e}")
        return False

def run_e2e_tests(
    logger: logging.Logger,
    test_name: Optional[str] = None,
    output_dir: str = "test_results",
    verbose: bool = False,
    llm_provider: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Run the end-to-end tests with pytest.
    
    Args:
        logger: Logger instance
        test_name: Specific test to run (optional)
        output_dir: Directory for test reports
        verbose: Whether to use verbose output
        llm_provider: LLM provider to use (optional)
    
    Returns:
        Tuple of (success, json_report_path, html_report_path)
    """
    logger.info("Running end-to-end tests...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for report filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_report = os.path.join(output_dir, f"e2e_test_report_{timestamp}.json")
    html_report = os.path.join(output_dir, f"e2e_test_report_{timestamp}.html")
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add test file or specific test
    if test_name:
        if ":" in test_name:
            # If test_name includes a specific test function (e.g., "test_file.py::test_func")
            cmd.append(test_name)
        else:
            # If test_name is just a test function name, find it in the e2e test file
            cmd.append(f"tests/test_e2e_rag_pipelines.py::{test_name}")
    else:
        # Run all tests in the e2e test file
        cmd.append("tests/test_e2e_rag_pipelines.py")
    
    # Add verbosity flag
    if verbose:
        cmd.append("-v")
    
    # Add report generation flags
    cmd.extend([
        "--json-report",
        f"--json-report-file={json_report}",
        "--html", html_report,
        "--self-contained-html"
    ])
    
    # Add LLM provider environment variable if specified
    env = os.environ.copy()
    if llm_provider:
        env["LLM_PROVIDER"] = llm_provider
        logger.info(f"Using LLM provider: {llm_provider}")
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run pytest
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env
        )
        
        # Log output
        if verbose:
            logger.debug(f"Pytest stdout:\n{result.stdout}")
        
        if result.stderr:
            logger.warning(f"Pytest stderr:\n{result.stderr}")
        
        # Check result
        if result.returncode == 0:
            logger.info("✅ End-to-end tests passed successfully.")
            return True, json_report, html_report
        else:
            logger.error(f"❌ End-to-end tests failed with return code {result.returncode}.")
            return False, json_report, html_report
    
    except Exception as e:
        logger.error(f"Error running end-to-end tests: {e}")
        return False, "", ""

def generate_test_summary(logger: logging.Logger, json_report_path: str) -> Dict[str, Any]:
    """Generate a summary of test results from the JSON report."""
    try:
        if not os.path.exists(json_report_path):
            logger.error(f"JSON report file not found: {json_report_path}")
            return {}
        
        with open(json_report_path, 'r') as f:
            report_data = json.load(f)
        
        summary = {
            "total": report_data.get("summary", {}).get("total", 0),
            "passed": report_data.get("summary", {}).get("passed", 0),
            "failed": report_data.get("summary", {}).get("failed", 0),
            "skipped": report_data.get("summary", {}).get("skipped", 0),
            "error": report_data.get("summary", {}).get("error", 0),
            "duration": report_data.get("duration", 0),
            "tests": []
        }
        
        # Extract test details
        for test_id, test_data in report_data.get("tests", {}).items():
            test_info = {
                "name": test_data.get("name", ""),
                "outcome": test_data.get("outcome", ""),
                "duration": test_data.get("duration", 0),
                "message": test_data.get("call", {}).get("longrepr", "") if test_data.get("outcome") == "failed" else ""
            }
            summary["tests"].append(test_info)
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating test summary: {e}")
        return {}

def display_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Display a progress bar in the console."""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

def main():
    """Main function to run the end-to-end tests."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run end-to-end tests with real PMC data.")
    parser.add_argument("--test", type=str, help="Specific test to run (e.g., test_basic_rag_with_real_data)")
    parser.add_argument("--min-docs", type=int, default=1000, help="Minimum number of documents required")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Directory for test reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-docker-check", action="store_true", help="Skip Docker container check")
    parser.add_argument("--skip-verification", action="store_true", help="Skip real data verification")
    parser.add_argument("--llm-provider", type=str, choices=["openai", "anthropic", "azure", "stub"],
                        help="LLM provider to use for tests")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    logger.info("Starting end-to-end test runner...")
    
    # Track overall success
    success = True
    
    # Step 1: Check if Docker is running
    if not args.skip_docker_check:
        logger.info("Step 1: Checking Docker status...")
        if not check_docker_running():
            logger.error("Docker daemon is not running. Please start Docker and try again.")
            return 1
        
        # Step 2: Check if IRIS container is running
        logger.info("Step 2: Checking IRIS container status...")
        if not check_iris_container_running():
            logger.warning("IRIS container is not running. Attempting to start it...")
            if not start_iris_container(logger):
                logger.error("Failed to start IRIS container. Please start it manually and try again.")
                logger.info("You can start the IRIS container with: make start-iris")
                return 1
    else:
        logger.info("Skipping Docker checks as requested.")
    
    # Step 3: Verify database is initialized
    logger.info("Step 3: Verifying database initialization...")
    if not initialize_database_if_needed(logger):
        logger.error("Failed to initialize database. Please initialize it manually and try again.")
        logger.info("You can initialize the database with: make init-db")
        success = False
    
    # Step 4: Verify database has enough documents
    if success:
        logger.info(f"Step 4: Verifying database has at least {args.min_docs} documents...")
        if not verify_database_initialized(logger, args.min_docs):
            logger.warning("Database doesn't have enough documents. Attempting to load more...")
            if not load_pmc_data_if_needed(logger, args.min_docs):
                logger.error("Failed to load enough PMC data. Please load data manually and try again.")
                logger.info("You can load data with: make load-data")
                success = False
    
    # Step 5: Run real data verification
    if success and not args.skip_verification:
        logger.info("Step 5: Running real data verification...")
        if not verify_real_data(logger, args.min_docs, args.output_dir, args.verbose):
            logger.error("Real data verification failed. Tests may not run correctly with real data.")
            logger.info("You can skip this verification with: --skip-verification")
            success = False
    elif args.skip_verification:
        logger.warning("Skipping real data verification as requested.")
    
    # Step 6: Run the end-to-end tests
    if success:
        logger.info("Step 6: Running end-to-end tests...")
        tests_passed, json_report, html_report = run_e2e_tests(
            logger, 
            test_name=args.test,
            output_dir=args.output_dir,
            verbose=args.verbose,
            llm_provider=args.llm_provider
        )
        
        if not tests_passed:
            logger.error("End-to-end tests failed.")
            success = False
        
        # Step 7: Generate and display test summary
        if json_report and os.path.exists(json_report):
            logger.info("Step 7: Generating test summary...")
            summary = generate_test_summary(logger, json_report)
            
            if summary:
                print("\n" + "=" * 80)
                print(f"TEST SUMMARY")
                print("=" * 80)
                print(f"Total tests: {summary['total']}")
                print(f"Passed: {summary['passed']}")
                print(f"Failed: {summary['failed']}")
                print(f"Skipped: {summary['skipped']}")
                print(f"Errors: {summary['error']}")
                print(f"Duration: {summary['duration']:.2f} seconds")
                print("-" * 80)
                
                # Print details of each test
                for test in summary['tests']:
                    status = "✅" if test['outcome'] == 'passed' else "❌"
                    print(f"{status} {test['name']} ({test['duration']:.2f}s)")
                    if test['outcome'] == 'failed' and test['message']:
                        print(f"   Error: {test['message'][:100]}...")
                
                print("-" * 80)
                print(f"Test reports saved to:")
                print(f"  - JSON: {json_report}")
                print(f"  - HTML: {html_report}")
                print("=" * 80)
    
    # Final status
    if success:
        logger.info("✅ All steps completed successfully.")
        return 0
    else:
        logger.error("❌ Some steps failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())