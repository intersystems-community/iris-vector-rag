#!/usr/bin/env python3
"""
Python utility to run any test with 1000+ documents.

This is a pure Python replacement for shell scripts, allowing better
integration with the project's testing infrastructure.
"""

import os
import sys
import argparse
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_pytest_1000_docs")

def main():
    """
    Main entry point for running tests with 1000+ documents.
    
    This utility sets up the environment to use the conftest_1000docs.py
    fixture and runs the specified pytest tests.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run pytest tests with 1000+ documents fixture"
    )
    parser.add_argument(
        "test_paths", 
        nargs="*",
        default=["tests/test_all_with_1000_docs.py"],
        help="Test files or directories to run (default: tests/test_all_with_1000_docs.py)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase verbosity"
    )
    parser.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Exit on first failure"
    )
    parser.add_argument(
        "-s", "--capture-no",
        action="store_true",
        help="Disable output capture"
    )
    parser.add_argument(
        "-m", "--marker",
        help="Only run tests matching the given marker expression"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTEST_CONFTEST_PATH"] = "tests/conftest_1000docs.py"
    
    # Build pytest command
    pytest_args = ["python", "-m", "pytest", "--confcutdir=tests", 
                   "-c", "tests/conftest_1000docs.py"]
    
    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")
        
    # Add exitfirst
    if args.exitfirst:
        pytest_args.append("-x")
        
    # Add capture-no
    if args.capture_no:
        pytest_args.append("-s")
        
    # Add marker if specified
    if args.marker:
        pytest_args.extend(["-m", args.marker])
        
    # Add test paths
    pytest_args.extend(args.test_paths)
    
    # Print command for visibility
    logger.info("=" * 70)
    logger.info("Running tests with 1000+ documents")
    logger.info("=" * 70)
    logger.info(f"Command: {' '.join(pytest_args)}")
    logger.info(f"Using conftest: {os.environ['PYTEST_CONFTEST_PATH']}")
    
    # Execute pytest
    try:
        result = subprocess.run(pytest_args, check=False)
        exit_code = result.returncode
        
        if exit_code == 0:
            logger.info("✅ All tests passed with 1000+ documents")
        else:
            logger.error(f"❌ Tests failed with exit code: {exit_code}")
            
        return exit_code
    except Exception as e:
        logger.error(f"Error running pytest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
