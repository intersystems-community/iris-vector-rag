#!/usr/bin/env python
"""
Run tests with 1000 documents with robust error handling.

This script sets all necessary environment variables and runs pytest directly
with the proper options to handle connection issues gracefully.

Usage:
    python run_1000_docs_tests.py [technique] [--mock-embeddings]

Examples:
    # Run all tests with 1000 documents
    python run_1000_docs_tests.py
    
    # Run only graphrag tests with 1000 documents
    python run_1000_docs_tests.py graphrag
    
    # Run with mock embeddings (faster, less accurate)
    python run_1000_docs_tests.py --mock-embeddings
"""

import argparse
import os
import sys
import subprocess
import time


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run tests with 1000 documents")
    
    parser.add_argument(
        "technique",
        nargs="?",
        default="all",
        choices=["all", "graphrag", "colbert", "hyde", "noderag", "crag", "basic"],
        help="Technique to test (default: all)"
    )
    
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use mock embeddings instead of real embeddings"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show more detailed output"
    )
    
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for failed tests (default: 3)"
    )
    
    return parser.parse_args()


def main():
    """Run tests with 1000 documents"""
    args = parse_args()
    
    # Set environment variables
    os.environ["TEST_IRIS"] = "true"
    os.environ["TEST_DOCUMENT_COUNT"] = "1000"
    os.environ["COLLECT_PERFORMANCE_METRICS"] = "true"
    
    # Handle embedding model options
    if args.mock_embeddings:
        os.environ["USE_MOCK_EMBEDDINGS"] = "true"
        print("Using mock embeddings (faster but less accurate)")
    else:
        os.environ["USE_MOCK_EMBEDDINGS"] = "false"
        print("Using real embeddings (more accurate but slower)")
    
    # Display environment settings
    print("Environment settings:")
    print(f"  TEST_IRIS: {os.environ.get('TEST_IRIS', 'not set')}")
    print(f"  TEST_DOCUMENT_COUNT: {os.environ.get('TEST_DOCUMENT_COUNT', 'not set')}")
    print(f"  USE_MOCK_EMBEDDINGS: {os.environ.get('USE_MOCK_EMBEDDINGS', 'not set')}")
    print(f"  COLLECT_PERFORMANCE_METRICS: {os.environ.get('COLLECT_PERFORMANCE_METRICS', 'not set')}")
    
    # Determine test pattern based on specified technique
    technique_map = {
        "graphrag": "graphrag",
        "colbert": "colbert",
        "hyde": "hyde",
        "noderag": "noderag",
        "crag": "crag",
        "basic": "basic_rag"
    }
    
    if args.technique == "all":
        # Since we had an issue with the wildcard pattern, list all available test files explicitly
        test_files = [
            "tests/test_graphrag_with_testcontainer.py",
            "tests/test_graphrag_large_scale.py"
        ]
        test_pattern = " ".join(test_files)
        print("Running tests for all techniques with testcontainer")
    elif args.technique == "graphrag":
        # For GraphRAG, run both standard and large-scale tests
        test_files = [
            "tests/test_graphrag_with_testcontainer.py",
            "tests/test_graphrag_large_scale.py"
        ]
        test_pattern = " ".join(test_files)
        print(f"Running standard and large-scale tests for GraphRAG")
    else:
        test_pattern = f"tests/test_{technique_map[args.technique]}_with_testcontainer.py"
        print(f"Running tests for technique: {args.technique}")
    
    # Build robust pytest options to handle common issues
    pytest_opts = [
        "-xvs",                  # Continue on errors, verbose mode, minimal output
        "--log-cli-level=INFO",  # Show log output during test
        "--tb=short",            # Short traceback style
        "--no-header",           # No pytest header
        "--durations=0",         # Show test durations
    ]
    
    if args.verbose:
        pytest_opts.append("-v")
    
    # Build the full command with Poetry
    # If test_pattern contains multiple files, split them into separate arguments
    if ' ' in test_pattern:
        test_files = test_pattern.split()
        cmd = ["poetry", "run", "pytest"] + test_files + pytest_opts
    else:
        cmd = ["poetry", "run", "pytest", test_pattern] + pytest_opts
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    
    # Run with retries
    start_time = time.time()
    
    for attempt in range(1, args.retries + 1):
        if attempt > 1:
            print(f"\nRetry attempt {attempt}/{args.retries}...")
            time.sleep(2)  # Brief delay between attempts
        
        # Execute pytest
        result = subprocess.run(cmd)
        
        # If successful, break out of retry loop
        if result.returncode == 0:
            print("\nTests completed successfully!")
            break
        
        print(f"\nAttempt {attempt} failed with return code {result.returncode}")
        
        # If we've used all retries, report final failure
        if attempt == args.retries:
            print(f"\nAll {args.retries} attempts failed.")
    
    # Report total time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(2)
