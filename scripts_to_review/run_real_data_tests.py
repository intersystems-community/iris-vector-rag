#!/usr/bin/env python
"""
Run all RAG technique tests with real data using testcontainers.

This script sets the TEST_IRIS environment variable to use testcontainers
for all tests, and runs pytest with the real data configuration.
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run RAG tests with real data")
    
    parser.add_argument(
        "--techniques", 
        nargs="+", 
        choices=["all", "graphrag", "colbert", "hyde", "noderag", "crag", "basic"],
        default=["all"],
        help="RAG techniques to test"
    )
    
    parser.add_argument(
        "--pmc-limit", 
        type=int, 
        default=30,
        help="Number of PMC documents to use for testing"
    )
    
    parser.add_argument(
        "--document-count",
        type=int,
        default=None,
        help="Total number of documents to process (overrides --pmc-limit)"
    )
    
    parser.add_argument(
        "--use-real-embeddings",
        action="store_true",
        help="Use real embedding model (slower but more realistic, default in non-CI environments)"
    )
    
    parser.add_argument(
        "--use-mock-embeddings",
        action="store_true",
        help="Use mock embedding model (faster but less realistic, mainly for CI environments)"
    )
    
    parser.add_argument(
        "--no-testcontainer",
        action="store_true",
        help="Don't use testcontainer (use real or mock connections instead)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run with verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Set environment variables
    if not args.no_testcontainer:
        os.environ["TEST_IRIS"] = "true"
    
    # Handle document count/limit
    if args.document_count is not None:
        os.environ["TEST_DOCUMENT_COUNT"] = str(args.document_count)
        print(f"Using document count: {args.document_count}")
    else:
        os.environ["TEST_PMC_LIMIT"] = str(args.pmc_limit)
        print(f"Using PMC limit: {args.pmc_limit}")
    
    # Handle embedding model options - real is the default
    if args.use_mock_embeddings:
        os.environ["USE_MOCK_EMBEDDINGS"] = "true"
        print("Using mock embeddings (faster but less accurate)")
    elif args.use_real_embeddings or not os.environ.get('USE_MOCK_EMBEDDINGS'):
        # Explicitly set to false to override any environment setting
        os.environ["USE_MOCK_EMBEDDINGS"] = "false"
        print("Using real embeddings (more accurate for real data testing)")
    
    # Build test patterns
    if "all" in args.techniques:
        test_pattern = "tests/test_*_with_testcontainer.py"
    else:
        patterns = []
        technique_map = {
            "graphrag": "graphrag",
            "colbert": "colbert",
            "hyde": "hyde",
            "noderag": "noderag",
            "crag": "crag",
            "basic": "basic_rag"
        }
        for technique in args.techniques:
            patterns.append(f"tests/test_{technique_map[technique]}_with_testcontainer.py")
        test_pattern = " ".join(patterns)
    
    # Build pytest command
    verbose_flag = "-v" if args.verbose else ""
    command = f"python -m pytest {test_pattern} {verbose_flag} --log-cli-level=INFO"
    
    print(f"Running command: {command}")
    print(f"Environment settings:")
    print(f"  TEST_IRIS: {os.environ.get('TEST_IRIS', 'not set')}")
    print(f"  TEST_PMC_LIMIT: {os.environ.get('TEST_PMC_LIMIT', 'not set')}")
    print(f"  USE_MOCK_EMBEDDINGS: {os.environ.get('USE_MOCK_EMBEDDINGS', 'not set')}")
    
    # Execute pytest
    result = subprocess.run(command, shell=True)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
