#!/usr/bin/env python
"""
Verification script to ensure all tests are using 1000+ documents.

This script validates that the testing infrastructure is set up correctly to 
enforce the project requirement of testing with at least 1000 documents.
"""

import os
import sys
import re
import glob
import logging
from typing import List, Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MIN_DOCUMENTS = 1000
TEST_DIR = "tests"
RAG_TECHNIQUES = [
    "basic_rag",
    "colbert",
    "noderag",
    "graphrag",
    "hyde",
    "context_reduction"
]

def find_test_files() -> List[str]:
    """Find all test files that should be using 1000+ documents."""
    # Get all Python test files
    test_files = glob.glob(f"{TEST_DIR}/test_*.py")
    
    # Focus on files that might be testing RAG techniques
    rag_test_files = []
    for file in test_files:
        basename = os.path.basename(file)
        # Include files that have "1000" in the name or test a RAG technique
        if "1000" in basename or any(technique in basename for technique in RAG_TECHNIQUES):
            rag_test_files.append(file)
    
    return rag_test_files

def check_document_count_assertions(file_path: str) -> Tuple[bool, int]:
    """
    Check if a test file contains assertions for minimum document count.
    
    Returns:
        Tuple of (has_assertion, count)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for document count constants
    min_doc_patterns = [
        r'MIN_DOCUMENTS\s*=\s*(\d+)',
        r'min_document[s_]*count\s*=\s*(\d+)',
        r'min_docs\s*=\s*(\d+)',
        r'assert\s+count\s*>=\s*(\d+)',
        r'assert.*documents.*>=\s*(\d+)',
        r'min_document_count\(\)\s*->\s*(\d+)'
    ]
    
    for pattern in min_doc_patterns:
        matches = re.findall(pattern, content)
        if matches:
            # Get the highest count mentioned
            counts = [int(count) for count in matches]
            return True, max(counts)
    
    # Check if the file imports conftest_1000docs or uses the fixture
    if "conftest_1000docs" in content or "verify_min_document_count" in content:
        return True, MIN_DOCUMENTS
    
    return False, 0

def check_all_test_files() -> Dict[str, Dict[str, Any]]:
    """Check all test files for document count assertions."""
    results = {}
    files = find_test_files()
    
    logger.info(f"Checking {len(files)} test files for 1000+ document assertions...")
    
    for file_path in files:
        basename = os.path.basename(file_path)
        has_assertion, count = check_document_count_assertions(file_path)
        
        result = {
            "file": file_path,
            "has_1000_docs_assertion": has_assertion,
            "doc_count": count,
            "compliant": has_assertion and count >= MIN_DOCUMENTS,
        }
        
        results[basename] = result
    
    return results

def check_conftest_files() -> Dict[str, Dict[str, Any]]:
    """Check conftest files for document count assertions."""
    results = {}
    conftest_files = [
        "tests/conftest.py",
        "tests/conftest_1000docs.py",
        "tests/conftest_real_pmc.py"
    ]
    
    logger.info(f"Checking conftest files for 1000+ document assertions...")
    
    for file_path in conftest_files:
        if not os.path.exists(file_path):
            continue
            
        basename = os.path.basename(file_path)
        has_assertion, count = check_document_count_assertions(file_path)
        
        result = {
            "file": file_path,
            "has_1000_docs_assertion": has_assertion,
            "doc_count": count,
            "compliant": has_assertion and count >= MIN_DOCUMENTS,
        }
        
        results[basename] = result
    
    return results

def check_main_test_runner() -> Dict[str, Any]:
    """Check if the main test runner enforces 1000+ documents."""
    run_files = [
        "run_all_tests_with_1000_docs.sh",
        "run_1000_docs_tests.py",
        "run_all_rag_techniques_with_1000_docs.sh"
    ]
    
    for file_path in run_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if it uses the 1000 docs fixture
        if "conftest_1000docs" in content:
            return {
                "file": file_path,
                "enforces_1000_docs": True,
                "mechanism": "conftest_1000docs.py"
            }
    
    return {
        "file": None,
        "enforces_1000_docs": False,
        "mechanism": None
    }

def print_summary(test_results, conftest_results, runner_result):
    """Print a summary of the verification results."""
    print("\n===== 1000+ Documents Testing Verification =====\n")
    
    # Check overall compliance
    all_test_files_compliant = all(result["compliant"] for result in test_results.values())
    conftest_compliant = any(result["compliant"] for result in conftest_results.values())
    runner_compliant = runner_result["enforces_1000_docs"]
    
    overall_compliant = all_test_files_compliant and conftest_compliant and runner_compliant
    
    # Print test files results
    print(f"Test Files ({len(test_results)}):")
    for name, result in test_results.items():
        status = "✅" if result["compliant"] else "❌"
        count = result["doc_count"] if result["has_1000_docs_assertion"] else "N/A"
        print(f"  {status} {name:<30} - Document Count: {count}")
    
    # Print conftest results
    print("\nConftest Files:")
    for name, result in conftest_results.items():
        status = "✅" if result["compliant"] else "❌"
        count = result["doc_count"] if result["has_1000_docs_assertion"] else "N/A"
        print(f"  {status} {name:<30} - Document Count: {count}")
    
    # Print runner result
    print("\nTest Runner:")
    runner_status = "✅" if runner_compliant else "❌"
    runner_file = runner_result["file"] if runner_result["file"] else "None found"
    print(f"  {runner_status} {runner_file}")
    
    # Print overall result
    print("\nOverall Compliance:")
    overall_status = "✅ PASSED" if overall_compliant else "❌ FAILED"
    print(f"  {overall_status} - Testing with 1000+ documents")
    
    if not overall_compliant:
        print("\nRecommendations:")
        if not conftest_compliant:
            print("  - Use tests/conftest_1000docs.py to enforce 1000+ documents")
        if not runner_compliant:
            print("  - Use run_all_tests_with_1000_docs.sh to run tests with 1000+ documents")
        if not all_test_files_compliant:
            print("  - Update test files to use fixtures from conftest_1000docs.py")
            print("  - Add explicit assertions for document count >= 1000")
    
    print("\n=================================================\n")
    
    return overall_compliant

def main():
    """Main function."""
    logger.info("Verifying 1000+ documents testing infrastructure...")
    
    # Check test files
    test_results = check_all_test_files()
    
    # Check conftest files
    conftest_results = check_conftest_files()
    
    # Check test runner
    runner_result = check_main_test_runner()
    
    # Print summary
    is_compliant = print_summary(test_results, conftest_results, runner_result)
    
    # Exit with status code
    sys.exit(0 if is_compliant else 1)

if __name__ == "__main__":
    main()
