#!/usr/bin/env python3
"""
Verify that all RAG techniques are being tested with 1000+ documents.

This script checks:
1. The test_all_with_1000_docs.py file to ensure all six RAG techniques are tested
2. That each test properly verifies document count
3. That the tests will be run with REAL PMC documents

This helps ensure compliance with the .clinerules requirement:
"Tests must use real PMC documents, not synthetic data. At least 1000 documents should be used."
"""

import os
import sys
import re
import logging
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_compliance")

# List of expected RAG techniques (all six must be tested)
EXPECTED_RAG_TECHNIQUES = {
    "basic_rag",
    "hyde",
    "colbert",
    "noderag",
    "graphrag",
    "crag"
}

def verify_test_file_exists() -> bool:
    """Verify that the test_all_with_1000_docs.py file exists."""
    test_file_path = os.path.join("tests", "test_all_with_1000_docs.py")
    if not os.path.exists(test_file_path):
        logger.error(f"❌ Test file missing: {test_file_path}")
        return False
    
    logger.info(f"✅ Test file exists: {test_file_path}")
    return True

def verify_conftest_real_pmc_exists() -> bool:
    """Verify that the conftest_real_pmc.py file exists."""
    conftest_path = os.path.join("tests", "conftest_real_pmc.py")
    if not os.path.exists(conftest_path):
        logger.error(f"❌ Real PMC conftest missing: {conftest_path}")
        return False
    
    logger.info(f"✅ Real PMC conftest exists: {conftest_path}")
    return True

def verify_all_techniques_tested() -> bool:
    """Verify that all six RAG techniques are tested in the test file."""
    test_file_path = os.path.join("tests", "test_all_with_1000_docs.py")
    
    if not os.path.exists(test_file_path):
        return False
    
    with open(test_file_path, "r") as f:
        content = f.read()
    
    # Look for test functions for each technique
    found_techniques = set()
    for technique in EXPECTED_RAG_TECHNIQUES:
        pattern = rf"def test_{technique}_with_1000_docs"
        if re.search(pattern, content):
            found_techniques.add(technique)
            logger.info(f"✅ Found test for {technique}")
        else:
            logger.error(f"❌ Missing test for {technique}")
    
    # Also check for imports of all pipeline modules
    for technique in EXPECTED_RAG_TECHNIQUES:
        pattern = rf"from {technique}\.pipeline import"
        if re.search(pattern, content):
            logger.info(f"✅ Found import for {technique} pipeline")
        else:
            logger.error(f"❌ Missing import for {technique} pipeline")
    
    # Verify document count checks
    if "verify_document_count" in content:
        logger.info("✅ Tests use verify_document_count fixture")
    else:
        logger.error("❌ Tests don't use verify_document_count fixture")
    
    # Verify use of requires_1000_docs marker
    if "@pytest.mark.requires_1000_docs" in content:
        logger.info("✅ Tests use requires_1000_docs marker")
    else:
        logger.error("❌ Tests don't use requires_1000_docs marker")
    
    # Check if all techniques are tested
    missing_techniques = EXPECTED_RAG_TECHNIQUES - found_techniques
    if missing_techniques:
        logger.error(f"❌ Missing tests for techniques: {', '.join(missing_techniques)}")
        return False
    
    logger.info(f"✅ All {len(EXPECTED_RAG_TECHNIQUES)} RAG techniques have tests")
    return True

def verify_makefile_targets() -> bool:
    """Verify that the Makefile contains targets for running tests with 1000+ documents."""
    if not os.path.exists("Makefile"):
        logger.error("❌ Makefile not found")
        return False
    
    with open("Makefile", "r") as f:
        content = f.read()
    
    targets_to_check = [
        "test-all-1000-docs-compliance",
        "test-with-real-pmc-db"
    ]
    
    all_targets_found = True
    for target in targets_to_check:
        if target in content:
            logger.info(f"✅ Found Makefile target: {target}")
        else:
            logger.error(f"❌ Missing Makefile target: {target}")
            all_targets_found = False
    
    return all_targets_found

def verify_real_pmc_data_available() -> bool:
    """Verify that real PMC data is available."""
    pmc_dir = os.path.join("data", "pmc_oas_downloaded")
    if not os.path.exists(pmc_dir):
        logger.warning(f"⚠️ PMC data directory not found: {pmc_dir}")
        return False
    
    # Count XML files in the directory
    xml_count = 0
    for dirpath, dirnames, filenames in os.walk(pmc_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_count += 1
    
    logger.info(f"Found {xml_count} PMC XML files in {pmc_dir}")
    
    if xml_count == 0:
        logger.warning("⚠️ No PMC XML files found")
        return False
    
    if xml_count < 1000:
        logger.warning(f"⚠️ Found only {xml_count} PMC XML files, need at least 1000")
        logger.warning("   Tests will try to use available files, but may not fully comply with requirements")
        return False
    
    logger.info(f"✅ Found {xml_count} PMC XML files (>= 1000 required)")
    return True

def main() -> int:
    """Main function to verify compliance."""
    logger.info("=" * 80)
    logger.info("Verifying 1000+ documents testing compliance")
    logger.info("=" * 80)
    
    # Track verification results
    results = {}
    
    # Check 1: Verify test file exists
    results["test_file_exists"] = verify_test_file_exists()
    
    # Check 2: Verify conftest_real_pmc.py exists
    results["conftest_real_pmc_exists"] = verify_conftest_real_pmc_exists()
    
    # Check 3: Verify all techniques are tested
    results["all_techniques_tested"] = verify_all_techniques_tested()
    
    # Check 4: Verify Makefile targets
    results["makefile_targets"] = verify_makefile_targets()
    
    # Check 5: Verify real PMC data available
    results["real_pmc_data_available"] = verify_real_pmc_data_available()
    
    # Summary
    logger.info("=" * 80)
    logger.info("Compliance Verification Summary:")
    logger.info("=" * 80)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {check}")
    
    # Overall result
    if all(results.values()):
        logger.info("=" * 80)
        logger.info("✅ FULLY COMPLIANT: All checks passed")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error("❌ NOT FULLY COMPLIANT: Some checks failed")
        logger.error("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
