#!/usr/bin/env python3
"""
Run all RAG tests with 1000+ REAL PMC documents.

This script uses the conftest_real_pmc.py fixture to ensure all tests run with
at least 1000 real PMC documents as required by the project standards.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_real_pmc_1000_tests")

def main():
    """Run all RAG tests with 1000+ real PMC documents."""
    logger.info("=" * 80)
    logger.info("Running ALL RAG tests with 1000+ REAL PMC documents")
    logger.info("This satisfies the .clinerules requirement for real data testing")
    logger.info("=" * 80)
    
    project_root = os.getcwd()
    tests_dir = os.path.join(project_root, "tests")
    original_conftest_path = os.path.join(tests_dir, "conftest.py")
    real_pmc_conftest_path = os.path.join(tests_dir, "conftest_real_pmc.py")
    temp_conftest_path = os.path.join(tests_dir, "conftest.py") # This will be conftest_real_pmc.py

    original_conftest_backup_path = os.path.join(tests_dir, "conftest_original.py.bak")
    
    renamed_original = False
    renamed_real_pmc = False

    try:
        # 1. Rename tests/conftest.py to tests/conftest_original.py.bak if it exists
        if os.path.exists(original_conftest_path) and original_conftest_path != real_pmc_conftest_path:
            logger.info(f"Backing up {original_conftest_path} to {original_conftest_backup_path}")
            os.rename(original_conftest_path, original_conftest_backup_path)
            renamed_original = True

        # 2. Rename tests/conftest_real_pmc.py to tests/conftest.py
        if os.path.exists(real_pmc_conftest_path):
            logger.info(f"Renaming {real_pmc_conftest_path} to {temp_conftest_path}")
            os.rename(real_pmc_conftest_path, temp_conftest_path)
            renamed_real_pmc = True
        else:
            logger.error(f"Error: {real_pmc_conftest_path} not found!")
            return 1

        # 3. Run pytest
        cmd = [
            "python", "-m", "pytest", 
            "-v", 
            "test_all_with_1000_docs.py" # Run specific file
        ]
        
        logger.info(f"Running command: {' '.join(cmd)} in directory {tests_dir}")
        # Execute pytest with the current working directory set to tests_dir
        result = subprocess.run(cmd, cwd=tests_dir, check=False, capture_output=True, text=True)
        
    except Exception as e:
        logger.error(f"Error during test execution or file operations: {e}")
        # Ensure cleanup happens even if subprocess.run fails before 'result' is assigned
        result = subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=str(e))
    finally:
        # 4. Revert renames
        if renamed_real_pmc and os.path.exists(temp_conftest_path):
            logger.info(f"Renaming {temp_conftest_path} back to {real_pmc_conftest_path}")
            os.rename(temp_conftest_path, real_pmc_conftest_path)
        
        if renamed_original and os.path.exists(original_conftest_backup_path):
            logger.info(f"Restoring {original_conftest_backup_path} to {original_conftest_path}")
            os.rename(original_conftest_backup_path, original_conftest_path)

    # Process result after cleanup
    if result.returncode == 0:
        logger.info("\n✅ All RAG techniques successfully tested with 1000+ REAL PMC documents!")
        return 0
    else:
        logger.error(f"\n❌ Tests failed with exit code: {result.returncode}")
        if result.stdout:
            logger.error(f"Stdout:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Stderr:\n{result.stderr}")
        return result.returncode

if __name__ == "__main__":
    sys.exit(main())
