#!/usr/bin/env python3
"""
A simple script to run all RAG tests with 1000+ documents

This script uses pytest to run all the tests in test_all_with_1000_docs.py
"""

import sys
import os
import subprocess

# Create a temporary conftest.py in the project root
temp_conftest = """
# Temporary conftest.py that imports fixtures from conftest_1000docs.py
import os
import sys
from tests.conftest_1000docs import *
"""

def run_tests():
    try:
        # Write temporary conftest.py in the project root
        with open('conftest.py', 'w') as f:
            f.write(temp_conftest)
        
        # Run the tests with pytest
        print("Running 1000+ documents tests...")
        cmd = ["python", "-m", "pytest", "-v", "tests/test_all_with_1000_docs.py"]
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n✅ All 1000+ document tests passed successfully!")
        else:
            print(f"\n❌ Tests failed with exit code: {result.returncode}")
        
        return result.returncode
    finally:
        # Clean up temporary conftest.py
        if os.path.exists('conftest.py'):
            os.remove('conftest.py')
            print("Cleaned up temporary conftest.py")

if __name__ == "__main__":
    sys.exit(run_tests())
