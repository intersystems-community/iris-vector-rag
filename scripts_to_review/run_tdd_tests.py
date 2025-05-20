#!/usr/bin/env python3
"""
Run TDD Tests for RAG

This script runs our TDD tests and provides detailed output.
"""

import sys
import importlib
import time
import logging
from typing import List, Callable, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_test(test_function: Callable) -> Tuple[bool, str]:
    """Run a single test function and return the result"""
    start_time = time.time()
    name = test_function.__name__
    
    logger.info(f"Running test: {name}")
    
    try:
        # Run the test
        test_function()
        duration = time.time() - start_time
        logger.info(f"✅ Test {name} PASSED in {duration:.2f}s")
        return True, f"✅ {name} passed in {duration:.2f}s"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Test {name} FAILED in {duration:.2f}s: {str(e)}")
        return False, f"❌ {name} failed in {duration:.2f}s: {str(e)}"

def run_tests_in_module(module_name: str) -> List[Tuple[bool, str]]:
    """Run all tests in a module"""
    results = []
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Find test functions
        test_functions = []
        for name in dir(module):
            if name.startswith("test_"):
                func = getattr(module, name)
                if callable(func):
                    test_functions.append(func)
        
        if not test_functions:
            logger.warning(f"No test functions found in module {module_name}")
            return []
        
        # Run each test
        for func in test_functions:
            result = run_test(func)
            results.append(result)
        
    except ImportError:
        logger.error(f"Could not import module {module_name}")
    except Exception as e:
        logger.error(f"Error running tests in {module_name}: {str(e)}")
    
    return results

def run_specific_test(module_name: str, test_name: str) -> List[Tuple[bool, str]]:
    """Run a specific test in a module"""
    results = []
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get the test function
        if hasattr(module, test_name):
            func = getattr(module, test_name)
            if callable(func):
                result = run_test(func)
                results.append(result)
            else:
                logger.error(f"{test_name} is not callable in {module_name}")
        else:
            logger.error(f"{test_name} not found in {module_name}")
    
    except ImportError:
        logger.error(f"Could not import module {module_name}")
    except Exception as e:
        logger.error(f"Error running test {test_name} in {module_name}: {str(e)}")
    
    return results

def main():
    """Main entry point"""
    all_tests = False
    module_name = "tests.test_tdd_simpler"
    test_name = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            all_tests = True
        elif ":" in arg:
            parts = arg.split(":")
            module_name = parts[0]
            test_name = parts[1] if len(parts) > 1 else None
        else:
            module_name = arg
    
    # Convert module name format if needed
    if "/" in module_name:
        module_name = module_name.replace("/", ".").replace(".py", "")
    
    # Run tests
    if all_tests:
        modules = [
            "tests.test_tdd_simpler",
            "tests.test_tdd_basic_rag",
            "tests.test_tdd_colbert"
        ]
        results = []
        for mod in modules:
            results.extend(run_tests_in_module(mod))
    elif test_name:
        results = run_specific_test(module_name, test_name)
    else:
        results = run_tests_in_module(module_name)
    
    # Summarize results
    total = len(results)
    passed = sum(1 for success, _ in results if success)
    failed = total - passed
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed, {failed} failed")
    print("=" * 70)
    
    for success, message in results:
        print(message)
    
    # Return non-zero exit code if any tests failed
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
