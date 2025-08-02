#!/usr/bin/env python3
"""
Post-Installation Test Runner
============================

🎯 CLEAR INSTRUCTIONS FOR YOUR INTERN:

After installing the RAG Templates Library, run this script to verify everything works:

    python scripts/run_post_installation_tests.py

This script will:
1. Check your environment setup
2. Run basic functionality tests
3. Run integration tests with real database
4. Run full end-to-end validation
5. Generate a clear PASS/FAIL report

NO CONFUSION - just run this one script after installation!
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'post_installation_test_{int(time.time())}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PostInstallationTester:
    """
    Simple, clear post-installation test runner.
    No confusion - just tells you if the installation works or not.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {
            "test_type": "POST_INSTALLATION_VALIDATION",
            "start_time": self.start_time.isoformat(),
            "phases": {},
            "final_status": None
        }
        
    def run_all_tests(self) -> bool:
        """Run all post-installation tests in order."""
        logger.info("🚀 STARTING POST-INSTALLATION TESTS")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Environment Check
            if not self.check_environment():
                return False
            
            # Phase 2: Basic Functionality
            if not self.test_basic_functionality():
                return False
            
            # Phase 3: Database Integration
            if not self.test_database_integration():
                return False
            
            # Phase 4: Full End-to-End
            if not self.test_full_e2e():
                return False
            
            # All tests passed!
            self.results["final_status"] = "SUCCESS"
            logger.info("🎉 ALL POST-INSTALLATION TESTS PASSED!")
            logger.info("✅ Your RAG Templates Library installation is working perfectly!")
            return True
            
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: {e}")
            self.results["final_status"] = "CRITICAL_FAILURE"
            self.results["critical_error"] = str(e)
            return False
    
    def check_environment(self) -> bool:
        """Phase 1: Check environment setup."""
        logger.info("🔍 PHASE 1: Environment Check")
        
        checks = {
            "python_version": self._check_python_version(),
            "required_packages": self._check_packages(),
            "iris_connection": self._check_iris_connection(),
            "environment_variables": self._check_env_vars()
        }
        
        self.results["phases"]["environment"] = checks
        
        if all(checks.values()):
            logger.info("✅ Environment check passed!")
            return True
        else:
            logger.error("❌ Environment check failed!")
            self._log_environment_issues(checks)
            return False
    
    def test_basic_functionality(self) -> bool:
        """Phase 2: Test basic functionality with unit tests."""
        logger.info("🔍 PHASE 2: Basic Functionality Tests")
        
        # Set unit test mode (mocks allowed)
        os.environ["RAG_TEST_MODE"] = "unit"
        
        # Run basic unit tests
        basic_tests = [
            "tests/test_simple_api_phase1.py::TestSimpleAPIPhase1::test_initialization",
            "tests/test_standard_api_phase2.py::TestStandardAPIPhase2::test_configuration_loading",
            "tests/test_core/test_connection.py",
            "tests/test_core/test_models.py"
        ]
        
        results = {}
        for test in basic_tests:
            result = self._run_pytest(test)
            test_name = test.split("::")[-1] if "::" in test else Path(test).stem
            results[test_name] = result
        
        self.results["phases"]["basic_functionality"] = results
        
        if all(results.values()):
            logger.info("✅ Basic functionality tests passed!")
            return True
        else:
            logger.error("❌ Basic functionality tests failed!")
            return False
    
    def test_database_integration(self) -> bool:
        """Phase 3: Test database integration."""
        logger.info("🔍 PHASE 3: Database Integration Tests")
        
        # Set integration test mode (some real components)
        os.environ["RAG_TEST_MODE"] = "integration"
        
        # Run integration tests
        integration_tests = [
            "tests/test_iris_connector.py::test_real_iris_connection",
            "tests/test_dbapi_connection.py",
            "tests/test_e2e_iris_rag_db_connection.py"
        ]
        
        results = {}
        for test in integration_tests:
            result = self._run_pytest(test)
            test_name = test.split("::")[-1] if "::" in test else Path(test).stem
            results[test_name] = result
        
        self.results["phases"]["database_integration"] = results
        
        if all(results.values()):
            logger.info("✅ Database integration tests passed!")
            return True
        else:
            logger.error("❌ Database integration tests failed!")
            return False
    
    def test_full_e2e(self) -> bool:
        """Phase 4: Full end-to-end tests with real data."""
        logger.info("🔍 PHASE 4: Full End-to-End Tests")
        
        # Set E2E test mode (NO MOCKS, real everything)
        os.environ["RAG_TEST_MODE"] = "e2e"
        os.environ["RAG_MOCKS_DISABLED"] = "true"
        
        # Run the most important E2E tests
        e2e_tests = [
            "tests/test_comprehensive_e2e_iris_rag_1000_docs.py",
            "tests/test_e2e_rag_pipelines.py",
            "tests/test_simple_api_phase1.py::TestSimpleAPIPhase1::test_real_database_integration",
            "tests/test_javascript_simple_api_phase3.py::TestJavaScriptIntegration::test_real_iris_connection",
            "tests/test_objectscript_integration_phase5.py::TestObjectScriptIntegration::test_real_library_consumption"
        ]
        
        results = {}
        for test in e2e_tests:
            result = self._run_pytest(test)
            test_name = test.split("::")[-1] if "::" in test else Path(test).stem
            results[test_name] = result
        
        self.results["phases"]["full_e2e"] = results
        
        if all(results.values()):
            logger.info("✅ Full end-to-end tests passed!")
            return True
        else:
            logger.error("❌ Full end-to-end tests failed!")
            return False
    
    def _check_python_version(self) -> bool:
        """Check Python version."""
        return sys.version_info >= (3, 9)
    
    def _check_packages(self) -> bool:
        """Check required packages."""
        required = ["iris_rag", "rag_templates", "common", "torch", "transformers"]
        missing = []
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.error(f"Missing packages: {missing}")
            return False
        return True
    
    def _check_iris_connection(self) -> bool:
        """Check IRIS database connection."""
        try:
            from common.iris_connection import IRISConnection
            conn = IRISConnection()
            conn.connect()
            conn.disconnect()
            return True
        except Exception as e:
            logger.error(f"IRIS connection failed: {e}")
            return False
    
    def _check_env_vars(self) -> bool:
        """Check environment variables."""
        required = ["IRIS_HOST", "IRIS_PORT", "IRIS_USERNAME", "IRIS_PASSWORD"]
        missing = [var for var in required if not os.environ.get(var)]
        
        if missing:
            logger.error(f"Missing environment variables: {missing}")
            return False
        return True
    
    def _run_pytest(self, test_path: str) -> bool:
        """Run a specific pytest."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=project_root)
            
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to run test {test_path}: {e}")
            return False
    
    def _log_environment_issues(self, checks: Dict[str, bool]):
        """Log specific environment issues."""
        for check, passed in checks.items():
            if not passed:
                logger.error(f"❌ {check} failed")
    
    def save_results(self):
        """Save test results."""
        end_time = datetime.now()
        self.results["end_time"] = end_time.isoformat()
        self.results["duration"] = (end_time - self.start_time).total_seconds()
        
        results_file = f"post_installation_results_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {results_file}")


def main():
    """Main function - keep it simple for the intern!"""
    print("🎯 RAG Templates Library - Post-Installation Test")
    print("=" * 50)
    print("This will verify your installation works correctly.")
    print("Please wait while we run the tests...")
    print()
    
    tester = PostInstallationTester()
    success = tester.run_all_tests()
    tester.save_results()
    
    print()
    print("=" * 50)
    if success:
        print("🎉 SUCCESS! Your installation is working perfectly!")
        print("✅ You can now use the RAG Templates Library.")
        print()
        print("Next steps:")
        print("- Check out the examples in the examples/ directory")
        print("- Read the documentation in docs/")
        print("- Try the simple API: from rag_templates.simple import RAG")
    else:
        print("❌ FAILURE! There are issues with your installation.")
        print("🔧 Please check the log file for details.")
        print("📧 Contact support if you need help.")
    
    print("=" * 50)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()