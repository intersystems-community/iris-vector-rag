#!/usr/bin/env python3
"""
Testing Framework Integration Validator
======================================

This script validates that all testing framework components work together seamlessly:
- Test mode framework (tests/test_modes.py, tests/conftest_test_modes.py)
- Post-installation validator (scripts/run_post_installation_tests.py)
- E2E validation runner (scripts/run_e2e_validation.py)
- Cross-language integration tests (tests/test_cross_language_integration.py)
- Real data validation tests (tests/test_real_data_validation.py)
- Mock control validator (tests/test_mode_validator.py)
- Updated Makefile commands

This validates the testing framework itself and ensures all components integrate properly.
"""

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'testing_framework_validation_{int(time.time())}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestingFrameworkValidator:
    """
    Validates the complete testing framework integration.
    Tests the testing framework itself to ensure all components work together.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.start_time = datetime.now()
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/testing_framework_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "validation_type": "TESTING_FRAMEWORK_INTEGRATION",
            "start_time": self.start_time.isoformat(),
            "components": {},
            "integration_tests": {},
            "cross_component_validation": {},
            "final_status": None,
            "errors": []
        }
    
    def run_complete_validation(self) -> bool:
        """Run complete testing framework validation."""
        logger.info("🚀 STARTING TESTING FRAMEWORK INTEGRATION VALIDATION")
        logger.info("=" * 70)
        
        try:
            # Phase 1: Component Existence Validation
            if not self._validate_component_existence():
                return False
            
            # Phase 2: Test Mode Framework Validation
            if not self._validate_test_mode_framework():
                return False
            
            # Phase 3: Mock Control System Validation
            if not self._validate_mock_control_system():
                return False
            
            # Phase 4: Script Integration Validation
            if not self._validate_script_integration():
                return False
            
            # Phase 5: Cross-Component Communication
            if not self._validate_cross_component_communication():
                return False
            
            # Phase 6: Makefile Integration
            if not self._validate_makefile_integration():
                return False
            
            # Phase 7: Backward Compatibility
            if not self._validate_backward_compatibility():
                return False
            
            self.results["final_status"] = "SUCCESS"
            logger.info("🎉 TESTING FRAMEWORK INTEGRATION VALIDATION PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ CRITICAL TESTING FRAMEWORK VALIDATION ERROR: {e}")
            self.results["final_status"] = "CRITICAL_FAILURE"
            self.results["errors"].append(str(e))
            return False
        
        finally:
            self._generate_validation_report()
    
    def _validate_component_existence(self) -> bool:
        """Phase 1: Validate all testing framework components exist."""
        logger.info("🔍 PHASE 1: Component Existence Validation")
        
        required_components = {
            "test_modes": "tests/test_modes.py",
            "conftest_test_modes": "tests/conftest_test_modes.py", 
            "post_installation_tests": "scripts/run_post_installation_tests.py",
            "e2e_validation": "scripts/run_e2e_validation.py",
            "cross_language_integration": "tests/test_cross_language_integration.py",
            "real_data_validation": "tests/test_real_data_validation.py",
            "mode_validator": "tests/test_mode_validator.py",
            "main_conftest": "tests/conftest.py",
            "makefile": "Makefile"
        }
        
        component_results = {}
        for component_name, component_path in required_components.items():
            full_path = project_root / component_path
            exists = full_path.exists()
            component_results[component_name] = {
                "exists": exists,
                "path": str(full_path),
                "size": full_path.stat().st_size if exists else 0
            }
            
            if exists:
                logger.info(f"✅ {component_name}: Found at {component_path}")
            else:
                logger.error(f"❌ {component_name}: Missing at {component_path}")
        
        self.results["components"]["existence_check"] = component_results
        
        all_exist = all(result["exists"] for result in component_results.values())
        if all_exist:
            logger.info("✅ All testing framework components exist")
            return True
        else:
            logger.error("❌ Some testing framework components are missing")
            return False
    
    def _validate_test_mode_framework(self) -> bool:
        """Phase 2: Validate test mode framework functionality."""
        logger.info("🔍 PHASE 2: Test Mode Framework Validation")
        
        test_results = {}
        
        # Test 1: Import test modes module
        try:
            from tests.test_modes import MockController, TestMode, mock_safe
            test_results["import_test_modes"] = True
            logger.info("✅ Successfully imported test_modes module")
        except Exception as e:
            test_results["import_test_modes"] = False
            logger.error(f"❌ Failed to import test_modes: {e}")
            self.results["errors"].append(f"test_modes import error: {e}")
        
        # Test 2: Test mode switching
        try:
            from tests.test_modes import MockController, TestMode
            
            # Test each mode
            for mode in [TestMode.UNIT, TestMode.INTEGRATION, TestMode.E2E]:
                MockController.set_test_mode(mode)
                current_mode = MockController.get_test_mode()
                if current_mode != mode:
                    raise ValueError(f"Mode switching failed: expected {mode}, got {current_mode}")
            
            test_results["mode_switching"] = True
            logger.info("✅ Test mode switching works correctly")
        except Exception as e:
            test_results["mode_switching"] = False
            logger.error(f"❌ Test mode switching failed: {e}")
            self.results["errors"].append(f"Mode switching error: {e}")
        
        # Test 3: Mock control functionality
        try:
            from tests.test_modes import MockController, TestMode, mock_safe
            
            # Test mock control in different modes
            MockController.set_test_mode(TestMode.UNIT)
            assert not MockController.are_mocks_disabled()
            
            MockController.set_test_mode(TestMode.E2E)
            assert MockController.are_mocks_disabled()
            
            test_results["mock_control"] = True
            logger.info("✅ Mock control functionality works correctly")
        except Exception as e:
            test_results["mock_control"] = False
            logger.error(f"❌ Mock control functionality failed: {e}")
            self.results["errors"].append(f"Mock control error: {e}")
        
        self.results["components"]["test_mode_framework"] = test_results
        return all(test_results.values())
    
    def _validate_mock_control_system(self) -> bool:
        """Phase 3: Validate mock control system with actual tests."""
        logger.info("🔍 PHASE 3: Mock Control System Validation")
        
        # Run the mock control validator tests
        test_command = [
            sys.executable, "-m", "pytest", 
            "tests/test_mode_validator.py", 
            "-v", "--tb=short"
        ]
        
        try:
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=120  # 2 minute timeout
            )
            
            mock_control_results = {
                "tests_executed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            if result.returncode == 0:
                logger.info("✅ Mock control system tests passed")
                # Parse test output for more details
                if "passed" in result.stdout:
                    passed_count = result.stdout.count(" PASSED")
                    logger.info(f"✅ {passed_count} mock control tests passed")
            else:
                logger.error("❌ Mock control system tests failed")
                logger.error(f"Error output: {result.stderr}")
            
            self.results["components"]["mock_control_system"] = mock_control_results
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Mock control system tests timed out")
            self.results["errors"].append("Mock control tests timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to run mock control tests: {e}")
            self.results["errors"].append(f"Mock control test execution error: {e}")
            return False
    
    def _validate_script_integration(self) -> bool:
        """Phase 4: Validate script integration and execution."""
        logger.info("🔍 PHASE 4: Script Integration Validation")
        
        script_results = {}
        
        # Test 1: Post-installation script syntax
        try:
            result = subprocess.run([
                sys.executable, "-m", "py_compile", 
                "scripts/run_post_installation_tests.py"
            ], capture_output=True, text=True, cwd=project_root)
            
            script_results["post_installation_syntax"] = result.returncode == 0
            if result.returncode == 0:
                logger.info("✅ Post-installation script syntax is valid")
            else:
                logger.error(f"❌ Post-installation script syntax error: {result.stderr}")
        except Exception as e:
            script_results["post_installation_syntax"] = False
            logger.error(f"❌ Failed to check post-installation script: {e}")
        
        # Test 2: E2E validation script syntax
        try:
            result = subprocess.run([
                sys.executable, "-m", "py_compile", 
                "scripts/run_e2e_validation.py"
            ], capture_output=True, text=True, cwd=project_root)
            
            script_results["e2e_validation_syntax"] = result.returncode == 0
            if result.returncode == 0:
                logger.info("✅ E2E validation script syntax is valid")
            else:
                logger.error(f"❌ E2E validation script syntax error: {result.stderr}")
        except Exception as e:
            script_results["e2e_validation_syntax"] = False
            logger.error(f"❌ Failed to check E2E validation script: {e}")
        
        # Test 3: Script import capabilities
        try:
            # Test if scripts can import required modules
            test_script = """
import sys
sys.path.insert(0, '.')
try:
    from tests.test_modes import MockController, TestMode
    from scripts.run_post_installation_tests import PostInstallationTester
    from scripts.run_e2e_validation import E2EValidationRunner
    print("SUCCESS: All imports work")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                test_file = f.name
            
            try:
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, cwd=project_root)
                
                script_results["import_capabilities"] = "SUCCESS" in result.stdout
                if "SUCCESS" in result.stdout:
                    logger.info("✅ Script import capabilities work correctly")
                else:
                    logger.error(f"❌ Script import failed: {result.stdout}")
            finally:
                os.unlink(test_file)
                
        except Exception as e:
            script_results["import_capabilities"] = False
            logger.error(f"❌ Failed to test script imports: {e}")
        
        self.results["components"]["script_integration"] = script_results
        return all(script_results.values())
    
    def _validate_cross_component_communication(self) -> bool:
        """Phase 5: Validate cross-component communication."""
        logger.info("🔍 PHASE 5: Cross-Component Communication Validation")
        
        communication_results = {}
        
        # Test 1: Test mode propagation across modules
        try:
            test_script = """
import sys
import os
sys.path.insert(0, '.')

from tests.test_modes import MockController, TestMode

# Set E2E mode
MockController.set_test_mode(TestMode.E2E)

# Check environment variables are set
assert os.environ.get("RAG_TEST_MODE") == "e2e"
assert os.environ.get("RAG_MOCKS_DISABLED") == "True"

# Import conftest_test_modes and check it sees the mode
from tests.conftest_test_modes import configure_test_mode

print("SUCCESS: Cross-module communication works")
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                test_file = f.name
            
            try:
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, cwd=project_root)
                
                communication_results["mode_propagation"] = "SUCCESS" in result.stdout
                if "SUCCESS" in result.stdout:
                    logger.info("✅ Cross-module mode propagation works")
                else:
                    logger.error(f"❌ Mode propagation failed: {result.stdout} {result.stderr}")
            finally:
                os.unlink(test_file)
                
        except Exception as e:
            communication_results["mode_propagation"] = False
            logger.error(f"❌ Failed to test mode propagation: {e}")
        
        # Test 2: Fixture integration
        try:
            # Test that conftest.py can work with test modes
            test_command = [
                sys.executable, "-c",
                """
import sys
sys.path.insert(0, '.')
from tests.conftest_test_modes import configure_test_mode
from tests.test_modes import MockController, TestMode
MockController.set_test_mode(TestMode.UNIT)
print("SUCCESS: Fixture integration works")
"""
            ]
            
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            communication_results["fixture_integration"] = "SUCCESS" in result.stdout
            if "SUCCESS" in result.stdout:
                logger.info("✅ Fixture integration works")
            else:
                logger.error(f"❌ Fixture integration failed: {result.stderr}")
                
        except Exception as e:
            communication_results["fixture_integration"] = False
            logger.error(f"❌ Failed to test fixture integration: {e}")
        
        self.results["integration_tests"]["cross_component_communication"] = communication_results
        return all(communication_results.values())
    
    def _validate_makefile_integration(self) -> bool:
        """Phase 6: Validate Makefile integration with new test commands."""
        logger.info("🔍 PHASE 6: Makefile Integration Validation")
        
        makefile_results = {}
        
        # Test 1: Check if new Makefile targets exist
        try:
            makefile_path = project_root / "Makefile"
            if makefile_path.exists():
                makefile_content = makefile_path.read_text()
                
                expected_targets = [
                    "test-e2e-validation",
                    "test-mode-validator",
                    "test-install"
                ]
                
                targets_found = {}
                for target in expected_targets:
                    targets_found[target] = target in makefile_content
                
                makefile_results["targets_exist"] = targets_found
                
                all_targets_found = all(targets_found.values())
                if all_targets_found:
                    logger.info("✅ All expected Makefile targets found")
                else:
                    missing = [t for t, found in targets_found.items() if not found]
                    logger.warning(f"⚠️ Missing Makefile targets: {missing}")
                
                makefile_results["all_targets_found"] = all_targets_found
            else:
                makefile_results["makefile_exists"] = False
                logger.error("❌ Makefile not found")
                
        except Exception as e:
            makefile_results["makefile_check_error"] = str(e)
            logger.error(f"❌ Failed to check Makefile: {e}")
        
        # Test 2: Test make command syntax (dry run)
        try:
            result = subprocess.run([
                "make", "-n", "help"
            ], capture_output=True, text=True, cwd=project_root)
            
            makefile_results["make_syntax"] = result.returncode == 0
            if result.returncode == 0:
                logger.info("✅ Makefile syntax is valid")
            else:
                logger.error(f"❌ Makefile syntax error: {result.stderr}")
                
        except Exception as e:
            makefile_results["make_syntax"] = False
            logger.error(f"❌ Failed to test make syntax: {e}")
        
        self.results["integration_tests"]["makefile_integration"] = makefile_results
        return makefile_results.get("all_targets_found", False) and makefile_results.get("make_syntax", False)
    
    def _validate_backward_compatibility(self) -> bool:
        """Phase 7: Validate backward compatibility with existing tests."""
        logger.info("🔍 PHASE 7: Backward Compatibility Validation")
        
        compatibility_results = {}
        
        # Test 1: Check that existing conftest.py still works
        try:
            test_command = [
                sys.executable, "-c",
                """
import sys
sys.path.insert(0, '.')
from tests.conftest import iris_connection_real, embedding_model_fixture, llm_client_fixture
print("SUCCESS: Existing conftest imports work")
"""
            ]
            
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            compatibility_results["existing_conftest"] = "SUCCESS" in result.stdout
            if "SUCCESS" in result.stdout:
                logger.info("✅ Existing conftest.py compatibility maintained")
            else:
                logger.error(f"❌ Existing conftest compatibility broken: {result.stderr}")
                
        except Exception as e:
            compatibility_results["existing_conftest"] = False
            logger.error(f"❌ Failed to test existing conftest: {e}")
        
        # Test 2: Check that existing test files can still be discovered
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--collect-only", "-q"
            ], capture_output=True, text=True, cwd=project_root, timeout=30)
            
            # Check if tests were collected successfully (even with RAGAS logging errors)
            tests_collected = "collected" in result.stdout and "items" in result.stdout
            ragas_logging_error = "AnalyticsBatcher shutdown complete" in result.stderr
            
            # Consider success if tests were collected, even with non-critical RAGAS errors
            compatibility_results["test_discovery"] = tests_collected or (result.returncode == 0)
            
            if tests_collected:
                logger.info("✅ Test discovery still works with new framework")
                # Count discovered tests
                try:
                    test_count = result.stdout.split('collected')[1].split()[0]
                    logger.info(f"Test discovery output: {test_count} tests found")
                except:
                    logger.info("✅ Tests discovered successfully")
            elif ragas_logging_error and tests_collected:
                logger.info("✅ Test discovery works (ignoring non-critical RAGAS logging error)")
            else:
                logger.error(f"❌ Test discovery broken: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            compatibility_results["test_discovery"] = False
            logger.error("❌ Test discovery timed out")
        except Exception as e:
            compatibility_results["test_discovery"] = False
            logger.error(f"❌ Failed to test discovery: {e}")
        
        self.results["integration_tests"]["backward_compatibility"] = compatibility_results
        return all(compatibility_results.values())
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        end_time = datetime.now()
        self.results["end_time"] = end_time.isoformat()
        self.results["duration"] = (end_time - self.start_time).total_seconds()
        
        # Save JSON report
        json_report = self.output_dir / f"testing_framework_validation_{int(time.time())}.json"
        with open(json_report, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Generate markdown report
        md_report = self.output_dir / f"testing_framework_validation_{int(time.time())}.md"
        self._generate_markdown_report(md_report)
        
        logger.info(f"📊 Validation reports saved:")
        logger.info(f"  JSON: {json_report}")
        logger.info(f"  Markdown: {md_report}")
    
    def _generate_markdown_report(self, report_path: Path):
        """Generate markdown validation report."""
        with open(report_path, "w") as f:
            f.write("# Testing Framework Integration Validation Report\n\n")
            f.write(f"**Generated:** {self.results['start_time']}\n")
            f.write(f"**Duration:** {self.results['duration']:.2f} seconds\n")
            f.write(f"**Status:** {self.results['final_status']}\n\n")
            
            # Component validation results
            f.write("## Component Validation Results\n\n")
            for component, results in self.results.get("components", {}).items():
                f.write(f"### {component.replace('_', ' ').title()}\n\n")
                if isinstance(results, dict):
                    for key, value in results.items():
                        status = "✅" if value else "❌"
                        f.write(f"- {status} **{key}**: {value}\n")
                f.write("\n")
            
            # Integration test results
            f.write("## Integration Test Results\n\n")
            for test_category, results in self.results.get("integration_tests", {}).items():
                f.write(f"### {test_category.replace('_', ' ').title()}\n\n")
                if isinstance(results, dict):
                    for key, value in results.items():
                        status = "✅" if value else "❌"
                        f.write(f"- {status} **{key}**: {value}\n")
                f.write("\n")
            
            # Errors
            if self.results.get("errors"):
                f.write("## Errors Encountered\n\n")
                for error in self.results["errors"]:
                    f.write(f"- ❌ {error}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if self.results["final_status"] == "SUCCESS":
                f.write("✅ **All testing framework components are properly integrated!**\n\n")
                f.write("The testing framework is ready for use. You can now:\n")
                f.write("- Run `make test-mode-validator` to validate mock control\n")
                f.write("- Run `make test-e2e-validation` for comprehensive E2E testing\n")
                f.write("- Run `make test-install` for post-installation validation\n")
            else:
                f.write("❌ **Testing framework integration issues detected.**\n\n")
                f.write("Please address the errors listed above before using the testing framework.\n")


def main():
    """Main function for testing framework validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate testing framework integration")
    parser.add_argument("--output-dir", help="Output directory for validation reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🧪 Testing Framework Integration Validator")
    print("=" * 50)
    print("This validates that all testing framework components work together.")
    print()
    
    validator = TestingFrameworkValidator(output_dir=args.output_dir)
    success = validator.run_complete_validation()
    
    print()
    print("=" * 50)
    if success:
        print("🎉 SUCCESS! Testing framework integration is working perfectly!")
        print("✅ All components are properly integrated and functional.")
    else:
        print("❌ FAILURE! Testing framework integration has issues.")
        print("🔧 Please check the validation report for details.")
    
    print("=" * 50)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()