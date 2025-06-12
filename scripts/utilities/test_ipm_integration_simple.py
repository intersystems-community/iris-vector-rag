#!/usr/bin/env python3
"""
Simple IPM Integration Test Runner

This script tests the IPM integration functionality without relying on pytest.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_ipm_integration_import():
    """Test that IPM integration can be imported."""
    try:
        from iris_rag.utils.ipm_integration import IPMIntegration
        print("✅ IPMIntegration import successful")
        return True
    except ImportError as e:
        print(f"❌ IPMIntegration import failed: {e}")
        return False

def test_ipm_integration_instantiation():
    """Test that IPMIntegration can be instantiated."""
    try:
        from iris_rag.utils.ipm_integration import IPMIntegration
        ipm = IPMIntegration()
        print("✅ IPMIntegration instantiation successful")
        print(f"   Package name: {ipm.package_name}")
        print(f"   Version: {ipm.version}")
        return True
    except Exception as e:
        print(f"❌ IPMIntegration instantiation failed: {e}")
        return False

def test_validate_environment_structure():
    """Test that validate_environment returns expected structure."""
    try:
        from iris_rag.utils.ipm_integration import IPMIntegration
        
        ipm = IPMIntegration()
        
        # Mock the internal methods to avoid actual system checks
        with patch.object(ipm, '_check_python_version') as mock_python, \
             patch.object(ipm, '_check_pip_available') as mock_pip, \
             patch.object(ipm, '_check_iris_python') as mock_iris, \
             patch.object(ipm, '_check_dependencies') as mock_deps, \
             patch.object(ipm, '_check_environment_variables') as mock_env:
            
            # Mock return values
            mock_python.return_value = {"valid": True, "version": "3.11.0", "message": "OK"}
            mock_pip.return_value = {"valid": True, "version": "pip 23.0", "message": "OK"}
            mock_iris.return_value = {"valid": True, "version": "5.1.2", "message": "OK"}
            mock_deps.return_value = {}
            mock_env.return_value = {}
            
            result = ipm.validate_environment()
            
            # Check structure
            required_keys = ["python_version", "pip_available", "iris_python", "dependencies", "environment_vars", "overall_status"]
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                print(f"❌ validate_environment missing keys: {missing_keys}")
                return False
            
            print("✅ validate_environment structure test passed")
            return True
            
    except Exception as e:
        print(f"❌ validate_environment structure test failed: {e}")
        return False

def test_config_template_generation():
    """Test configuration template generation."""
    try:
        from iris_rag.utils.ipm_integration import IPMIntegration
        
        ipm = IPMIntegration()
        
        # Test without file output
        config_str = ipm.generate_config_template()
        
        # Check that config was generated
        if len(config_str) == 0:
            print("❌ Config template generation failed: empty string")
            return False
        
        # Check for expected sections
        required_sections = ["database:", "embeddings:", "pipelines:", "llm:"]
        missing_sections = [section for section in required_sections if section not in config_str]
        
        if missing_sections:
            print(f"❌ Config template missing sections: {missing_sections}")
            return False
        
        # Test with file output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config_str_file = ipm.generate_config_template(temp_path)
            
            # Check that file was created
            if not os.path.exists(temp_path):
                print("❌ Config template file was not created")
                return False
            
            # Check file contents
            with open(temp_path, 'r') as f:
                file_content = f.read()
            
            if file_content != config_str_file:
                print("❌ Config template file content mismatch")
                return False
            
            print("✅ Config template generation test passed")
            return True
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Config template generation test failed: {e}")
        return False

def test_installation_info():
    """Test getting installation information."""
    try:
        from iris_rag.utils.ipm_integration import IPMIntegration
        
        ipm = IPMIntegration()
        
        # Mock the methods to avoid actual system checks
        with patch.object(ipm, 'validate_environment') as mock_validate, \
             patch.object(ipm, 'verify_installation') as mock_verify:
            
            mock_validate.return_value = {"overall_status": True}
            mock_verify.return_value = {"success": True}
            
            result = ipm.get_installation_info()
            
            # Check structure
            required_keys = ["package_name", "version", "environment", "installation"]
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                print(f"❌ get_installation_info missing keys: {missing_keys}")
                return False
            
            # Check values
            if result["package_name"] != "intersystems-iris-rag":
                print(f"❌ Incorrect package name: {result['package_name']}")
                return False
            
            if result["version"] != "0.1.0":
                print(f"❌ Incorrect version: {result['version']}")
                return False
            
            print("✅ Installation info test passed")
            return True
            
    except Exception as e:
        print(f"❌ Installation info test failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions."""
    try:
        from iris_rag.utils.ipm_integration import (
            validate_ipm_environment,
            install_via_ipm,
            verify_ipm_installation
        )
        
        # Test that functions exist and are callable
        if not callable(validate_ipm_environment):
            print("❌ validate_ipm_environment is not callable")
            return False
        
        if not callable(install_via_ipm):
            print("❌ install_via_ipm is not callable")
            return False
        
        if not callable(verify_ipm_installation):
            print("❌ verify_ipm_installation is not callable")
            return False
        
        print("✅ Convenience functions test passed")
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        return False

def test_iris_rag_package_structure():
    """Test iris_rag package structure."""
    try:
        # Test main package import
        import iris_rag
        
        # Check version
        if not hasattr(iris_rag, '__version__'):
            print("❌ iris_rag package missing __version__")
            return False
        
        if iris_rag.__version__ != "0.1.0":
            print(f"❌ Incorrect iris_rag version: {iris_rag.__version__}")
            return False
        
        # Test create_pipeline function
        from iris_rag import create_pipeline
        
        if not callable(create_pipeline):
            print("❌ create_pipeline is not callable")
            return False
        
        print("✅ iris_rag package structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ iris_rag package structure test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and return overall result."""
    print("🧪 Running IPM Integration Tests")
    print("=" * 50)
    
    tests = [
        ("IPM Integration Import", test_ipm_integration_import),
        ("IPM Integration Instantiation", test_ipm_integration_instantiation),
        ("Validate Environment Structure", test_validate_environment_structure),
        ("Config Template Generation", test_config_template_generation),
        ("Installation Info", test_installation_info),
        ("Convenience Functions", test_convenience_functions),
        ("iris_rag Package Structure", test_iris_rag_package_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

def main():
    """Main test function."""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()