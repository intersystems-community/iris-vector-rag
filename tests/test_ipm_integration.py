"""
Tests for IPM integration functionality.

This module tests the IPM installation, configuration, and integration components.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from iris_rag.utils.ipm_integration import (
    IPMIntegration,
    validate_ipm_environment,
    install_via_ipm,
    verify_ipm_installation
)


class TestIPMIntegration:
    """Test cases for IPM integration utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ipm = IPMIntegration()
    
    def test_package_info(self):
        """Test package information is correct."""
        assert self.ipm.package_name == "intersystems-iris-rag"
        assert self.ipm.version == "0.1.0"
    
    def test_validate_environment_structure(self):
        """Test that environment validation returns expected structure."""
        with patch.object(self.ipm, '_check_python_version') as mock_python, \
             patch.object(self.ipm, '_check_pip_available') as mock_pip, \
             patch.object(self.ipm, '_check_iris_python') as mock_iris, \
             patch.object(self.ipm, '_check_dependencies') as mock_deps, \
             patch.object(self.ipm, '_check_environment_variables') as mock_env:
            
            # Mock return values
            mock_python.return_value = {"valid": True, "version": "3.11.0", "message": "OK"}
            mock_pip.return_value = {"valid": True, "version": "pip 23.0", "message": "OK"}
            mock_iris.return_value = {"valid": True, "version": "5.1.2", "message": "OK"}
            mock_deps.return_value = {}
            mock_env.return_value = {}
            
            result = self.ipm.validate_environment()
            
            # Check structure
            assert "python_version" in result
            assert "pip_available" in result
            assert "iris_python" in result
            assert "dependencies" in result
            assert "environment_vars" in result
            assert "overall_status" in result
            
            # Check overall status calculation
            assert result["overall_status"] is True
    
    def test_check_python_version_valid(self):
        """Test Python version check with valid version."""
        with patch('sys.version_info', (3, 11, 5)):
            result = self.ipm._check_python_version()
            
            assert result["valid"] is True
            assert result["version"] == "3.11.5"
            assert "OK" in result["message"]
    
    def test_check_python_version_invalid(self):
        """Test Python version check with invalid version."""
        with patch('sys.version_info', (3, 9, 0)):
            result = self.ipm._check_python_version()
            
            assert result["valid"] is False
            assert result["version"] == "3.9.0"
            assert "required" in result["message"]
    
    @patch('subprocess.run')
    def test_check_pip_available_success(self, mock_run):
        """Test pip availability check when pip is available."""
        mock_run.return_value = Mock(returncode=0, stdout="pip 23.0.1")
        
        result = self.ipm._check_pip_available()
        
        assert result["valid"] is True
        assert "pip 23.0.1" in result["version"]
        assert "available" in result["message"]
    
    @patch('subprocess.run')
    def test_check_pip_available_failure(self, mock_run):
        """Test pip availability check when pip is not available."""
        mock_run.return_value = Mock(returncode=1, stdout="")
        
        result = self.ipm._check_pip_available()
        
        assert result["valid"] is False
        assert result["version"] == ""
        assert "not found" in result["message"]
    
    def test_check_iris_python_available(self):
        """Test IRIS Python check when available."""
        with patch('builtins.__import__') as mock_import:
            mock_module = Mock()
            mock_module.__version__ = "5.1.2"
            mock_import.return_value = mock_module
            
            result = self.ipm._check_iris_python()
            
            assert result["valid"] is True
            assert result["version"] == "5.1.2"
            assert "available" in result["message"]
    
    def test_check_iris_python_not_available(self):
        """Test IRIS Python check when not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module")):
            result = self.ipm._check_iris_python()
            
            assert result["valid"] is False
            assert result["version"] == ""
            assert "not found" in result["message"]
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        with patch('builtins.__import__') as mock_import:
            # Mock successful imports for some dependencies
            def import_side_effect(name):
                if name in ["torch", "sqlalchemy"]:
                    return Mock()
                else:
                    raise ImportError(f"No module named '{name}'")
            
            mock_import.side_effect = import_side_effect
            
            result = self.ipm._check_dependencies()
            
            # Check structure
            assert "torch" in result
            assert "sentence_transformers" in result
            assert "sqlalchemy" in result
            
            # Check availability
            assert result["torch"]["available"] is True
            assert result["sqlalchemy"]["available"] is True
            assert result["sentence_transformers"]["available"] is False
    
    @patch.dict(os.environ, {
        'IRIS_HOST': 'localhost',
        'IRIS_PORT': '1972',
        'OPENAI_API_KEY': 'test-key'
    })
    def test_check_environment_variables(self):
        """Test environment variable checking."""
        result = self.ipm._check_environment_variables()
        
        # Check structure
        assert "IRIS_HOST" in result
        assert "IRIS_PORT" in result
        assert "OPENAI_API_KEY" in result
        assert "IRIS_NAMESPACE" in result
        
        # Check values
        assert result["IRIS_HOST"]["set"] is True
        assert result["IRIS_HOST"]["value"] == "localhost"
        assert result["IRIS_PORT"]["set"] is True
        assert result["IRIS_PORT"]["value"] == "1972"
        assert result["OPENAI_API_KEY"]["set"] is True
        assert result["OPENAI_API_KEY"]["value"] == "test-key"
        assert result["IRIS_NAMESPACE"]["set"] is False
    
    @patch('subprocess.run')
    def test_install_package_success(self, mock_run):
        """Test successful package installation."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Successfully installed intersystems-iris-rag",
            stderr=""
        )
        
        result = self.ipm.install_package()
        
        assert result["success"] is True
        assert "Successfully installed" in result["stdout"]
        assert result["stderr"] == ""
        assert "pip install intersystems-iris-rag" in result["command"]
    
    @patch('subprocess.run')
    def test_install_package_failure(self, mock_run):
        """Test failed package installation."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="ERROR: Could not find a version"
        )
        
        result = self.ipm.install_package()
        
        assert result["success"] is False
        assert result["stdout"] == ""
        assert "ERROR" in result["stderr"]
    
    @patch('subprocess.run')
    def test_install_package_upgrade(self, mock_run):
        """Test package installation with upgrade."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = self.ipm.install_package(upgrade=True)
        
        assert result["success"] is True
        assert "--upgrade" in result["command"]
    
    def test_verify_installation_success(self):
        """Test successful installation verification."""
        with patch('builtins.__import__') as mock_import:
            # Mock successful iris_rag import
            mock_iris_rag = Mock()
            mock_iris_rag.__version__ = "0.1.0"
            
            mock_create_pipeline = Mock()
            
            def import_side_effect(name, fromlist=None):
                if name == "iris_rag":
                    return mock_iris_rag
                elif fromlist and "create_pipeline" in fromlist:
                    return Mock(create_pipeline=mock_create_pipeline)
                return Mock()
            
            mock_import.side_effect = import_side_effect
            
            result = self.ipm.verify_installation()
            
            assert result["success"] is True
            assert result["version"] == "0.1.0"
            assert result["import_test"] == "passed"
            assert result["factory_test"] == "passed"
    
    def test_verify_installation_import_failure(self):
        """Test installation verification with import failure."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'iris_rag'")):
            result = self.ipm.verify_installation()
            
            assert result["success"] is False
            assert result["version"] == "unknown"
            assert result["import_test"] == "failed"
            assert result["factory_test"] == "not tested"
            assert "Import failed" in result["message"]
    
    def test_generate_config_template(self):
        """Test configuration template generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config_str = self.ipm.generate_config_template(temp_path)
            
            # Check that config was generated
            assert len(config_str) > 0
            assert "database:" in config_str
            assert "embeddings:" in config_str
            assert "pipelines:" in config_str
            assert "llm:" in config_str
            
            # Check that file was created
            assert os.path.exists(temp_path)
            
            # Check file contents
            with open(temp_path, 'r') as f:
                file_content = f.read()
            
            assert file_content == config_str
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_generate_config_template_no_file(self):
        """Test configuration template generation without file output."""
        config_str = self.ipm.generate_config_template()
        
        # Check that config was generated
        assert len(config_str) > 0
        assert "database:" in config_str
        assert "embeddings:" in config_str
        assert "pipelines:" in config_str
        assert "llm:" in config_str
    
    def test_get_installation_info(self):
        """Test getting comprehensive installation information."""
        with patch.object(self.ipm, 'validate_environment') as mock_validate, \
             patch.object(self.ipm, 'verify_installation') as mock_verify:
            
            mock_validate.return_value = {"overall_status": True}
            mock_verify.return_value = {"success": True}
            
            result = self.ipm.get_installation_info()
            
            assert "package_name" in result
            assert "version" in result
            assert "environment" in result
            assert "installation" in result
            
            assert result["package_name"] == "intersystems-iris-rag"
            assert result["version"] == "0.1.0"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('iris_rag.utils.ipm_integration.IPMIntegration')
    def test_validate_ipm_environment(self, mock_ipm_class):
        """Test validate_ipm_environment convenience function."""
        mock_ipm = Mock()
        mock_ipm.validate_environment.return_value = {"status": "ok"}
        mock_ipm_class.return_value = mock_ipm
        
        result = validate_ipm_environment()
        
        assert result == {"status": "ok"}
        mock_ipm.validate_environment.assert_called_once()
    
    @patch('iris_rag.utils.ipm_integration.IPMIntegration')
    def test_install_via_ipm(self, mock_ipm_class):
        """Test install_via_ipm convenience function."""
        mock_ipm = Mock()
        mock_ipm.install_package.return_value = {"success": True}
        mock_ipm_class.return_value = mock_ipm
        
        result = install_via_ipm()
        
        assert result == {"success": True}
        mock_ipm.install_package.assert_called_once()
    
    @patch('iris_rag.utils.ipm_integration.IPMIntegration')
    def test_verify_ipm_installation(self, mock_ipm_class):
        """Test verify_ipm_installation convenience function."""
        mock_ipm = Mock()
        mock_ipm.verify_installation.return_value = {"success": True}
        mock_ipm_class.return_value = mock_ipm
        
        result = verify_ipm_installation()
        
        assert result == {"success": True}
        mock_ipm.verify_installation.assert_called_once()


class TestCommandLineInterface:
    """Test command-line interface functionality."""
    
    @patch('sys.argv', ['ipm_integration.py', '--validate'])
    @patch('iris_rag.utils.ipm_integration.IPMIntegration')
    @patch('builtins.print')
    def test_cli_validate(self, mock_print, mock_ipm_class):
        """Test CLI validate command."""
        mock_ipm = Mock()
        mock_ipm.validate_environment.return_value = {"status": "ok"}
        mock_ipm_class.return_value = mock_ipm
        
        # Import and run the CLI
        from iris_rag.utils.ipm_integration import IPMIntegration
        import argparse
        
        # Simulate CLI execution
        parser = argparse.ArgumentParser()
        parser.add_argument("--validate", action="store_true")
        args = parser.parse_args(['--validate'])
        
        if args.validate:
            ipm = IPMIntegration()
            result = ipm.validate_environment()
            print(json.dumps(result, indent=2))
        
        mock_ipm.validate_environment.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])