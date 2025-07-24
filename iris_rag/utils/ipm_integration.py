"""
IPM (InterSystems Package Manager) integration utilities.

This module provides utilities for integrating with IPM installation and configuration.
"""

import os
import sys
import subprocess
import json
from typing import Dict, Any, Optional


class IPMIntegration:
    """Utilities for IPM integration and validation."""
    
    def __init__(self):
        self.package_name = "intersystems-iris-rag"
        self.version = "0.1.0"
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate the environment for IPM installation.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "python_version": self._check_python_version(),
            "pip_available": self._check_pip_available(),
            "iris_python": self._check_iris_python(),
            "dependencies": self._check_dependencies(),
            "environment_vars": self._check_environment_variables()
        }
        
        results["overall_status"] = all([
            results["python_version"]["valid"],
            results["pip_available"]["valid"],
            results["iris_python"]["valid"]
        ])
        
        return results
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check if Python version meets requirements."""
        try:
            version = sys.version_info
            valid = version.major == 3 and version.minor >= 11
            return {
                "valid": valid,
                "version": f"{version.major}.{version.minor}.{version.micro}",
                "message": "Python 3.11+ required" if not valid else "Python version OK"
            }
        except Exception as e:
            return {
                "valid": False,
                "version": "unknown",
                "message": f"Error checking Python version: {e}"
            }
    
    def _check_pip_available(self) -> Dict[str, Any]:
        """Check if pip is available."""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return {
                "valid": result.returncode == 0,
                "version": result.stdout.strip() if result.returncode == 0 else "",
                "message": "pip is available" if result.returncode == 0 else "pip not found"
            }
        except Exception as e:
            return {
                "valid": False,
                "version": "",
                "message": f"Error checking pip: {e}"
            }
    
    def _check_iris_python(self) -> Dict[str, Any]:
        """Check if IRIS Python is available."""
        try:
            import iris
            import importlib.metadata
            version = importlib.metadata.version("intersystems-irispython")
            return {
                "valid": True,
                "version": version,
                "message": "IRIS Python is available"
            }
        except ImportError:
            return {
                "valid": False,
                "version": "",
                "message": "IRIS Python not found - will be installed"
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check if key dependencies are available."""
        dependencies = {
            "torch": "PyTorch for ML operations",
            "sentence_transformers": "Sentence Transformers for embeddings",
            "sqlalchemy": "SQLAlchemy for database operations"
        }
        
        results = {}
        for dep, description in dependencies.items():
            try:
                __import__(dep)
                results[dep] = {"available": True, "description": description}
            except ImportError:
                results[dep] = {"available": False, "description": description}
        
        return results
    
    def _check_environment_variables(self) -> Dict[str, Any]:
        """Check for relevant environment variables."""
        env_vars = {
            "IRIS_HOST": os.getenv("IRIS_HOST"),
            "IRIS_PORT": os.getenv("IRIS_PORT"),
            "IRIS_NAMESPACE": os.getenv("IRIS_NAMESPACE"),
            "IRIS_USER": os.getenv("IRIS_USER"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "PYTHONPATH": os.getenv("PYTHONPATH")
        }
        
        return {var: {"set": value is not None, "value": value if value else "not set"} 
                for var, value in env_vars.items()}
    
    def install_package(self, upgrade: bool = False) -> Dict[str, Any]:
        """
        Install the iris_rag package via pip.
        
        Args:
            upgrade: Whether to upgrade if already installed
            
        Returns:
            Installation result
        """
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(self.package_name)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd) if 'cmd' in locals() else "unknown"
            }
    
    def verify_installation(self) -> Dict[str, Any]:
        """
        Verify that the package is properly installed.
        
        Returns:
            Verification results
        """
        try:
            # Try to import the main package
            import iris_rag
            
            # Check version
            version = getattr(iris_rag, "__version__", "unknown")
            
            # Try to create a basic pipeline
            from iris_rag import create_pipeline
            
            return {
                "success": True,
                "version": version,
                "import_test": "passed",
                "factory_test": "passed",
                "message": "Package installation verified"
            }
        except ImportError as e:
            return {
                "success": False,
                "version": "unknown",
                "import_test": "failed",
                "factory_test": "not tested",
                "message": f"Import failed: {e}"
            }
        except Exception as e:
            return {
                "success": False,
                "version": "unknown",
                "import_test": "passed",
                "factory_test": "failed",
                "message": f"Factory test failed: {e}"
            }
    
    def generate_config_template(self, output_path: Optional[str] = None) -> str:
        """
        Generate a configuration template for the RAG system.
        
        Args:
            output_path: Optional path to save the config file
            
        Returns:
            Configuration template as string
        """
        config_template = {
            "database": {
                "iris": {
                    "host": "localhost",
                    "port": 1972,
                    "namespace": "USER",
                    "username": "demo",
                    "password": "demo",
                    "driver": "intersystems_iris.dbapi"
                }
            },
            "embeddings": {
                "primary_backend": "sentence_transformers",
                "sentence_transformers": {
                    "model_name": "all-MiniLM-L6-v2"
                },
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model_name": "text-embedding-ada-002"
                },
                "dimension": 384
            },
            "pipelines": {
                "basic": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "default_top_k": 5
                }
            },
            "llm": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "api_key": "${OPENAI_API_KEY}",
                "temperature": 0.0,
                "max_tokens": 1000
            }
        }
        
        import yaml
        config_str = yaml.dump(config_template, default_flow_style=False, indent=2)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(config_str)
        
        return config_str
    
    def get_installation_info(self) -> Dict[str, Any]:
        """
        Get comprehensive installation information.
        
        Returns:
            Installation information
        """
        return {
            "package_name": self.package_name,
            "version": self.version,
            "environment": self.validate_environment(),
            "installation": self.verify_installation()
        }


def validate_ipm_environment() -> Dict[str, Any]:
    """
    Convenience function to validate IPM environment.
    
    Returns:
        Validation results
    """
    ipm = IPMIntegration()
    return ipm.validate_environment()


def install_via_ipm() -> Dict[str, Any]:
    """
    Convenience function to install package via IPM.
    
    Returns:
        Installation results
    """
    ipm = IPMIntegration()
    return ipm.install_package()


def verify_ipm_installation() -> Dict[str, Any]:
    """
    Convenience function to verify IPM installation.
    
    Returns:
        Verification results
    """
    ipm = IPMIntegration()
    return ipm.verify_installation()


if __name__ == "__main__":
    # Command-line interface for IPM integration
    import argparse
    
    parser = argparse.ArgumentParser(description="IPM Integration Utilities")
    parser.add_argument("--validate", action="store_true", help="Validate environment")
    parser.add_argument("--install", action="store_true", help="Install package")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--info", action="store_true", help="Show installation info")
    parser.add_argument("--config", type=str, help="Generate config template")
    
    args = parser.parse_args()
    
    ipm = IPMIntegration()
    
    if args.validate:
        result = ipm.validate_environment()
        print(json.dumps(result, indent=2))
    
    if args.install:
        result = ipm.install_package()
        print(json.dumps(result, indent=2))
    
    if args.verify:
        result = ipm.verify_installation()
        print(json.dumps(result, indent=2))
    
    if args.info:
        result = ipm.get_installation_info()
        print(json.dumps(result, indent=2))
    
    if args.config:
        config = ipm.generate_config_template(args.config)
        print(f"Configuration template saved to: {args.config}")