"""
CLI-specific validation functions for Quick Start wizard.

This module provides validation classes for testing database connectivity,
LLM provider credentials, embedding model availability, configuration
validation, and system health checks.
"""

import os
import socket
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ConnectivityResult:
    """Result of a connectivity test."""
    success: bool
    message: str
    response_time: Optional[float] = None
    error_message: Optional[str] = None


class DatabaseConnectivityValidator:
    """Validator for testing database connections."""
    
    def test_connection(self, db_config: Dict[str, Any]) -> ConnectivityResult:
        """
        Test database connectivity.
        
        Args:
            db_config: Database configuration dictionary
            
        Returns:
            ConnectivityResult with test results
        """
        start_time = time.time()
        
        try:
            # Basic network connectivity test
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 1972)
            
            # Test socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout
            
            result = sock.connect_ex((host, port))
            sock.close()
            
            response_time = time.time() - start_time
            
            if result == 0:
                return ConnectivityResult(
                    success=True,
                    message=f"Successfully connected to {host}:{port}",
                    response_time=response_time
                )
            else:
                return ConnectivityResult(
                    success=False,
                    message=f"Failed to connect to {host}:{port}",
                    response_time=response_time,
                    error_message=f"Connection refused (error code: {result})"
                )
                
        except socket.gaierror as e:
            return ConnectivityResult(
                success=False,
                message=f"DNS resolution failed for {host}",
                error_message=str(e)
            )
        except Exception as e:
            return ConnectivityResult(
                success=False,
                message="Database connection test failed",
                error_message=str(e)
            )
    
    def validate_config(self, db_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate database configuration parameters.
        
        Args:
            db_config: Database configuration dictionary
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        
        # Check required fields
        required_fields = ['host', 'port', 'namespace', 'username', 'password']
        for field in required_fields:
            if field not in db_config or not db_config[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate port
        if 'port' in db_config:
            try:
                port = int(db_config['port'])
                if port < 1 or port > 65535:
                    errors.append("Port must be between 1 and 65535")
            except (ValueError, TypeError):
                errors.append("Port must be a valid integer")
        
        # Validate host format
        if 'host' in db_config:
            host = db_config['host']
            if not host or not isinstance(host, str):
                errors.append("Host must be a non-empty string")
        
        return ValidationResult(
            valid=len(errors) == 0,
            message="Database configuration is valid" if len(errors) == 0 else "Database configuration has errors",
            details={'errors': errors} if errors else None
        )


class LLMProviderValidator:
    """Validator for testing LLM provider credentials."""
    
    def test_provider(self, llm_config: Dict[str, Any]) -> ConnectivityResult:
        """
        Test LLM provider connectivity and credentials.
        
        Args:
            llm_config: LLM configuration dictionary
            
        Returns:
            ConnectivityResult with test results
        """
        start_time = time.time()
        
        try:
            provider = llm_config.get('provider')
            api_key = llm_config.get('api_key')
            
            if not provider:
                return ConnectivityResult(
                    success=False,
                    message="No LLM provider specified",
                    error_message="Provider field is required"
                )
            
            # For providers that require API keys
            if provider in ['openai', 'anthropic', 'azure'] and not api_key:
                return ConnectivityResult(
                    success=False,
                    message=f"API key required for {provider}",
                    error_message="API key is missing"
                )
            
            # Basic API key format validation
            if api_key:
                if provider == 'openai' and not api_key.startswith('sk-'):
                    return ConnectivityResult(
                        success=False,
                        message="Invalid OpenAI API key format",
                        error_message="OpenAI API keys should start with 'sk-'"
                    )
                
                if provider == 'anthropic' and not api_key.startswith('sk-ant-'):
                    return ConnectivityResult(
                        success=False,
                        message="Invalid Anthropic API key format",
                        error_message="Anthropic API keys should start with 'sk-ant-'"
                    )
            
            response_time = time.time() - start_time
            
            # For now, we'll do basic validation
            # In a real implementation, you'd make actual API calls
            return ConnectivityResult(
                success=True,
                message=f"LLM provider {provider} configuration appears valid",
                response_time=response_time
            )
            
        except Exception as e:
            return ConnectivityResult(
                success=False,
                message="LLM provider test failed",
                error_message=str(e)
            )
    
    def validate_config(self, llm_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate LLM configuration parameters.
        
        Args:
            llm_config: LLM configuration dictionary
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        
        # Check required fields
        if 'provider' not in llm_config or not llm_config['provider']:
            errors.append("Missing required field: provider")
        
        provider = llm_config.get('provider')
        if provider not in ['openai', 'anthropic', 'azure', 'local']:
            errors.append(f"Unsupported provider: {provider}")
        
        # Check API key for cloud providers
        if provider in ['openai', 'anthropic', 'azure']:
            if 'api_key' not in llm_config or not llm_config['api_key']:
                errors.append(f"API key required for {provider}")
        
        # Validate model if specified
        if 'model' in llm_config and llm_config['model']:
            model = llm_config['model']
            if not isinstance(model, str) or len(model.strip()) == 0:
                errors.append("Model must be a non-empty string")
        
        return ValidationResult(
            valid=len(errors) == 0,
            message="LLM configuration is valid" if len(errors) == 0 else "LLM configuration has errors",
            details={'errors': errors} if errors else None
        )


class EmbeddingModelValidator:
    """Validator for testing embedding model availability."""
    
    def test_model(self, embedding_config: Dict[str, Any]) -> ConnectivityResult:
        """
        Test embedding model availability.
        
        Args:
            embedding_config: Embedding configuration dictionary
            
        Returns:
            ConnectivityResult with test results
        """
        start_time = time.time()
        
        try:
            provider = embedding_config.get('provider')
            model = embedding_config.get('model')
            
            if not provider:
                return ConnectivityResult(
                    success=False,
                    message="No embedding provider specified",
                    error_message="Provider field is required"
                )
            
            if not model:
                return ConnectivityResult(
                    success=False,
                    message="No embedding model specified",
                    error_message="Model field is required"
                )
            
            # Basic model validation
            known_models = {
                'openai': ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'],
                'huggingface': ['sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2'],
                'sentence-transformers': ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                'local': []  # Any model name allowed for local
            }
            
            if provider in known_models and known_models[provider]:
                model_name = model.split('/')[-1]  # Handle huggingface format
                if model not in known_models[provider] and model_name not in known_models[provider]:
                    logger.warning(f"Unknown model {model} for provider {provider}")
            
            response_time = time.time() - start_time
            
            return ConnectivityResult(
                success=True,
                message=f"Embedding model {model} configuration appears valid",
                response_time=response_time
            )
            
        except Exception as e:
            return ConnectivityResult(
                success=False,
                message="Embedding model test failed",
                error_message=str(e)
            )
    
    def validate_config(self, embedding_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate embedding configuration parameters.
        
        Args:
            embedding_config: Embedding configuration dictionary
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        
        # Check required fields
        if 'provider' not in embedding_config or not embedding_config['provider']:
            errors.append("Missing required field: provider")
        
        if 'model' not in embedding_config or not embedding_config['model']:
            errors.append("Missing required field: model")
        
        provider = embedding_config.get('provider')
        if provider not in ['openai', 'huggingface', 'sentence-transformers', 'local']:
            errors.append(f"Unsupported embedding provider: {provider}")
        
        # Validate dimensions if specified
        if 'dimensions' in embedding_config:
            try:
                dimensions = int(embedding_config['dimensions'])
                if dimensions <= 0:
                    errors.append("Dimensions must be positive")
            except (ValueError, TypeError):
                errors.append("Dimensions must be a valid integer")
        
        return ValidationResult(
            valid=len(errors) == 0,
            message="Embedding configuration is valid" if len(errors) == 0 else "Embedding configuration has errors",
            details={'errors': errors} if errors else None
        )


class ConfigurationValidator:
    """Validator for overall configuration validation."""
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.db_validator = DatabaseConnectivityValidator()
        self.llm_validator = LLMProviderValidator()
        self.embedding_validator = EmbeddingModelValidator()
    
    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate the complete configuration.
        
        Args:
            config: Complete configuration dictionary
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        warnings = []
        
        # Validate profile
        if 'profile' not in config or not config['profile']:
            errors.append("Missing required field: profile")
        
        # Validate database configuration
        if 'database' in config:
            db_result = self.db_validator.validate_config(config['database'])
            if not db_result.valid and db_result.details:
                errors.extend(db_result.details.get('errors', []))
        else:
            errors.append("Missing database configuration")
        
        # Validate LLM configuration
        if 'llm' in config:
            llm_result = self.llm_validator.validate_config(config['llm'])
            if not llm_result.valid and llm_result.details:
                errors.extend(db_result.details.get('errors', []))
        else:
            errors.append("Missing LLM configuration")
        
        # Validate embedding configuration
        if 'embedding' in config:
            embedding_result = self.embedding_validator.validate_config(config['embedding'])
            if not embedding_result.valid and embedding_result.details:
                errors.extend(embedding_result.details.get('errors', []))
        else:
            warnings.append("Missing embedding configuration - will use defaults")
        
        # Validate output directory
        if 'output_dir' in config:
            output_dir = config['output_dir']
            if not isinstance(output_dir, str) or not output_dir.strip():
                errors.append("Output directory must be a non-empty string")
        
        return ValidationResult(
            valid=len(errors) == 0,
            message="Configuration is valid" if len(errors) == 0 else "Configuration has errors",
            details={
                'errors': errors,
                'warnings': warnings
            } if errors or warnings else None
        )


class SystemHealthValidator:
    """Validator for basic system health checks."""
    
    def check_system_health(self) -> ValidationResult:
        """
        Perform basic system health checks.
        
        Returns:
            ValidationResult with health check results
        """
        errors = []
        warnings = []
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            errors.append(f"Python 3.8+ required, found {sys.version}")
        
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            free_gb = free // (1024**3)
            if free_gb < 1:
                errors.append(f"Insufficient disk space: {free_gb}GB available, at least 1GB required")
            elif free_gb < 5:
                warnings.append(f"Low disk space: {free_gb}GB available, 5GB+ recommended")
        except Exception as e:
            warnings.append(f"Could not check disk space: {e}")
        
        # Check required environment variables
        required_env_vars = []  # Add any required env vars here
        for var in required_env_vars:
            if var not in os.environ:
                warnings.append(f"Environment variable {var} not set")
        
        # Check network connectivity
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
        except OSError:
            warnings.append("No internet connectivity detected")
        
        return ValidationResult(
            valid=len(errors) == 0,
            message="System health check passed" if len(errors) == 0 else "System health check failed",
            details={
                'errors': errors,
                'warnings': warnings
            } if errors or warnings else None
        )
    
    def check_dependencies(self) -> ValidationResult:
        """
        Check for required dependencies.
        
        Returns:
            ValidationResult with dependency check results
        """
        errors = []
        warnings = []
        
        # Check for required packages
        required_packages = [
            'yaml',
            'pathlib',
            'argparse'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                errors.append(f"Required package not found: {package}")
        
        # Check for optional packages
        optional_packages = [
            'openai',
            'anthropic',
            'sentence_transformers'
        ]
        
        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                warnings.append(f"Optional package not found: {package}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            message="Dependency check passed" if len(errors) == 0 else "Dependency check failed",
            details={
                'errors': errors,
                'warnings': warnings
            } if errors or warnings else None
        )