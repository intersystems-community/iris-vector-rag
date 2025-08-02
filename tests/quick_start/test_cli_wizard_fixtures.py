"""
Test fixtures and utilities for CLI wizard testing.

This module provides reusable fixtures, mock objects, and utility functions
for testing the Quick Start CLI wizard functionality.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import Quick Start components for mocking
from quick_start.config.template_engine import ConfigurationTemplateEngine
from quick_start.config.schema_validator import ConfigurationSchemaValidator
from quick_start.config.integration_factory import IntegrationFactory
from quick_start.data.sample_manager import SampleDataManager


@dataclass
class MockCLIResult:
    """Mock result object for CLI operations."""
    success: bool
    message: str
    data: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


@dataclass
class MockValidationResult:
    """Mock validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class MockConnectionResult:
    """Mock connection test result."""
    success: bool
    message: str
    error_message: Optional[str] = None
    response_time: Optional[float] = None


class CLIWizardTestFixtures:
    """Collection of test fixtures for CLI wizard testing."""
    
    @staticmethod
    @pytest.fixture
    def sample_profiles():
        """Sample profile configurations for testing."""
        return {
            "quick_start_minimal": {
                "metadata": {
                    "profile": "quick_start_minimal",
                    "version": "2024.1",
                    "description": "Minimal profile for basic testing"
                },
                "sample_data": {
                    "source": "pmc",
                    "document_count": 10,
                    "categories": ["biomedical"]
                },
                "database": {
                    "iris": {
                        "host": "localhost",
                        "port": 1972,
                        "namespace": "USER",
                        "username": "demo",
                        "password": "demo"
                    }
                },
                "embeddings": {
                    "provider": "openai",
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536
                },
                "llm": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                },
                "mcp_server": {
                    "enabled": True,
                    "port": 3000,
                    "tools": ["basic", "health_check"]
                }
            },
            "quick_start_standard": {
                "metadata": {
                    "profile": "quick_start_standard",
                    "version": "2024.1",
                    "description": "Standard profile for moderate workloads"
                },
                "sample_data": {
                    "source": "pmc",
                    "document_count": 100,
                    "categories": ["biomedical", "clinical"]
                },
                "database": {
                    "iris": {
                        "host": "localhost",
                        "port": 1972,
                        "namespace": "USER",
                        "username": "demo",
                        "password": "demo"
                    }
                },
                "embeddings": {
                    "provider": "openai",
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536
                },
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5
                },
                "mcp_server": {
                    "enabled": True,
                    "port": 3000,
                    "tools": ["basic", "health_check", "search", "analytics"]
                },
                "performance": {
                    "batch_size": 16,
                    "max_workers": 2
                }
            },
            "quick_start_extended": {
                "metadata": {
                    "profile": "quick_start_extended",
                    "version": "2024.1",
                    "description": "Extended profile for high-performance workloads"
                },
                "sample_data": {
                    "source": "pmc",
                    "document_count": 1000,
                    "categories": ["biomedical", "clinical", "research"]
                },
                "database": {
                    "iris": {
                        "host": "localhost",
                        "port": 1972,
                        "namespace": "USER",
                        "username": "demo",
                        "password": "demo"
                    }
                },
                "embeddings": {
                    "provider": "openai",
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536
                },
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet",
                    "temperature": 0.3
                },
                "mcp_server": {
                    "enabled": True,
                    "port": 3000,
                    "tools": ["basic", "health_check", "search", "analytics", "advanced", "monitoring"]
                },
                "performance": {
                    "batch_size": 32,
                    "max_workers": 4
                }
            }
        }
    
    @staticmethod
    @pytest.fixture
    def sample_environment_variables():
        """Sample environment variables for testing."""
        return {
            "IRIS_HOST": "localhost",
            "IRIS_PORT": "1972",
            "IRIS_NAMESPACE": "USER",
            "IRIS_USERNAME": "demo",
            "IRIS_PASSWORD": "demo",
            "OPENAI_API_KEY": "sk-test-key-12345",
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "EMBEDDING_MODEL": "text-embedding-ada-002",
            "LLM_MODEL": "gpt-3.5-turbo",
            "MCP_SERVER_PORT": "3000",
            "QUICK_START_PROFILE": "quick_start_standard",
            "QUICK_START_NON_INTERACTIVE": "false"
        }
    
    @staticmethod
    @pytest.fixture
    def mock_user_inputs():
        """Mock user inputs for interactive testing."""
        return {
            "minimal_profile_setup": [
                '1',  # Select minimal profile
                'localhost',  # Database host
                '1972',  # Database port
                'USER',  # Namespace
                'demo',  # Username
                '1',  # OpenAI LLM
                'gpt-3.5-turbo',  # Model
                '0.7',  # Temperature
                'y'  # Confirm setup
            ],
            "standard_profile_setup": [
                '2',  # Select standard profile
                '100',  # Document count
                'localhost',  # Database host
                '1972',  # Database port
                'USER',  # Namespace
                'demo',  # Username
                '1',  # OpenAI LLM
                'gpt-4',  # Model
                '0.5',  # Temperature
                '1',  # OpenAI embeddings
                'text-embedding-ada-002',  # Embedding model
                'y',  # Generate docker-compose
                'y'  # Confirm setup
            ],
            "extended_profile_setup": [
                '3',  # Select extended profile
                '1000',  # Document count
                'localhost',  # Database host
                '1972',  # Database port
                'USER',  # Namespace
                'demo',  # Username
                '2',  # Anthropic LLM
                'claude-3-sonnet',  # Model
                '0.3',  # Temperature
                '1',  # OpenAI embeddings
                'text-embedding-ada-002',  # Embedding model
                'y',  # Generate docker-compose
                'y',  # Generate sample data script
                'y'  # Confirm setup
            ],
            "custom_profile_setup": [
                '4',  # Select custom
                'my_custom_profile',  # Profile name
                '200',  # Document count
                'basic,search,analytics',  # Tools
                'localhost',  # Database host
                '1972',  # Database port
                'USER',  # Namespace
                'demo',  # Username
                '1',  # OpenAI LLM
                'gpt-4',  # Model
                '0.6',  # Temperature
                'y'  # Confirm
            ],
            "error_recovery": [
                '5',  # Invalid selection
                '1',  # Valid selection
                'invalid-host',  # Invalid host
                'localhost',  # Valid host
                'invalid-port',  # Invalid port
                '1972',  # Valid port
                'y'  # Confirm
            ]
        }
    
    @staticmethod
    @pytest.fixture
    def mock_cli_arguments():
        """Mock CLI arguments for non-interactive testing."""
        return {
            "minimal_non_interactive": [
                '--profile', 'minimal',
                '--database-host', 'localhost',
                '--database-port', '1972',
                '--database-namespace', 'USER',
                '--database-username', 'demo',
                '--database-password', 'demo',
                '--llm-provider', 'openai',
                '--llm-model', 'gpt-3.5-turbo',
                '--non-interactive'
            ],
            "standard_with_overrides": [
                '--profile', 'standard',
                '--document-count', '200',
                '--database-host', 'localhost',
                '--database-port', '1972',
                '--llm-provider', 'openai',
                '--llm-model', 'gpt-4',
                '--embedding-provider', 'openai',
                '--embedding-model', 'text-embedding-ada-002',
                '--generate-docker-compose',
                '--non-interactive'
            ],
            "extended_production": [
                '--profile', 'extended',
                '--document-count', '1000',
                '--database-host', 'prod.iris.host',
                '--database-port', '1972',
                '--database-namespace', 'PROD',
                '--llm-provider', 'anthropic',
                '--llm-model', 'claude-3-sonnet',
                '--embedding-provider', 'openai',
                '--output-dir', '/opt/rag-config',
                '--generate-docker-compose',
                '--generate-sample-script',
                '--non-interactive'
            ],
            "validation_only": [
                '--validate-only',
                '--config', 'test_config.yaml'
            ],
            "help_commands": [
                '--help'
            ],
            "list_commands": [
                '--list-profiles'
            ]
        }


class MockQuickStartComponents:
    """Mock implementations of Quick Start components for testing."""
    
    @staticmethod
    def create_mock_template_engine(sample_profiles):
        """Create a mock template engine."""
        engine = Mock(spec=ConfigurationTemplateEngine)
        engine.get_available_profiles.return_value = list(sample_profiles.keys())
        
        def mock_resolve_template(context):
            profile = context.profile
            if profile in sample_profiles:
                return sample_profiles[profile]
            else:
                raise ValueError(f"Profile not found: {profile}")
        
        engine.resolve_template.side_effect = mock_resolve_template
        engine.load_template.side_effect = lambda name: sample_profiles.get(name, {})
        
        return engine
    
    @staticmethod
    def create_mock_schema_validator():
        """Create a mock schema validator."""
        validator = Mock(spec=ConfigurationSchemaValidator)
        validator.validate_configuration.return_value = True
        validator.get_validation_errors.return_value = []
        
        def mock_validate_with_errors(config, schema_name="base_config", profile=None):
            # Simulate validation errors for invalid configurations
            if not config.get("metadata"):
                raise ValueError("Missing metadata section")
            if not config.get("database"):
                raise ValueError("Missing database section")
            return True
        
        validator.validate_configuration.side_effect = mock_validate_with_errors
        
        return validator
    
    @staticmethod
    def create_mock_integration_factory():
        """Create a mock integration factory."""
        factory = Mock(spec=IntegrationFactory)
        
        def mock_integrate_template(template_name, target_manager, **kwargs):
            result = Mock()
            result.success = True
            result.converted_config = {"integrated": True, "template": template_name}
            result.errors = []
            result.warnings = []
            result.metadata = {"integration_completed": True}
            return result
        
        factory.integrate_template.side_effect = mock_integrate_template
        
        return factory
    
    @staticmethod
    def create_mock_sample_manager():
        """Create a mock sample data manager."""
        manager = Mock(spec=SampleDataManager)
        manager.get_available_sources.return_value = [
            {"type": "pmc", "name": "PMC API", "available": True},
            {"type": "local_cache", "name": "Local Cache", "available": True},
            {"type": "custom_set", "name": "Custom Dataset", "available": True}
        ]
        
        async def mock_estimate_requirements(config):
            doc_count = config.get("document_count", 100)
            return {
                "disk_space": doc_count * 50 * 1024,  # 50KB per doc
                "memory": max(512 * 1024 * 1024, doc_count * 1024),  # At least 512MB
                "estimated_time": doc_count * 2.0,  # 2 seconds per doc
                "network_bandwidth": doc_count * 50 * 1024
            }
        
        manager.estimate_requirements.side_effect = mock_estimate_requirements
        
        return manager


class CLIWizardTestUtilities:
    """Utility functions for CLI wizard testing."""
    
    @staticmethod
    def create_test_config_file(config: Dict[str, Any], file_path: Path) -> Path:
        """Create a test configuration file."""
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return file_path
    
    @staticmethod
    def create_test_env_file(env_vars: Dict[str, str], file_path: Path) -> Path:
        """Create a test environment file."""
        with open(file_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        return file_path
    
    @staticmethod
    def create_test_docker_compose(config: Dict[str, Any], file_path: Path) -> Path:
        """Create a test docker-compose file."""
        docker_config = {
            "version": "3.8",
            "services": {
                "iris": {
                    "image": "intersystemsdc/iris-community:latest",
                    "ports": [f"{config.get('database', {}).get('iris', {}).get('port', 1972)}:1972"],
                    "environment": {
                        "IRIS_USERNAME": config.get('database', {}).get('iris', {}).get('username', 'demo'),
                        "IRIS_PASSWORD": config.get('database', {}).get('iris', {}).get('password', 'demo')
                    }
                },
                "rag-app": {
                    "build": ".",
                    "ports": [f"{config.get('mcp_server', {}).get('port', 3000)}:3000"],
                    "depends_on": ["iris"],
                    "environment": {
                        "IRIS_HOST": "iris",
                        "IRIS_PORT": "1972"
                    }
                }
            }
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(docker_config, f, default_flow_style=False)
        return file_path
    
    @staticmethod
    def create_test_sample_script(config: Dict[str, Any], file_path: Path) -> Path:
        """Create a test sample data script."""
        script_content = f"""#!/usr/bin/env python3
\"\"\"
Sample data setup script generated by Quick Start CLI wizard.
\"\"\"

import asyncio
from quick_start.data.sample_manager import SampleDataManager
from quick_start.data.interfaces import SampleDataConfig, DataSourceType

async def main():
    # Configuration
    config = SampleDataConfig(
        source_type=DataSourceType.{config.get('sample_data', {}).get('source', 'PMC').upper()},
        document_count={config.get('sample_data', {}).get('document_count', 100)},
        categories={config.get('sample_data', {}).get('categories', ['biomedical'])},
        storage_path="./sample_data"
    )
    
    # Initialize manager
    manager = SampleDataManager(None)
    
    # Download samples
    print("Downloading sample documents...")
    documents = await manager.download_samples(config)
    print(f"Downloaded {{len(documents)}} documents")
    
    # Validate samples
    print("Validating documents...")
    validation_result = await manager.validate_samples(config.storage_path)
    if validation_result.is_valid:
        print("All documents are valid")
    else:
        print(f"Validation errors: {{validation_result.errors}}")
    
    # Ingest samples
    print("Ingesting documents into database...")
    ingestion_result = await manager.ingest_samples(config.storage_path, config)
    print(f"Ingested {{ingestion_result.documents_ingested}} documents")

if __name__ == "__main__":
    asyncio.run(main())
"""
        
        with open(file_path, 'w') as f:
            f.write(script_content)
        file_path.chmod(0o755)  # Make executable
        return file_path
    
    @staticmethod
    def mock_connection_test(config: Dict[str, Any], success: bool = True) -> MockConnectionResult:
        """Mock connection test result."""
        if success:
            return MockConnectionResult(
                success=True,
                message="Connection successful",
                response_time=0.123
            )
        else:
            return MockConnectionResult(
                success=False,
                message="Connection failed",
                error_message="Connection refused"
            )
    
    @staticmethod
    def mock_validation_result(config: Dict[str, Any], is_valid: bool = True) -> MockValidationResult:
        """Mock validation result."""
        if is_valid:
            return MockValidationResult(
                is_valid=True,
                errors=[],
                warnings=[]
            )
        else:
            errors = []
            if not config.get("metadata"):
                errors.append("Missing metadata section")
            if not config.get("database"):
                errors.append("Missing database configuration")
            
            return MockValidationResult(
                is_valid=False,
                errors=errors,
                warnings=["Configuration may be incomplete"]
            )
    
    @staticmethod
    def assert_config_structure(config: Dict[str, Any], profile: str):
        """Assert that configuration has expected structure for profile."""
        assert "metadata" in config
        assert config["metadata"]["profile"] == profile
        assert "database" in config
        assert "iris" in config["database"]
        assert "embeddings" in config
        assert "llm" in config
        
        if profile == "quick_start_minimal":
            assert config["sample_data"]["document_count"] <= 50
            assert len(config["mcp_server"]["tools"]) <= 3
        elif profile == "quick_start_standard":
            assert config["sample_data"]["document_count"] <= 500
            assert "performance" in config
        elif profile == "quick_start_extended":
            assert config["sample_data"]["document_count"] <= 5000
            assert "advanced" in config["mcp_server"]["tools"]
    
    @staticmethod
    def assert_files_created(output_dir: Path, expected_files: List[str]):
        """Assert that expected files were created in output directory."""
        for file_name in expected_files:
            file_path = output_dir / file_name
            assert file_path.exists(), f"Expected file not created: {file_name}"
            assert file_path.stat().st_size > 0, f"File is empty: {file_name}"


# Export fixtures for use in test files
@pytest.fixture
def sample_profiles():
    return CLIWizardTestFixtures.sample_profiles()

@pytest.fixture
def sample_environment_variables():
    return CLIWizardTestFixtures.sample_environment_variables()

@pytest.fixture
def mock_user_inputs():
    return CLIWizardTestFixtures.mock_user_inputs()

@pytest.fixture
def mock_cli_arguments():
    return CLIWizardTestFixtures.mock_cli_arguments()

@pytest.fixture
def mock_template_engine(sample_profiles):
    return MockQuickStartComponents.create_mock_template_engine(sample_profiles)

@pytest.fixture
def mock_schema_validator():
    return MockQuickStartComponents.create_mock_schema_validator()

@pytest.fixture
def mock_integration_factory():
    return MockQuickStartComponents.create_mock_integration_factory()

@pytest.fixture
def mock_sample_manager():
    return MockQuickStartComponents.create_mock_sample_manager()