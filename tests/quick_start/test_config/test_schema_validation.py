"""
Tests for JSON schema validation framework in the Quick Start configuration system.

This module tests the schema validation capabilities that ensure configuration
templates conform to expected structures and data types.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from quick_start.config.interfaces import (
    ConfigurationContext,
    ValidationError
)
from quick_start.config.template_engine import ConfigurationTemplateEngine


class TestSchemaValidation:
    """Test suite for JSON schema validation functionality."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create a temporary directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            
            # Create a simple base template
            base_template = {
                "metadata": {
                    "version": "1.0.0",
                    "schema_version": "2024.1"
                },
                "database": {
                    "iris": {
                        "host": "${IRIS_HOST:-localhost}",
                        "port": "${IRIS_PORT:-1972}",
                        "namespace": "${IRIS_NAMESPACE:-USER}"
                    }
                },
                "storage": {
                    "iris": {
                        "table_name": "${IRIS_TABLE_NAME:-RAG.SourceDocuments}",
                        "vector_dimension": "${VECTOR_DIMENSION:-384}"
                    }
                }
            }
            
            # Create a template with invalid structure
            invalid_template = {
                "metadata": {
                    "version": "1.0.0"
                    # Missing required schema_version
                },
                "database": {
                    "iris": {
                        "host": "localhost",
                        "port": "invalid_port_type",  # Should be integer
                        "namespace": "USER"
                    }
                }
            }
            
            # Create a template with missing required fields
            incomplete_template = {
                "metadata": {
                    "version": "1.0.0",
                    "schema_version": "2024.1"
                }
                # Missing required database section
            }
            
            # Copy test template files from test_data directory if they exist
            test_data_dir = Path(__file__).parent.parent / "test_data"
            import shutil
            
            # Try to copy existing test templates first
            if (test_data_dir / "valid_template.yaml").exists():
                shutil.copy2(test_data_dir / "valid_template.yaml", template_dir)
            else:
                # Fallback to creating base_template as valid_template
                with open(template_dir / "valid_template.yaml", "w") as f:
                    import yaml
                    yaml.dump(base_template, f)
                    
            if (test_data_dir / "invalid_template.yaml").exists():
                shutil.copy2(test_data_dir / "invalid_template.yaml", template_dir)
            else:
                # Fallback to creating invalid_template
                with open(template_dir / "invalid_template.yaml", "w") as f:
                    import yaml
                    yaml.dump(invalid_template, f)
                
            # Always create incomplete_template for specific tests
            with open(template_dir / "incomplete_template.yaml", "w") as f:
                import yaml
                yaml.dump(incomplete_template, f)
            
            yield template_dir

    @pytest.fixture
    def schema_validator(self, temp_template_dir):
        """Create a schema validator instance."""
        from quick_start.config.schema_validator import ConfigurationSchemaValidator
        return ConfigurationSchemaValidator()

    @pytest.fixture
    def template_engine_with_validation(self, temp_template_dir):
        """Create a template engine with schema validation enabled."""
        engine = ConfigurationTemplateEngine(template_dir=temp_template_dir)
        # Enable schema validation
        engine.enable_schema_validation = True
        return engine

    def test_schema_validator_initialization(self, schema_validator):
        """Test that schema validator initializes correctly."""
        assert schema_validator is not None
        assert hasattr(schema_validator, 'validate_configuration')
        assert hasattr(schema_validator, 'load_schema')

    def test_load_base_configuration_schema(self, schema_validator):
        """Test loading the base configuration schema."""
        schema = schema_validator.load_schema('base_config')
        
        assert schema is not None
        assert isinstance(schema, dict)
        assert 'type' in schema
        assert schema['type'] == 'object'
        assert 'properties' in schema
        
        # Check for required top-level properties
        properties = schema['properties']
        assert 'metadata' in properties
        assert 'database' in properties
        assert 'storage' in properties

    def test_validate_valid_configuration(self, schema_validator, temp_template_dir):
        """Test validation of a valid configuration."""
        # Load a valid configuration
        import yaml
        with open(temp_template_dir / "valid_template.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Should not raise any exceptions
        result = schema_validator.validate_configuration(config, 'base_config')
        assert result is True

    def test_validate_invalid_configuration_type_error(self, schema_validator, temp_template_dir):
        """Test validation fails for configuration with type errors."""
        import yaml
        with open(temp_template_dir / "invalid_template.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Should raise ValidationError due to invalid port type
        with pytest.raises(ValidationError) as exc_info:
            schema_validator.validate_configuration(config, 'base_config')
        
        assert "port" in str(exc_info.value).lower()
        assert "type" in str(exc_info.value).lower()

    def test_validate_incomplete_configuration(self, schema_validator, temp_template_dir):
        """Test validation fails for configuration missing required fields."""
        import yaml
        with open(temp_template_dir / "incomplete_template.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Should raise ValidationError due to missing required fields
        with pytest.raises(ValidationError) as exc_info:
            schema_validator.validate_configuration(config, 'base_config')
        
        assert "required" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()

    def test_validate_configuration_with_environment_variables(self, temp_template_dir):
        """Test validation works with environment variable placeholders through template engine."""
        from quick_start.config.template_engine import ConfigurationTemplateEngine
        from quick_start.config.interfaces import ConfigurationContext
        
        # Create a template with environment variables
        template_content = """
metadata:
  version: "1.0.0"
  schema_version: "2024.1"

database:
  iris:
    host: "${IRIS_HOST:-localhost}"
    port: "${IRIS_PORT:-1972}"
    namespace: "${IRIS_NAMESPACE:-USER}"
    username: "${IRIS_USERNAME:-admin}"
    password: "${IRIS_PASSWORD:-password}"

storage:
  data_directory: "${DATA_DIR:-./data}"

vector_index:
  dimension: "${VECTOR_DIMENSION:-1536}"
  metric: "cosine"

embeddings:
  provider: "openai"
  model: "text-embedding-ada-002"
  dimension: 1536

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000
"""
        
        # Write template file
        template_file = temp_template_dir / "env_vars_template.yaml"
        template_file.write_text(template_content)
        
        # Create template engine with validation enabled
        engine = ConfigurationTemplateEngine(template_dir=temp_template_dir)
        engine.enable_schema_validation = True
        
        context = ConfigurationContext(
            profile="env_vars_template",
            environment="development",
            overrides={},
            template_path=temp_template_dir,
            environment_variables={}
        )
        
        # Should resolve and validate successfully with environment variables processed
        config = engine.resolve_template(context)
        assert config is not None
        assert config["database"]["iris"]["host"] == "localhost"  # Default value
        assert config["database"]["iris"]["port"] == 1972  # Converted to int
        assert config["storage"]["data_directory"] == "./data"  # Default value

    def test_template_engine_with_schema_validation_enabled(self, template_engine_with_validation, temp_template_dir):
        """Test template engine with schema validation enabled."""
        context = ConfigurationContext(
            profile="valid_template",
            environment="development",
            overrides={},
            template_path=temp_template_dir,
            environment_variables={}
        )
        
        # Should resolve successfully for valid template
        config = template_engine_with_validation.resolve_template(context)
        assert config is not None
        assert "metadata" in config
        assert "database" in config

    def test_template_engine_validation_failure(self, template_engine_with_validation, temp_template_dir):
        """Test template engine fails validation for invalid templates."""
        context = ConfigurationContext(
            profile="invalid_template",
            environment="development",
            overrides={},
            template_path=temp_template_dir,
            environment_variables={}
        )
        
        # Should raise ValidationError for invalid template
        with pytest.raises(ValidationError):
            template_engine_with_validation.resolve_template(context)

    def test_schema_validation_can_be_disabled(self, temp_template_dir):
        """Test that schema validation can be disabled."""
        engine = ConfigurationTemplateEngine(template_dir=temp_template_dir)
        engine.enable_schema_validation = False
        
        context = ConfigurationContext(
            profile="invalid_template",
            environment="development",
            overrides={},
            template_path=temp_template_dir,
            environment_variables={}
        )
        
        # Should succeed even with invalid template when validation is disabled
        config = engine.resolve_template(context)
        assert config is not None

    def test_validate_quick_start_profile_schema(self, schema_validator):
        """Test validation of quick start profile specific schema."""
        quick_start_config = {
            "extends": "base_config",
            "metadata": {
                "profile": "quick_start",
                "description": "Quick start configuration"
            },
            "sample_data": {
                "enabled": True,
                "document_count": 100,
                "source": "pmc_sample"
            },
            "mcp_server": {
                "enabled": True,
                "port": 8080,
                "tools": ["basic", "hyde", "health_check"]
            }
        }
        
        # Should validate successfully
        result = schema_validator.validate_configuration(quick_start_config, 'quick_start')
        assert result is True

    def test_validate_profile_specific_constraints(self, schema_validator):
        """Test validation of profile-specific constraints."""
        # Test minimal profile constraints
        minimal_config = {
            "extends": "quick_start",
            "metadata": {
                "profile": "quick_start_minimal"
            },
            "sample_data": {
                "document_count": 10  # Should be <= 50 for minimal
            },
            "mcp_server": {
                "tools": ["basic", "health_check"]  # Limited tools for minimal
            }
        }
        
        result = schema_validator.validate_configuration(minimal_config, 'quick_start_minimal')
        assert result is True
        
        # Test invalid minimal profile (too many documents)
        invalid_minimal = {
            "extends": "quick_start",
            "metadata": {
                "profile": "quick_start_minimal"
            },
            "sample_data": {
                "document_count": 1000  # Too many for minimal profile
            }
        }
        
        with pytest.raises(ValidationError):
            schema_validator.validate_configuration(invalid_minimal, 'quick_start_minimal')

    def test_schema_version_compatibility(self, schema_validator):
        """Test schema version compatibility checking."""
        # Test with supported schema version - complete valid configuration
        config_v1 = {
            "metadata": {
                "version": "1.0.0",
                "schema_version": "2024.1"
            },
            "database": {
                "iris": {
                    "host": "localhost",
                    "port": 1972,
                    "namespace": "USER",
                    "username": "admin",
                    "password": "password"
                }
            },
            "storage": {
                "data_directory": "./data"
            },
            "vector_index": {
                "dimension": 1536,
                "metric": "cosine"
            },
            "embeddings": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "dimension": 1536
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        result = schema_validator.validate_configuration(config_v1, 'base_config')
        assert result is True
        
        # Test with unsupported schema version - complete valid configuration
        config_unsupported = {
            "metadata": {
                "version": "1.0.0",
                "schema_version": "2025.1"  # Future version
            },
            "database": {
                "iris": {
                    "host": "localhost",
                    "port": 1972,
                    "namespace": "USER",
                    "username": "admin",
                    "password": "password"
                }
            },
            "storage": {
                "data_directory": "./data"
            },
            "vector_index": {
                "dimension": 1536,
                "metric": "cosine"
            },
            "embeddings": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "dimension": 1536
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            schema_validator.validate_configuration(config_unsupported, 'base_config')
        
        assert "schema_version" in str(exc_info.value).lower()

    def test_custom_validation_rules(self, schema_validator):
        """Test custom validation rules beyond basic JSON schema."""
        # Test with a complete valid config but with invalid vector dimension
        config_invalid_vector_dim = {
            "metadata": {
                "version": "1.0.0",
                "schema_version": "2024.1"
            },
            "database": {
                "iris": {
                    "host": "localhost",
                    "port": 1972,
                    "namespace": "USER",
                    "username": "admin",
                    "password": "password"
                }
            },
            "storage": {
                "data_directory": "./data"
            },
            "vector_index": {
                "dimension": 0,  # Invalid dimension - should be >= 1
                "metric": "cosine"
            },
            "embeddings": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "dimension": 1536
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            schema_validator.validate_configuration(config_invalid_vector_dim, 'base_config')
        
        # Should fail due to invalid vector dimension
        assert "dimension" in str(exc_info.value).lower()

    def test_validation_error_details(self, schema_validator):
        """Test that validation errors provide detailed information."""
        invalid_config = {
            "metadata": {
                "version": "1.0.0"
                # Missing schema_version
            },
            "database": {
                "iris": {
                    "host": "localhost",
                    "port": "invalid",  # Wrong type
                    "namespace": "USER"
                }
            }
            # Missing storage section
        }
        
        with pytest.raises(ValidationError) as exc_info:
            schema_validator.validate_configuration(invalid_config, 'base_config')
        
        error_message = str(exc_info.value)
        # Should contain specific field information
        assert any(field in error_message.lower() for field in ['port', 'schema_version', 'storage'])
        # Should contain path information
        assert 'database' in error_message.lower() or 'metadata' in error_message.lower()