"""
Tests for the Configuration Template Engine.

This module tests the template inheritance system, environment variable injection,
and configuration validation functionality.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

from quick_start.config.template_engine import ConfigurationTemplateEngine
from quick_start.config.interfaces import (
    ConfigurationContext,
    ConfigurationError,
    ValidationError,
    TemplateNotFoundError,
    InheritanceError,
)


class TestConfigurationTemplateEngine:
    """Test suite for ConfigurationTemplateEngine."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create a temporary directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            
            # Create base configuration template
            base_config = {
                "metadata": {
                    "version": "1.0.0",
                    "description": "Base configuration"
                },
                "database": {
                    "iris": {
                        "host": "${IRIS_HOST:-localhost}",
                        "port": "${IRIS_PORT:-1972}",
                        "connection_pool": {
                            "max_connections": 10
                        }
                    }
                },
                "performance": {
                    "batch_size": 32,
                    "max_workers": 4
                }
            }
            
            # Create quick_start template that extends base
            quick_start_config = {
                "extends": "base_config",
                "metadata": {
                    "profile": "quick_start",
                    "description": "Quick start configuration"
                },
                "database": {
                    "iris": {
                        "connection_pool": {
                            "max_connections": 5  # Override base value
                        }
                    }
                },
                "sample_data": {
                    "enabled": True,
                    "document_count": "${SAMPLE_DOC_COUNT:-50}"
                },
                "performance": {
                    "batch_size": 16  # Override base value
                }
            }
            
            # Create quick_start_minimal that extends quick_start
            minimal_config = {
                "extends": "quick_start",
                "metadata": {
                    "profile": "quick_start_minimal",
                    "description": "Minimal quick start with 10 documents"
                },
                "sample_data": {
                    "document_count": 10  # Override parent value
                },
                "performance": {
                    "batch_size": 8,  # Override parent value
                    "max_workers": 1   # Override grandparent value
                }
            }
            
            # Write template files
            with open(template_dir / "base_config.yaml", "w") as f:
                yaml.dump(base_config, f)
            
            with open(template_dir / "quick_start.yaml", "w") as f:
                yaml.dump(quick_start_config, f)
                
            with open(template_dir / "quick_start_minimal.yaml", "w") as f:
                yaml.dump(minimal_config, f)
            
            yield template_dir

    @pytest.fixture
    def template_engine(self, temp_template_dir):
        """Create a ConfigurationTemplateEngine instance."""
        return ConfigurationTemplateEngine(temp_template_dir)

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing."""
        return {
            "IRIS_HOST": "test-host",
            "IRIS_PORT": "1973",
            "SAMPLE_DOC_COUNT": "25"
        }

    def test_template_engine_initialization(self, temp_template_dir):
        """Test that ConfigurationTemplateEngine can be initialized properly."""
        engine = ConfigurationTemplateEngine(temp_template_dir)
        
        assert engine is not None
        assert engine.template_dir == temp_template_dir
        assert hasattr(engine, 'load_template')
        assert hasattr(engine, 'resolve_template')
        assert hasattr(engine, 'validate_configuration')

    def test_load_single_template(self, template_engine):
        """Test loading a single template without inheritance."""
        config = template_engine.load_template("base_config")
        
        assert config is not None
        assert config["metadata"]["version"] == "1.0.0"
        assert config["database"]["iris"]["host"] == "${IRIS_HOST:-localhost}"
        assert config["performance"]["batch_size"] == 32

    def test_template_not_found_error(self, template_engine):
        """Test that loading a non-existent template raises appropriate error."""
        with pytest.raises(TemplateNotFoundError) as exc_info:
            template_engine.load_template("non_existent_template")
        
        assert "non_existent_template" in str(exc_info.value)

    def test_build_inheritance_chain(self, template_engine):
        """Test building inheritance chain for templates."""
        # Test simple inheritance chain
        chain = template_engine._build_inheritance_chain("quick_start")
        expected_chain = ["base_config", "quick_start"]
        assert chain == expected_chain
        
        # Test deeper inheritance chain
        chain = template_engine._build_inheritance_chain("quick_start_minimal")
        expected_chain = ["base_config", "quick_start", "quick_start_minimal"]
        assert chain == expected_chain

    def test_resolve_template_with_inheritance(self, template_engine, mock_env_vars):
        """Test resolving template with inheritance chain."""
        context = ConfigurationContext(
            profile="quick_start_minimal",
            environment="test",
            overrides={},
            template_path=template_engine.template_dir,
            environment_variables=mock_env_vars
        )
        
        config = template_engine.resolve_template(context)
        
        # Check that inheritance worked correctly
        assert config["metadata"]["profile"] == "quick_start_minimal"
        assert config["metadata"]["version"] == "1.0.0"  # From base
        
        # Check that overrides worked correctly
        assert config["database"]["iris"]["connection_pool"]["max_connections"] == 5  # From quick_start
        assert config["performance"]["batch_size"] == 8  # From minimal
        assert config["performance"]["max_workers"] == 1  # From minimal
        
        # Check that sample_data was inherited and overridden
        assert config["sample_data"]["enabled"] is True  # From quick_start
        assert config["sample_data"]["document_count"] == 10  # From minimal

    def test_environment_variable_injection(self, template_engine, mock_env_vars):
        """Test that environment variables are properly injected."""
        context = ConfigurationContext(
            profile="base_config",
            environment="test",
            overrides={},
            template_path=template_engine.template_dir,
            environment_variables=mock_env_vars
        )
        
        config = template_engine.resolve_template(context)
        
        # Check that environment variables were injected
        assert config["database"]["iris"]["host"] == "test-host"
        assert config["database"]["iris"]["port"] == 1973  # Should be converted to int

    def test_environment_variable_defaults(self, template_engine):
        """Test that default values are used when environment variables are not set."""
        context = ConfigurationContext(
            profile="base_config",
            environment="test",
            overrides={},
            template_path=template_engine.template_dir,
            environment_variables={}  # No environment variables
        )
        
        config = template_engine.resolve_template(context)
        
        # Check that default values were used
        assert config["database"]["iris"]["host"] == "localhost"
        assert config["database"]["iris"]["port"] == 1972  # Should be converted to int

    def test_context_overrides(self, template_engine, mock_env_vars):
        """Test that context overrides take precedence over template values."""
        overrides = {
            "performance": {
                "batch_size": 64,
                "timeout": 60
            },
            "custom_setting": "override_value"
        }
        
        context = ConfigurationContext(
            profile="quick_start",
            environment="test",
            overrides=overrides,
            template_path=template_engine.template_dir,
            environment_variables=mock_env_vars
        )
        
        config = template_engine.resolve_template(context)
        
        # Check that overrides were applied
        assert config["performance"]["batch_size"] == 64  # Overridden
        assert config["performance"]["max_workers"] == 4  # Not overridden, from base
        assert config["performance"]["timeout"] == 60  # New value from override
        assert config["custom_setting"] == "override_value"  # New setting

    def test_deep_merge_functionality(self, template_engine):
        """Test that deep merging works correctly for nested dictionaries."""
        # This test verifies the _deep_merge method behavior
        base = {
            "level1": {
                "level2": {
                    "keep_this": "base_value",
                    "override_this": "base_value"
                },
                "keep_level2": "base_value"
            },
            "keep_level1": "base_value"
        }
        
        override = {
            "level1": {
                "level2": {
                    "override_this": "override_value",
                    "new_value": "new"
                },
                "new_level2": "new"
            },
            "new_level1": "new"
        }
        
        result = template_engine._deep_merge(base, override)
        
        # Check that deep merge preserved and overrode correctly
        assert result["level1"]["level2"]["keep_this"] == "base_value"
        assert result["level1"]["level2"]["override_this"] == "override_value"
        assert result["level1"]["level2"]["new_value"] == "new"
        assert result["level1"]["keep_level2"] == "base_value"
        assert result["level1"]["new_level2"] == "new"
        assert result["keep_level1"] == "base_value"
        assert result["new_level1"] == "new"

    def test_get_available_profiles(self, template_engine):
        """Test getting list of available configuration profiles."""
        profiles = template_engine.get_available_profiles()
        
        assert "base_config" in profiles
        assert "quick_start" in profiles
        assert "quick_start_minimal" in profiles
        assert len(profiles) == 3

    def test_circular_inheritance_detection(self, temp_template_dir):
        """Test that circular inheritance is detected and raises an error."""
        # Create templates with circular inheritance
        circular_a = {
            "extends": "circular_b",
            "value": "a"
        }
        circular_b = {
            "extends": "circular_a",
            "value": "b"
        }
        
        with open(temp_template_dir / "circular_a.yaml", "w") as f:
            yaml.dump(circular_a, f)
        with open(temp_template_dir / "circular_b.yaml", "w") as f:
            yaml.dump(circular_b, f)
        
        engine = ConfigurationTemplateEngine(temp_template_dir)
        
        with pytest.raises(InheritanceError) as exc_info:
            engine._build_inheritance_chain("circular_a")
        
        assert "circular" in str(exc_info.value).lower()

    def test_invalid_yaml_handling(self, temp_template_dir):
        """Test handling of invalid YAML files."""
        # Create invalid YAML file
        invalid_yaml_path = temp_template_dir / "invalid.yaml"
        with open(invalid_yaml_path, "w") as f:
            f.write("invalid: yaml: content: [unclosed")
        
        engine = ConfigurationTemplateEngine(temp_template_dir)
        
        with pytest.raises(ConfigurationError) as exc_info:
            engine.load_template("invalid")
        
        assert "yaml" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_template_caching(self, template_engine):
        """Test that templates are cached after first load."""
        # Load template twice
        config1 = template_engine.load_template("base_config")
        config2 = template_engine.load_template("base_config")
        
        # Should be the same object (cached)
        assert config1 is config2

    def test_environment_variable_type_conversion(self, template_engine):
        """Test that environment variables are converted to appropriate types."""
        env_vars = {
            "INT_VALUE": "42",
            "FLOAT_VALUE": "3.14",
            "BOOL_TRUE": "true",
            "BOOL_FALSE": "false",
            "STRING_VALUE": "hello"
        }
        
        # Create template with various types
        template_config = {
            "settings": {
                "int_setting": "${INT_VALUE:-0}",
                "float_setting": "${FLOAT_VALUE:-0.0}",
                "bool_true_setting": "${BOOL_TRUE:-false}",
                "bool_false_setting": "${BOOL_FALSE:-true}",
                "string_setting": "${STRING_VALUE:-default}"
            }
        }
        
        template_path = template_engine.template_dir / "type_test.yaml"
        with open(template_path, "w") as f:
            yaml.dump(template_config, f)
        
        context = ConfigurationContext(
            profile="type_test",
            environment="test",
            overrides={},
            template_path=template_engine.template_dir,
            environment_variables=env_vars
        )
        
        config = template_engine.resolve_template(context)
        
        # Check type conversions
        assert config["settings"]["int_setting"] == 42
        assert config["settings"]["float_setting"] == 3.14
        assert config["settings"]["bool_true_setting"] is True
        assert config["settings"]["bool_false_setting"] is False
        assert config["settings"]["string_setting"] == "hello"