"""
Tests for Quick Start profile template files.

This module tests that the actual template files (base_config.yaml, quick_start.yaml,
and the three profile variants) work correctly with the template engine.
"""

import pytest
import os
from pathlib import Path

from quick_start.config.template_engine import ConfigurationTemplateEngine
from quick_start.config.interfaces import ConfigurationContext


class TestProfileTemplates:
    """Test the actual Quick Start profile template files."""

    @pytest.fixture
    def template_engine(self):
        """Create a template engine using the real template directory."""
        template_dir = Path(__file__).parent.parent.parent.parent / "quick_start" / "config" / "templates"
        return ConfigurationTemplateEngine(template_dir=template_dir)

    @pytest.fixture
    def base_context(self):
        """Create a basic configuration context."""
        template_dir = Path(__file__).parent.parent.parent.parent / "quick_start" / "config" / "templates"
        return ConfigurationContext(
            profile="quick_start",
            environment="development",
            overrides={},
            template_path=template_dir,
            environment_variables={
                "IRIS_HOST": "localhost",
                "IRIS_PORT": "1972",
                "IRIS_NAMESPACE": "USER",
                "IRIS_USERNAME": "demo",
                "IRIS_PASSWORD": "demo",
                "OPENAI_API_KEY": "test-key"
            }
        )

    def test_base_config_template_loads(self, template_engine):
        """Test that base_config.yaml loads correctly."""
        template_dir = Path(__file__).parent.parent.parent.parent / "quick_start" / "config" / "templates"
        context = ConfigurationContext(
            profile="base_config",
            environment="development",
            overrides={},
            template_path=template_dir,
            environment_variables={}
        )
        config = template_engine.resolve_template(context)
        
        # Verify basic structure
        assert "database" in config
        assert "storage" in config
        assert "vector_index" in config
        assert "embeddings" in config
        assert "llm" in config
        assert "logging" in config
        assert "security" in config
        assert "performance" in config

    def test_quick_start_template_inheritance(self, template_engine, base_context):
        """Test that quick_start.yaml properly inherits from base_config.yaml."""
        config = template_engine.resolve_template(base_context)
        
        # Should have base config structure
        assert "database" in config
        assert "storage" in config
        assert "vector_index" in config
        
        # Should have quick start specific additions
        assert "metadata" in config
        assert config["metadata"]["profile"] == "quick_start"
        assert "sample_data" in config
        assert "mcp_server" in config
        
        # Verify environment variable substitution
        assert config["database"]["iris"]["host"] == "localhost"
        assert config["database"]["iris"]["port"] == 1972
        assert config["database"]["iris"]["namespace"] == "USER"

    def test_minimal_profile_template(self, template_engine, base_context):
        """Test quick_start_minimal.yaml profile."""
        context = ConfigurationContext(
            profile="quick_start_minimal",
            environment="development",
            overrides={},
            template_path=base_context.template_path,
            environment_variables=base_context.environment_variables
        )
        
        config = template_engine.resolve_template(context)
        
        # Should inherit from quick_start
        assert "database" in config
        assert "sample_data" in config
        assert "mcp_server" in config
        
        # Should have minimal-specific settings
        assert config["metadata"]["profile"] == "quick_start_minimal"
        assert config["sample_data"]["document_count"] == 10
        assert config["performance"]["max_workers"] == 1
        
        # Should have limited MCP tools
        enabled_tools = config["mcp_server"]["tools"]["enabled"]
        assert "rag_basic" in enabled_tools
        assert "rag_hyde" in enabled_tools
        assert "rag_health_check" in enabled_tools
        assert len(enabled_tools) == 3

    def test_standard_profile_template(self, template_engine, base_context):
        """Test quick_start_standard.yaml profile."""
        context = ConfigurationContext(
            profile="quick_start_standard",
            environment="development",
            overrides={},
            template_path=base_context.template_path,
            environment_variables=base_context.environment_variables
        )
        
        config = template_engine.resolve_template(context)
        
        # Should inherit from quick_start
        assert "database" in config
        assert "sample_data" in config
        assert "mcp_server" in config
        
        # Should have standard-specific settings
        assert config["metadata"]["profile"] == "quick_start_standard"
        assert config["sample_data"]["document_count"] == 100
        assert config["performance"]["max_workers"] == 2
        
        # Should have moderate set of MCP tools
        enabled_tools = config["mcp_server"]["tools"]["enabled"]
        assert "rag_basic" in enabled_tools
        assert "rag_hyde" in enabled_tools
        assert "rag_crag" in enabled_tools
        assert "rag_hybrid_ifind" in enabled_tools
        assert len(enabled_tools) == 6

    def test_extended_profile_template(self, template_engine, base_context):
        """Test quick_start_extended.yaml profile."""
        context = ConfigurationContext(
            profile="quick_start_extended",
            environment="development",
            overrides={},
            template_path=base_context.template_path,
            environment_variables=base_context.environment_variables
        )
        
        config = template_engine.resolve_template(context)
        
        # Should inherit from quick_start
        assert "database" in config
        assert "sample_data" in config
        assert "mcp_server" in config
        
        # Should have extended-specific settings
        assert config["metadata"]["profile"] == "quick_start_extended"
        assert config["sample_data"]["document_count"] == 1000
        assert config["performance"]["max_workers"] == 4
        
        # Should have full set of MCP tools
        enabled_tools = config["mcp_server"]["tools"]["enabled"]
        assert "rag_basic" in enabled_tools
        assert "rag_crag" in enabled_tools
        assert "rag_hyde" in enabled_tools
        assert "rag_graphrag" in enabled_tools
        assert "rag_hybrid_ifind" in enabled_tools
        assert "rag_colbert" in enabled_tools
        assert "rag_noderag" in enabled_tools
        assert "rag_sqlrag" in enabled_tools
        assert len(enabled_tools) == 11
        
        # Should have additional features enabled
        assert config["monitoring"]["enabled"] is True
        assert config["monitoring"]["dashboard"]["enabled"] is True
        assert config["documentation"]["server"]["enabled"] is True

    def test_all_profiles_available(self, template_engine):
        """Test that all expected profiles are available."""
        profiles = template_engine.get_available_profiles()
        
        expected_profiles = {
            "base_config",
            "quick_start", 
            "quick_start_minimal",
            "quick_start_standard", 
            "quick_start_extended"
        }
        
        assert expected_profiles.issubset(set(profiles))

    def test_profile_inheritance_chain(self, template_engine):
        """Test that profile inheritance chains are correct."""
        # Test minimal profile inheritance
        chain = template_engine._build_inheritance_chain("quick_start_minimal")
        assert chain == ["base_config", "quick_start", "quick_start_minimal"]
        
        # Test standard profile inheritance
        chain = template_engine._build_inheritance_chain("quick_start_standard")
        assert chain == ["base_config", "quick_start", "quick_start_standard"]
        
        # Test extended profile inheritance
        chain = template_engine._build_inheritance_chain("quick_start_extended")
        assert chain == ["base_config", "quick_start", "quick_start_extended"]

    def test_environment_variable_defaults(self, template_engine):
        """Test that environment variable defaults work correctly."""
        # Test with minimal environment variables
        template_dir = Path(__file__).parent.parent.parent.parent / "quick_start" / "config" / "templates"
        context = ConfigurationContext(
            profile="quick_start",
            environment="development",
            overrides={},
            template_path=template_dir,
            environment_variables={}  # No environment variables provided
        )
        
        config = template_engine.resolve_template(context)
        
        # Should use defaults from templates
        assert config["database"]["iris"]["host"] == "localhost"
        assert config["database"]["iris"]["port"] == 1972
        assert config["database"]["iris"]["namespace"] == "USER"
        assert config["database"]["iris"]["username"] == "_SYSTEM"
        assert config["database"]["iris"]["password"] == "SYS"

    def test_profile_specific_overrides(self, template_engine, base_context):
        """Test that each profile correctly overrides base settings."""
        profiles_to_test = [
            ("quick_start_minimal", 10, 1),
            ("quick_start_standard", 100, 2), 
            ("quick_start_extended", 1000, 4)
        ]
        
        for profile_name, expected_docs, expected_workers in profiles_to_test:
            context = ConfigurationContext(
                profile=profile_name,
                environment="development",
                overrides={},
                template_path=base_context.template_path,
                environment_variables=base_context.environment_variables
            )
            
            config = template_engine.resolve_template(context)
            
            assert config["sample_data"]["document_count"] == expected_docs
            assert config["performance"]["max_workers"] == expected_workers
            assert config["metadata"]["profile"] == profile_name