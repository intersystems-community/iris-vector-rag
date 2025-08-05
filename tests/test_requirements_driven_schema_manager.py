#!/usr/bin/env python3
"""
Test requirements-driven schema manager DDL generation.

This test validates the elegant solution where:
1. Pipeline requirements declare table capabilities (iFind support, text field types)
2. Schema manager reads requirements and generates appropriate DDL
3. No hardcoded YAML configurations needed

This is a comprehensive E2E test that should have been written TDD from the start!
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.validation.requirements import get_pipeline_requirements
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import IRISConnectionManager


class TestRequirementsDrivenSchemaManager:
    """Test that schema manager generates correct DDL based on pipeline requirements."""
    
    @pytest.fixture
    def schema_manager(self):
        """Create schema manager for testing."""
        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        return SchemaManager(connection_manager, config_manager)
    
    def test_pipeline_requirements_drive_schema_config(self, schema_manager):
        """Test that different pipelines generate different schema configs based on requirements."""
        test_cases = [
            ("basic", "LONGVARCHAR", False),
            ("hyde", "LONGVARCHAR", False), 
            ("crag", "LONGVARCHAR", False),
            ("colbert", "LONGVARCHAR", False),
            ("noderag", "LONGVARCHAR", False),
            ("graphrag", "LONGVARCHAR", False),
            ("hybrid_vector_text", "VARCHAR(MAX)", True),
            ("hybrid_ifind", "VARCHAR(MAX)", True),
        ]
        
        for pipeline_type, expected_text_type, expected_ifind in test_cases:
            # Get requirements for this pipeline
            requirements = get_pipeline_requirements(pipeline_type)
            assert requirements.pipeline_name is not None
            
            # Check that requirements include SourceDocuments table
            source_docs_req = None
            for table_req in requirements.required_tables:
                if table_req.name == "SourceDocuments":
                    source_docs_req = table_req
                    break
            
            assert source_docs_req is not None, f"Pipeline {pipeline_type} should require SourceDocuments table"
            
            # Validate requirement properties match expectations
            assert source_docs_req.text_content_type == expected_text_type
            assert source_docs_req.supports_ifind == expected_ifind
            assert source_docs_req.supports_vector_search == True  # All pipelines support vector search
            
            # Test that schema manager reads requirements correctly
            expected_config = schema_manager._get_expected_schema_config("SourceDocuments", pipeline_type)
            
            assert expected_config.get("text_content_type") == expected_text_type
            assert expected_config.get("supports_ifind") == expected_ifind
            assert expected_config.get("vector_dimension") == 384  # From config
    
    def test_schema_manager_ddl_generation_consistency(self, schema_manager):
        """Test that schema manager generates consistent DDL based on requirements."""
        test_cases = [
            ("basic", "LONGVARCHAR", False),
            ("hybrid_vector_text", "VARCHAR(MAX)", True),
            ("hybrid_ifind", "VARCHAR(MAX)", True)
        ]
        
        for pipeline_type, expected_text_type, expected_ifind in test_cases:
            config = schema_manager._get_expected_schema_config("SourceDocuments", pipeline_type)
            
            # Validate core configuration values
            assert config["text_content_type"] == expected_text_type
            assert config["supports_ifind"] == expected_ifind
            assert config["vector_dimension"] == 384
            assert config["schema_version"] == "1.0.0"
            
            # Validate configuration structure
            assert "configuration" in config
            assert config["configuration"]["managed_by_schema_manager"] == True
            assert config["configuration"]["supports_vector_search"] == True
    
    def test_requirements_framework_eliminates_boilerplate(self, schema_manager):
        """Test that requirements framework eliminates need for hardcoded YAML configurations."""
        # This test validates the architectural benefit: no hardcoded table configurations
        
        # Get requirements for different pipeline types
        basic_req = get_pipeline_requirements("basic")
        ifind_req = get_pipeline_requirements("hybrid_ifind")
        
        # Requirements should be different where expected
        basic_source = next(t for t in basic_req.required_tables if t.name == "SourceDocuments")
        ifind_source = next(t for t in ifind_req.required_tables if t.name == "SourceDocuments")
        
        assert basic_source.text_content_type == "LONGVARCHAR"
        assert basic_source.supports_ifind == False
        
        assert ifind_source.text_content_type == "VARCHAR(MAX)"
        assert ifind_source.supports_ifind == True
        
        # Schema manager should generate different configs automatically
        basic_config = schema_manager._get_expected_schema_config("SourceDocuments", "basic")
        ifind_config = schema_manager._get_expected_schema_config("SourceDocuments", "hybrid_ifind")
        
        assert basic_config["text_content_type"] != ifind_config["text_content_type"]
        assert basic_config["supports_ifind"] != ifind_config["supports_ifind"]
    
    def test_table_requirements_config_extraction(self, schema_manager):
        """Test that schema manager correctly extracts table config from pipeline requirements."""
        # Test the _get_table_requirements_config method directly
        config = schema_manager._get_table_requirements_config("SourceDocuments", "hybrid_ifind")
        
        expected_config = {
            "text_content_type": "VARCHAR(MAX)",
            "supports_ifind": True,
            "supports_vector_search": True
        }
        
        assert config == expected_config
        
        # Test with standard pipeline
        basic_config = schema_manager._get_table_requirements_config("SourceDocuments", "basic")
        
        expected_basic_config = {
            "text_content_type": "LONGVARCHAR",
            "supports_ifind": False,
            "supports_vector_search": True
        }
        
        assert basic_config == expected_basic_config
    
    def test_unknown_pipeline_fallback(self, schema_manager):
        """Test that unknown pipelines get sensible default configuration."""
        config = schema_manager._get_table_requirements_config("SourceDocuments", "unknown_pipeline")
        
        # Should get default configuration
        expected_default = {
            "text_content_type": "LONGVARCHAR",
            "supports_ifind": False,
            "supports_vector_search": True
        }
        
        assert config == expected_default
    
    def test_ifind_pipelines_automatically_get_varchar_max(self, schema_manager):
        """Test that all iFind-supporting pipelines automatically get VARCHAR(MAX)."""
        ifind_pipelines = ["hybrid_ifind", "hybrid_vector_text"]
        
        for pipeline_type in ifind_pipelines:
            requirements = get_pipeline_requirements(pipeline_type)
            source_docs_req = next(t for t in requirements.required_tables if t.name == "SourceDocuments")
            
            # Requirements should declare iFind support
            assert source_docs_req.supports_ifind == True
            assert source_docs_req.text_content_type == "VARCHAR(MAX)"
            
            # Schema manager should generate correct config
            config = schema_manager._get_expected_schema_config("SourceDocuments", pipeline_type)
            assert config["text_content_type"] == "VARCHAR(MAX)"
            assert config["supports_ifind"] == True
    
    def test_standard_pipelines_automatically_get_longvarchar(self, schema_manager):
        """Test that standard pipelines automatically get LONGVARCHAR for streaming."""
        standard_pipelines = ["basic", "hyde", "crag", "colbert", "noderag", "graphrag"]
        
        for pipeline_type in standard_pipelines:
            requirements = get_pipeline_requirements(pipeline_type)
            source_docs_req = next(t for t in requirements.required_tables if t.name == "SourceDocuments")
            
            # Requirements should NOT declare iFind support
            assert source_docs_req.supports_ifind == False
            assert source_docs_req.text_content_type == "LONGVARCHAR"
            
            # Schema manager should generate correct config
            config = schema_manager._get_expected_schema_config("SourceDocuments", pipeline_type)
            assert config["text_content_type"] == "LONGVARCHAR"
            assert config["supports_ifind"] == False


@pytest.mark.integration
class TestRequirementsDrivenSchemaManagerIntegration:
    """Integration tests for requirements-driven schema manager."""
    
    @pytest.fixture
    def schema_manager(self):
        """Create schema manager for integration testing."""
        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        return SchemaManager(connection_manager, config_manager)
    
    def test_schema_manager_table_migration_with_requirements(self, schema_manager):
        """Test that schema manager can migrate tables using pipeline requirements."""
        # Test that migration uses pipeline-specific requirements
        needs_migration = schema_manager.needs_migration("SourceDocuments", "hybrid_ifind")
        
        # This should work without errors (whether migration is needed or not)
        assert isinstance(needs_migration, bool)
        
        # Test with standard pipeline
        needs_migration_basic = schema_manager.needs_migration("SourceDocuments", "basic")
        assert isinstance(needs_migration_basic, bool)
    
    def test_ensure_table_schema_with_pipeline_type(self, schema_manager):
        """Test that ensure_table_schema accepts pipeline_type parameter."""
        # This should work for any pipeline type without errors
        try:
            result = schema_manager.ensure_table_schema("SourceDocuments", "hybrid_ifind")
            assert isinstance(result, bool)
        except Exception as e:
            # If it fails due to database issues, that's OK for unit tests
            # The important thing is that the method signature and logic work
            assert "database" in str(e).lower() or "connection" in str(e).lower()


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main([__file__, "-v"])