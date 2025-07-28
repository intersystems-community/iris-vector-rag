"""
Test suite for Phase 2 of the Library Consumption Framework: Standard API Layer.

This module implements the TDD anchor tests for the advanced Standard API
that provides configurable pipelines, technique selection, and dependency injection.

Following TDD workflow:
1. RED: Write failing tests first
2. GREEN: Implement minimum code to pass tests  
3. REFACTOR: Clean up code while keeping tests passing
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class TestStandardAPIPhase2:
    """Test suite for Standard API Phase 2 implementation."""
    
    def test_configurable_rag_initialization(self):
        """
        TDD Anchor Test: ConfigurableRAG works with config.
        
        This test verifies that the Standard API can be initialized
        with advanced configuration options.
        """
        # This test should fail initially (RED phase)
        from rag_templates.standard import ConfigurableRAG
        
        config = {
            "technique": "basic",
            "llm_provider": "anthropic",
            "embedding_model": "text-embedding-3-large",
            "max_results": 10
        }
        
        # Should initialize with configuration
        rag = ConfigurableRAG(config)
        
        # Should have configuration loaded
        assert rag is not None
        assert hasattr(rag, '_config')
        assert hasattr(rag, '_technique')
        assert rag._technique == "basic"
    
    def test_technique_selection(self):
        """
        TDD Anchor Test: Different techniques can be selected.
        
        This test verifies that the Standard API can select
        different RAG techniques dynamically.
        """
        from rag_templates.standard import ConfigurableRAG
        
        # Test basic technique
        basic_config = {"technique": "basic"}
        basic_rag = ConfigurableRAG(basic_config)
        assert basic_rag._technique == "basic"
        
        # Test colbert technique
        colbert_config = {"technique": "colbert"}
        colbert_rag = ConfigurableRAG(colbert_config)
        assert colbert_rag._technique == "colbert"
        
        # Test hyde technique
        hyde_config = {"technique": "hyde"}
        hyde_rag = ConfigurableRAG(hyde_config)
        assert hyde_rag._technique == "hyde"
    
    def test_pipeline_factory_creation(self):
        """
        TDD Anchor Test: PipelineFactory creates pipelines dynamically.
        
        This test verifies that the PipelineFactory can create
        different pipeline instances based on technique selection.
        """
        from rag_templates.core.pipeline_factory import PipelineFactory
        from rag_templates.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        factory = PipelineFactory(config_manager)
        
        # Should be able to create basic pipeline
        basic_pipeline = factory.create_pipeline("basic")
        assert basic_pipeline is not None
        assert hasattr(basic_pipeline, 'execute')
        
        # Should be able to create colbert pipeline
        colbert_pipeline = factory.create_pipeline("colbert")
        assert colbert_pipeline is not None
        assert hasattr(colbert_pipeline, 'execute')
    
    def test_dependency_injection(self):
        """
        TDD Anchor Test: Components are injected automatically.
        
        This test verifies that the PipelineFactory automatically
        injects required dependencies (LLM, embeddings, vector store).
        """
        from rag_templates.core.pipeline_factory import PipelineFactory
        from rag_templates.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        factory = PipelineFactory(config_manager)
        
        # Mock the dependencies
        with patch('common.utils.get_embedding_func') as mock_embedding:
            with patch('common.utils.get_llm_func') as mock_llm:
                mock_embedding.return_value = MagicMock()
                mock_llm.return_value = MagicMock()
                
                pipeline = factory.create_pipeline("basic")
                
                # Dependencies should have been injected
                assert hasattr(pipeline, 'llm_func')
                assert hasattr(pipeline, 'embedding_manager')
    
    def test_technique_registry(self):
        """
        TDD Anchor Test: Techniques can be registered and discovered.
        
        This test verifies that the TechniqueRegistry can register
        and discover available RAG techniques.
        """
        from rag_templates.core.technique_registry import TechniqueRegistry
        
        registry = TechniqueRegistry()
        
        # Should have built-in techniques registered
        techniques = registry.list_techniques()
        assert isinstance(techniques, list)
        assert len(techniques) > 0
        
        # Should include basic techniques
        technique_names = [t['name'] for t in techniques]
        assert "basic" in technique_names
        assert "colbert" in technique_names
        assert "hyde" in technique_names
    
    def test_advanced_configuration(self):
        """
        TDD Anchor Test: Complex configurations are supported.
        
        This test verifies that the Standard API supports
        complex, technique-specific configurations.
        """
        from rag_templates.standard import ConfigurableRAG
        
        complex_config = {
            "technique": "colbert",
            "llm_provider": "anthropic",
            "llm_config": {
                "model": "claude-3-sonnet",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            "embedding_model": "text-embedding-3-large",
            "embedding_config": {
                "dimension": 3072,
                "batch_size": 16
            },
            "technique_config": {
                "max_query_length": 512,
                "doc_maxlen": 180,
                "top_k": 15
            },
            "vector_index": {
                "type": "HNSW",
                "M": 32,
                "efConstruction": 400
            }
        }
        
        rag = ConfigurableRAG(complex_config)
        
        # Should handle complex configuration
        assert rag._config["technique"] == "colbert"
        assert rag._config["llm_config"]["temperature"] == 0.1
        assert rag._config["technique_config"]["top_k"] == 15
    
    def test_backward_compatibility(self):
        """
        TDD Anchor Test: Standard API doesn't break Simple API.
        
        This test verifies that the Standard API maintains
        backward compatibility with the Simple API.
        """
        # Simple API should still work
        from rag_templates.simple import RAG
        
        simple_rag = RAG()
        assert simple_rag is not None
        
        # Standard API should work alongside Simple API
        from rag_templates.standard import ConfigurableRAG
        
        standard_rag = ConfigurableRAG({"technique": "basic"})
        assert standard_rag is not None
        
        # Both should be independent
        assert type(simple_rag) != type(standard_rag)
    
    def test_advanced_query_options(self):
        """
        Test that the Standard API supports advanced query options.
        """
        from rag_templates.standard import ConfigurableRAG
        
        rag = ConfigurableRAG({"technique": "basic"})
        
        # Mock the underlying pipeline
        with patch.object(rag, '_get_pipeline') as mock_pipeline:
            mock_pipeline.return_value.execute.return_value = {
                "answer": "Test answer",
                "query": "test query",
                "retrieved_documents": [{"content": "doc1"}],
                "sources": ["source1"],
                "metadata": {"similarity_scores": [0.9]}
            }
            
            # Should support advanced query options
            result = rag.query("What is machine learning?", {
                "include_sources": True,
                "min_similarity": 0.8,
                "source_filter": "academic_papers",
                "max_results": 15
            })
            
            # Should return enhanced result
            assert isinstance(result, dict)
            assert "answer" in result
            assert "sources" in result
            assert "metadata" in result


class TestPipelineFactoryPhase2:
    """Test suite for PipelineFactory Phase 2 implementation."""
    
    def test_dynamic_pipeline_loading(self):
        """
        Test that PipelineFactory can load pipelines dynamically.
        """
        from rag_templates.core.pipeline_factory import PipelineFactory
        from rag_templates.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        factory = PipelineFactory(config_manager)
        
        # Should be able to load different pipeline types
        pipeline_types = ["basic", "colbert", "hyde", "crag"]
        
        for pipeline_type in pipeline_types:
            try:
                pipeline = factory.create_pipeline(pipeline_type)
                assert pipeline is not None
            except ImportError:
                # Some pipelines might not be available in test environment
                pytest.skip(f"Pipeline {pipeline_type} not available")
    
    def test_component_lifecycle_management(self):
        """
        Test that PipelineFactory manages component lifecycle properly.
        """
        from rag_templates.core.pipeline_factory import PipelineFactory
        from rag_templates.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        factory = PipelineFactory(config_manager)
        
        # Should manage component creation and cleanup
        with patch('common.iris_connection_manager.get_iris_connection') as mock_conn:
            mock_conn.return_value = MagicMock()
            
            pipeline = factory.create_pipeline("basic")
            
            # Should have created connection manager
            mock_conn.assert_called_once()
    
    def test_configuration_validation(self):
        """
        Test that PipelineFactory validates configuration before creation.
        """
        from rag_templates.core.pipeline_factory import PipelineFactory
        from rag_templates.core.config_manager import ConfigurationManager
        from rag_templates.core.errors import ConfigurationError
        
        config_manager = ConfigurationManager()
        factory = PipelineFactory(config_manager)
        
        # Should validate technique exists
        with pytest.raises((ConfigurationError, ValueError)):
            factory.create_pipeline("nonexistent_technique")


class TestTechniqueRegistryPhase2:
    """Test suite for TechniqueRegistry Phase 2 implementation."""
    
    def test_technique_metadata_management(self):
        """
        Test that TechniqueRegistry manages technique metadata.
        """
        from rag_templates.core.technique_registry import TechniqueRegistry
        
        registry = TechniqueRegistry()
        
        # Should provide technique metadata
        basic_info = registry.get_technique_info("basic")
        assert basic_info is not None
        assert "name" in basic_info
        assert "module" in basic_info
        assert "class" in basic_info
    
    def test_technique_requirements_validation(self):
        """
        Test that TechniqueRegistry validates technique requirements.
        """
        from rag_templates.core.technique_registry import TechniqueRegistry
        
        registry = TechniqueRegistry()
        
        # Should validate technique dependencies
        is_available = registry.is_technique_available("basic")
        assert isinstance(is_available, bool)
    
    def test_custom_technique_registration(self):
        """
        Test that TechniqueRegistry supports custom technique plugins.
        """
        from rag_templates.core.technique_registry import TechniqueRegistry
        
        registry = TechniqueRegistry()
        
        # Should be able to register custom technique
        custom_technique = {
            "name": "custom_rag",
            "module": "custom_package.rag",
            "class": "CustomRAGPipeline",
            "enabled": True,
            "params": {"custom_param": "value"}
        }
        
        registry.register_technique(custom_technique)
        
        # Should be in the registry
        techniques = registry.list_techniques()
        custom_names = [t['name'] for t in techniques if t['name'] == 'custom_rag']
        assert len(custom_names) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])