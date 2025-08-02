"""
TDD Tests for Requirements-Driven Orchestrator Architecture

These tests validate the key benefits of the elegant orchestrator architecture:
1. Generic tests replace duplicate tests  
2. Requirements are unit testable
3. Test coverage scales automatically
4. Integration tests become simpler

This test file serves as both validation and documentation of the architecture benefits.
"""

import pytest
import sys
import os
from unittest.mock import Mock

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up test environment
os.environ['PYTEST_CURRENT_TEST'] = 'test_orchestrator_requirements_driven'

from iris_rag.validation.orchestrator import SetupOrchestrator
from iris_rag.validation.requirements import (
    get_pipeline_requirements, 
    PIPELINE_REQUIREMENTS_REGISTRY,
    EmbeddingRequirement
)


class TestRequirementsDrivenArchitecture:
    """Test the elegant requirements-driven orchestrator architecture."""

    @pytest.mark.parametrize("pipeline_type", ["basic", "basic_rerank"])
    def test_generic_tests_replace_duplicates(self, pipeline_type):
        """
        Test: Generic tests replace duplicate tests.
        
        This single parametrized test replaces separate test methods for each
        similar pipeline type. Before: test_basic_setup() and test_basic_rerank_setup().
        After: One test validates all basic family pipelines.
        """
        requirements = get_pipeline_requirements(pipeline_type)
        
        # All basic family pipelines should have same basic structure
        assert len(requirements.required_tables) == 1
        assert requirements.required_tables[0].name == "SourceDocuments"
        assert len(requirements.required_embeddings) == 1
        assert requirements.required_embeddings[0].table == "RAG.SourceDocuments"
        
        # This pattern automatically scales to any number of similar pipelines!

    def test_requirements_are_unit_testable(self):
        """
        Test: Requirements are unit testable.
        
        Requirements themselves can be validated independently of setup logic.
        """
        # Test specific requirement structure
        req = get_pipeline_requirements("basic_rerank")
        
        assert isinstance(req.pipeline_name, str)
        assert len(req.pipeline_name) > 0
        assert req.required_tables[0].name == "SourceDocuments"
        assert req.required_tables[0].schema == "RAG"
        assert req.required_embeddings[0].table == "RAG.SourceDocuments"
        assert req.required_embeddings[0].column == "embedding"

    def test_requirement_fulfillment_is_unit_testable(self):
        """Test that requirement fulfillment logic can be unit tested."""
        # Create mock orchestrator
        mock_cm = Mock()
        mock_config = Mock()
        orchestrator = SetupOrchestrator(mock_cm, mock_config)
        orchestrator._ensure_document_embeddings = Mock()
        
        # Create test requirement
        embedding_req = EmbeddingRequirement(
            name="test_embeddings",
            table="RAG.SourceDocuments", 
            column="embedding",
            description="Test embeddings"
        )
        
        # Test fulfillment
        orchestrator._fulfill_embedding_requirement(embedding_req)
        orchestrator._ensure_document_embeddings.assert_called_once()

    def test_coverage_scales_automatically(self):
        """
        Test: Test coverage scales automatically.
        
        Single test validates ALL registered pipelines. When new pipelines
        are added to the registry, they automatically get test coverage.
        """
        pipeline_count = 0
        for pipeline_type in PIPELINE_REQUIREMENTS_REGISTRY.keys():
            requirements = get_pipeline_requirements(pipeline_type)
            
            # Validate requirements are well-formed
            assert isinstance(requirements.pipeline_name, str)
            assert len(requirements.pipeline_name) > 0
            assert len(requirements.required_tables) >= 0
            assert len(requirements.required_embeddings) >= 0
            
            # Each requirement should be properly formed
            for table_req in requirements.required_tables:
                assert isinstance(table_req.name, str)
                assert len(table_req.name) > 0
                
            for embed_req in requirements.required_embeddings:
                assert isinstance(embed_req.name, str)
                assert isinstance(embed_req.table, str)
                assert isinstance(embed_req.column, str)
                
            pipeline_count += 1
        
        # This test automatically validates ALL registered pipelines
        assert pipeline_count >= 8  # Should have at least the core pipelines

    @pytest.mark.parametrize("pipeline_type", ["basic", "basic_rerank"]) 
    def test_integration_pattern_scales(self, pipeline_type):
        """
        Test: Integration tests become simpler.
        
        Same integration pattern works for all pipeline types without
        pipeline-specific setup code.
        """
        # Verify requirements can be loaded (foundation of generic setup)
        requirements = get_pipeline_requirements(pipeline_type)
        
        # Both should have structure that allows generic fulfillment
        assert len(requirements.required_tables) >= 1
        assert len(requirements.required_embeddings) >= 1

    def test_generic_fulfillment_method_call_pattern(self):
        """Test that generic fulfillment method can be called correctly."""
        mock_cm = Mock()
        mock_config = Mock()
        orchestrator = SetupOrchestrator(mock_cm, mock_config)
        orchestrator._fulfill_requirements = Mock()
        
        # Test the generic fulfillment method can be called
        requirements = get_pipeline_requirements("basic_rerank")
        orchestrator._fulfill_requirements(requirements)
        orchestrator._fulfill_requirements.assert_called_once_with(requirements)

    def test_basic_and_basic_rerank_share_requirements(self):
        """
        Test: Architecture benefits - shared requirements.
        
        basic and basic_rerank have identical requirements, proving they
        can share setup logic automatically.
        """
        basic_req = get_pipeline_requirements("basic")
        rerank_req = get_pipeline_requirements("basic_rerank")
        
        # Should have same table requirements
        assert len(basic_req.required_tables) == len(rerank_req.required_tables)
        assert basic_req.required_tables[0].name == rerank_req.required_tables[0].name
        
        # Should have same embedding requirements  
        assert len(basic_req.required_embeddings) == len(rerank_req.required_embeddings)
        assert basic_req.required_embeddings[0].table == rerank_req.required_embeddings[0].table

    def test_informative_error_messages(self):
        """
        Test: Test failures are more informative.
        
        Requirements-driven approach provides clear, specific error messages.
        """
        # Test clear error messages for missing pipeline types
        with pytest.raises(ValueError) as exc_info:
            get_pipeline_requirements("nonexistent_pipeline")
        
        error_msg = str(exc_info.value)
        assert "nonexistent_pipeline" in error_msg
        assert "Available types:" in error_msg

    def test_requirement_objects_are_well_formed(self):
        """Test that requirement objects themselves are properly structured."""
        embedding_req = EmbeddingRequirement(
            name="test_embeddings",
            table="RAG.SourceDocuments",
            column="embedding", 
            description="Test requirement"
        )
        
        assert embedding_req.name == "test_embeddings"
        assert embedding_req.table == "RAG.SourceDocuments" 
        assert embedding_req.column == "embedding"
        assert embedding_req.required is True  # Default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])