"""
Test NodeRAG integration with the validation system.

This test verifies that NodeRAG has been properly added to the iris_rag
validation system and can be created through the validated factory.
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock

from iris_rag.validation.requirements import get_pipeline_requirements, NodeRAGRequirements
from iris_rag.validation.factory import ValidatedPipelineFactory
from iris_rag.validation.orchestrator import SetupOrchestrator
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.noderag import NodeRAGPipeline

logger = logging.getLogger(__name__)


class TestNodeRAGValidationIntegration:
    """Test NodeRAG integration with the validation system."""
    
    def test_noderag_requirements_exist(self):
        """Test that NodeRAG requirements can be retrieved."""
        requirements = get_pipeline_requirements("noderag")
        
        assert isinstance(requirements, NodeRAGRequirements)
        assert requirements.pipeline_name == "noderag"
        
        # Check required tables
        required_tables = requirements.required_tables
        assert len(required_tables) == 1
        assert required_tables[0].name == "SourceDocuments"
        assert required_tables[0].schema == "RAG"
        
        # Check required embeddings
        required_embeddings = requirements.required_embeddings
        assert len(required_embeddings) == 1
        assert required_embeddings[0].name == "document_embeddings"
        assert required_embeddings[0].table == "RAG.SourceDocuments"
        assert required_embeddings[0].column == "embedding"
        
        # Check optional tables
        optional_tables = requirements.optional_tables
        table_names = [table.name for table in optional_tables]
        assert "KnowledgeGraphNodes" in table_names
        assert "KnowledgeGraphEdges" in table_names
        assert "DocumentChunks" in table_names
    
    def test_noderag_factory_creation(self):
        """Test that NodeRAG pipeline can be created through the factory."""
        # Mock dependencies
        mock_connection_manager = Mock(spec=ConnectionManager)
        mock_config_manager = Mock(spec=ConfigurationManager)
        
        # Mock the validation to pass
        factory = ValidatedPipelineFactory(mock_connection_manager, mock_config_manager)
        
        # Mock the validator to return a successful validation
        mock_validation_report = Mock()
        mock_validation_report.overall_valid = True
        factory.validator.validate_pipeline_requirements = Mock(return_value=mock_validation_report)
        
        # Create NodeRAG pipeline
        pipeline = factory.create_pipeline(
            pipeline_type="noderag",
            validate_requirements=True,
            auto_setup=False
        )
        
        assert isinstance(pipeline, NodeRAGPipeline)
        assert pipeline.connection_manager == mock_connection_manager
        assert pipeline.config_manager == mock_config_manager
    
    def test_noderag_in_available_pipelines(self):
        """Test that NodeRAG appears in the list of available pipelines."""
        # Mock dependencies
        mock_connection_manager = Mock(spec=ConnectionManager)
        mock_config_manager = Mock(spec=ConfigurationManager)
        
        factory = ValidatedPipelineFactory(mock_connection_manager, mock_config_manager)
        
        # Mock validation to avoid database calls
        factory.validate_pipeline_type = Mock(return_value={
            "pipeline_type": "noderag",
            "valid": True,
            "summary": "All requirements met",
            "table_issues": [],
            "embedding_issues": [],
            "suggestions": []
        })
        
        available_pipelines = factory.list_available_pipelines()
        
        assert "noderag" in available_pipelines
        assert available_pipelines["noderag"]["pipeline_type"] == "noderag"
    
    def test_noderag_orchestrator_setup(self):
        """Test that the orchestrator can handle NodeRAG setup."""
        # Mock dependencies
        mock_connection_manager = Mock(spec=ConnectionManager)
        mock_config_manager = Mock(spec=ConfigurationManager)
        
        orchestrator = SetupOrchestrator(mock_connection_manager, mock_config_manager)
        
        # Mock the database operations
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connection_manager.get_connection.return_value = mock_connection
        
        # Mock table existence checks (tables don't exist)
        mock_cursor.fetchone.side_effect = [
            [0],  # KnowledgeGraphNodes doesn't exist
            [0],  # KnowledgeGraphEdges doesn't exist
            [0],  # Documents without embeddings count
        ]
        
        # Mock embedding manager
        mock_embedding_manager = Mock()
        orchestrator.embedding_manager = mock_embedding_manager
        
        # Mock validator
        mock_validation_report = Mock()
        mock_validation_report.overall_valid = True
        orchestrator.validator.validate_pipeline_requirements = Mock(return_value=mock_validation_report)
        
        # Test setup
        result = orchestrator.setup_pipeline("noderag", auto_fix=True)
        
        assert result.overall_valid
        
        # Verify that setup methods were called
        mock_connection_manager.get_connection.assert_called()
        mock_cursor.execute.assert_called()  # Should have executed table creation SQL
    
    def test_noderag_requirements_validation_structure(self):
        """Test the structure of NodeRAG requirements validation."""
        requirements = get_pipeline_requirements("noderag")
        all_requirements = requirements.get_all_requirements()
        
        assert "pipeline_name" in all_requirements
        assert "required_tables" in all_requirements
        assert "required_embeddings" in all_requirements
        assert "optional_tables" in all_requirements
        assert "optional_embeddings" in all_requirements
        
        assert all_requirements["pipeline_name"] == "noderag"
        
        # Verify required components
        assert len(all_requirements["required_tables"]) == 1
        assert len(all_requirements["required_embeddings"]) == 1
        
        # Verify optional components for enhanced functionality
        assert len(all_requirements["optional_tables"]) == 3  # KG nodes, edges, chunks
        assert len(all_requirements["optional_embeddings"]) == 2  # node and chunk embeddings


if __name__ == "__main__":
    # Run basic validation test
    test_instance = TestNodeRAGValidationIntegration()
    
    print("Testing NodeRAG validation integration...")
    
    try:
        test_instance.test_noderag_requirements_exist()
        print("✓ NodeRAG requirements test passed")
        
        test_instance.test_noderag_factory_creation()
        print("✓ NodeRAG factory creation test passed")
        
        test_instance.test_noderag_in_available_pipelines()
        print("✓ NodeRAG available pipelines test passed")
        
        test_instance.test_noderag_orchestrator_setup()
        print("✓ NodeRAG orchestrator setup test passed")
        
        test_instance.test_noderag_requirements_validation_structure()
        print("✓ NodeRAG requirements structure test passed")
        
        print("\n=== ALL NODERAG VALIDATION INTEGRATION TESTS PASSED ===")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()