"""
Tests for the pre-condition validation system.

This module tests the validation architecture that ensures pipelines
have all required data before execution.
"""

import pytest
from unittest.mock import Mock, patch
from iris_rag.validation.requirements import (
    BasicRAGRequirements, ColBERTRequirements, get_pipeline_requirements
)
from iris_rag.validation.validator import PreConditionValidator
from iris_rag.validation.orchestrator import SetupOrchestrator
from iris_rag.validation.factory import ValidatedPipelineFactory, PipelineValidationError


class TestPipelineRequirements:
    """Test pipeline requirements definitions."""
    
    def test_basic_rag_requirements(self):
        """Test BasicRAGRequirements definition."""
        requirements = BasicRAGRequirements()
        
        assert requirements.pipeline_name == "basic_rag"
        assert len(requirements.required_tables) == 1
        assert requirements.required_tables[0].name == "SourceDocuments"
        assert requirements.required_tables[0].schema == "RAG"
        
        assert len(requirements.required_embeddings) == 1
        assert requirements.required_embeddings[0].name == "document_embeddings"
        assert requirements.required_embeddings[0].table == "RAG.SourceDocuments"
        assert requirements.required_embeddings[0].column == "embedding"
    
    def test_colbert_requirements(self):
        """Test ColBERTRequirements definition."""
        requirements = ColBERTRequirements()
        
        assert requirements.pipeline_name == "colbert_rag"
        assert len(requirements.required_tables) == 2
        
        table_names = [table.name for table in requirements.required_tables]
        assert "SourceDocuments" in table_names
        assert "DocumentTokenEmbeddings" in table_names
        
        assert len(requirements.required_embeddings) == 2
        embedding_names = [emb.name for emb in requirements.required_embeddings]
        assert "document_embeddings" in embedding_names
        assert "token_embeddings" in embedding_names
    
    def test_get_pipeline_requirements(self):
        """Test pipeline requirements registry."""
        # Test valid pipeline types
        basic_req = get_pipeline_requirements("basic")
        assert isinstance(basic_req, BasicRAGRequirements)
        
        colbert_req = get_pipeline_requirements("colbert")
        assert isinstance(colbert_req, ColBERTRequirements)
        
        # Test invalid pipeline type
        with pytest.raises(ValueError, match="Unknown pipeline type"):
            get_pipeline_requirements("invalid_type")


class TestPreConditionValidator:
    """Test pre-condition validator."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager."""
        manager = Mock()
        connection = Mock()
        cursor = Mock()
        
        manager.get_connection.return_value = connection
        connection.cursor.return_value = cursor
        
        return manager, connection, cursor
    
    @pytest.fixture
    def validator(self, mock_connection_manager):
        """Create validator with mocked connection."""
        manager, _, _ = mock_connection_manager
        return PreConditionValidator(manager)
    
    def test_validate_table_requirement_success(self, validator, mock_connection_manager):
        """Test successful table validation."""
        _, _, cursor = mock_connection_manager
        cursor.fetchone.return_value = [100]  # 100 rows
        
        requirements = BasicRAGRequirements()
        table_req = requirements.required_tables[0]
        
        result = validator._validate_table_requirement(table_req)
        
        assert result.is_valid
        assert "100 rows" in result.message
        assert result.details["row_count"] == 100
    
    def test_validate_table_requirement_insufficient_rows(self, validator, mock_connection_manager):
        """Test table validation with insufficient rows."""
        _, _, cursor = mock_connection_manager
        cursor.fetchone.return_value = [0]  # 0 rows
        
        requirements = BasicRAGRequirements()
        table_req = requirements.required_tables[0]
        table_req.min_rows = 1  # Require at least 1 row
        
        result = validator._validate_table_requirement(table_req)
        
        assert not result.is_valid
        assert "0 rows" in result.message
        assert "requires at least 1" in result.message
    
    def test_validate_table_requirement_table_not_exists(self, validator, mock_connection_manager):
        """Test table validation when table doesn't exist."""
        _, _, cursor = mock_connection_manager
        cursor.execute.side_effect = Exception("Table does not exist")
        
        requirements = BasicRAGRequirements()
        table_req = requirements.required_tables[0]
        
        result = validator._validate_table_requirement(table_req)
        
        assert not result.is_valid
        assert "does not exist" in result.message
    
    def test_validate_embedding_requirement_success(self, validator, mock_connection_manager):
        """Test successful embedding validation."""
        _, _, cursor = mock_connection_manager
        realistic_vector = "0.1,0.2,0.3,0.4,0.5"  # IRIS VECTOR format without brackets
        cursor.fetchone.side_effect = [
            [100, 100],  # total_rows, rows_with_embeddings
            [realistic_vector]  # sample embedding in IRIS VECTOR format
        ]
        
        requirements = BasicRAGRequirements()
        embedding_req = requirements.required_embeddings[0]
        
        result = validator._validate_embedding_requirement(embedding_req)
        
        assert result.is_valid
        assert "100/100" in result.message
        assert result.details["completeness_ratio"] == 1.0
    
    def test_validate_embedding_requirement_incomplete(self, validator, mock_connection_manager):
        """Test embedding validation with incomplete embeddings."""
        _, _, cursor = mock_connection_manager
        # Create a realistic IRIS VECTOR format
        realistic_vector = "0.1,0.2,0.3,0.4,0.5"  # IRIS VECTOR format without brackets
        cursor.fetchone.side_effect = [
            [100, 50],  # total_rows, rows_with_embeddings (50% complete)
            [realistic_vector]  # sample embedding in IRIS VECTOR format
        ]
        
        requirements = BasicRAGRequirements()
        embedding_req = requirements.required_embeddings[0]
        
        result = validator._validate_embedding_requirement(embedding_req)
        
        assert not result.is_valid
        assert "50/100" in result.message
        assert result.details["completeness_ratio"] == 0.5
    
    def test_validate_embedding_requirement_no_embeddings(self, validator, mock_connection_manager):
        """Test embedding validation with no embeddings."""
        _, _, cursor = mock_connection_manager
        cursor.fetchone.return_value = [100, 0]  # total_rows, rows_with_embeddings
        
        requirements = BasicRAGRequirements()
        embedding_req = requirements.required_embeddings[0]
        
        result = validator._validate_embedding_requirement(embedding_req)
        
        assert not result.is_valid
        assert "No embeddings found" in result.message
    
    def test_validate_pipeline_requirements_all_valid(self, validator, mock_connection_manager):
        """Test complete pipeline validation when all requirements are met."""
        _, _, cursor = mock_connection_manager
        realistic_vector = "0.1,0.2,0.3,0.4,0.5"  # IRIS VECTOR format without brackets
        cursor.fetchone.side_effect = [
            [100],  # table row count for required SourceDocuments
            [100],  # table row count for optional DocumentChunks
            [100, 100],  # embedding completeness for required SourceDocuments
            [realistic_vector],  # sample embedding for required SourceDocuments
            [100, 100],  # embedding completeness for optional DocumentChunks
            [realistic_vector]   # sample embedding for optional DocumentChunks
        ]
        
        requirements = BasicRAGRequirements()
        report = validator.validate_pipeline_requirements(requirements)
    
        assert report.overall_valid
    
    def test_validate_pipeline_requirements_with_issues(self, validator, mock_connection_manager):
        """Test complete pipeline validation with issues."""
        _, _, cursor = mock_connection_manager
        cursor.fetchone.side_effect = [
            [0],  # table row count (insufficient)
            [0, 0],  # embedding completeness (no embeddings)
        ]
        
        requirements = BasicRAGRequirements()
        report = validator.validate_pipeline_requirements(requirements)
        
        assert not report.overall_valid
        assert "Pipeline not ready" in report.summary
        assert len(report.setup_suggestions) > 0
    
    def test_quick_validate(self, validator, mock_connection_manager):
        """Test quick validation method."""
        _, _, cursor = mock_connection_manager
        cursor.fetchone.side_effect = [
            [100],  # table row count for required SourceDocuments
            [100],  # table row count for optional DocumentChunks
            [100, 100],  # embedding completeness for required SourceDocuments
            ["0.1,0.2,0.3,0.4,0.5"],  # sample embedding for required SourceDocuments
            [100, 100],  # embedding completeness for optional DocumentChunks
            ["0.1,0.2,0.3,0.4,0.5"]   # sample embedding for optional DocumentChunks
        ]
    
        result = validator.quick_validate("basic")
        assert result is True
        
        # Test with issues
        cursor.fetchone.side_effect = [
            [0],  # table row count (insufficient)
            [0],  # table row count for optional table
            [0, 0],  # embedding completeness (no embeddings)
        ]
        
        result = validator.quick_validate("basic")
        assert result is False


class TestSetupOrchestrator:
    """Test setup orchestrator."""
    
    @pytest.fixture
    def mock_managers(self):
        """Mock connection and config managers."""
        connection_manager = Mock()
        config_manager = Mock()
        
        connection = Mock()
        cursor = Mock()
        connection_manager.get_connection.return_value = connection
        connection.cursor.return_value = cursor
        
        return connection_manager, config_manager, connection, cursor
    
    @pytest.fixture
    def orchestrator(self, mock_managers):
        """Create orchestrator with mocked dependencies."""
        connection_manager, config_manager, _, _ = mock_managers
        
        with patch('iris_rag.validation.orchestrator.EmbeddingManager') as mock_embedding_manager:
            with patch('iris_rag.validation.orchestrator.PreConditionValidator') as mock_validator:
                orchestrator = SetupOrchestrator(config_manager, connection_manager)
                orchestrator.embedding_manager = mock_embedding_manager.return_value
                orchestrator.validator = mock_validator.return_value
                return orchestrator
    
    def test_ensure_document_embeddings_all_present(self, orchestrator, mock_managers):
        """Test ensuring document embeddings when all are present."""
        connection_manager, _, _, cursor = mock_managers
        cursor.fetchone.return_value = [0]  # No missing embeddings
        
        with patch('common.iris_connection_manager.get_iris_connection') as mock_get_connection:
            mock_get_connection.return_value = connection_manager.get_connection.return_value
            orchestrator._ensure_document_embeddings()
        
        # Should not call embedding generation
        orchestrator.embedding_manager.embed_texts.assert_not_called()
    
    def test_ensure_document_embeddings_some_missing(self, orchestrator, mock_managers):
        """Test ensuring document embeddings when some are missing."""
        connection_manager, _, _, cursor = mock_managers
        cursor.fetchone.return_value = [5]  # 5 missing embeddings
        cursor.fetchall.return_value = [
            ("doc1", "content1"),
            ("doc2", "content2")
        ]
        
        # Mock embedding generation
        orchestrator.embedding_manager.embed_texts.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        with patch('common.iris_connection_manager.get_iris_connection') as mock_get_connection:
            mock_get_connection.return_value = connection_manager.get_connection.return_value
            orchestrator._ensure_document_embeddings()
        
        # Should call embedding generation
        orchestrator.embedding_manager.embed_texts.assert_called_once()
        
        # Should update database
        assert cursor.execute.call_count >= 2  # At least 2 UPDATE calls
    
    def test_setup_basic_pipeline(self, orchestrator, mock_managers):
        """Test setting up basic pipeline."""
        connection_manager, _, _, cursor = mock_managers
        cursor.fetchone.return_value = [0]  # No missing embeddings
        
        with patch('common.iris_connection_manager.get_iris_connection') as mock_get_connection:
            mock_get_connection.return_value = connection_manager.get_connection.return_value
            
            requirements = BasicRAGRequirements()
            orchestrator._setup_basic_pipeline(requirements)
            
            # Should check for document embeddings
            cursor.execute.assert_called()
    
    def test_setup_colbert_pipeline(self, orchestrator, mock_managers):
        """Test setting up ColBERT pipeline."""
        _, _, _, cursor = mock_managers
        cursor.fetchone.side_effect = [
            [0],  # No missing document embeddings
            [0],  # Token embeddings table doesn't exist
            [0],  # No existing token embeddings
        ]
        cursor.fetchall.return_value = [("doc1", "content1")]
        
        # Mock embedding generation
        orchestrator.embedding_manager.embed_text.return_value = [0.1, 0.2, 0.3]
        
        requirements = ColBERTRequirements()
        orchestrator._setup_colbert_pipeline(requirements)
        
        # Should create token embeddings table and generate embeddings
        assert cursor.execute.call_count >= 3


class TestValidatedPipelineFactory:
    """Test validated pipeline factory."""
    
    @pytest.fixture
    def mock_managers(self):
        """Mock connection and config managers."""
        connection_manager = Mock()
        config_manager = Mock()
        return connection_manager, config_manager
    
    @pytest.fixture
    def factory(self, mock_managers):
        """Create factory with mocked dependencies."""
        connection_manager, config_manager = mock_managers
        
        with patch('iris_rag.validation.factory.PreConditionValidator') as mock_validator:
            with patch('iris_rag.validation.factory.SetupOrchestrator') as mock_orchestrator:
                factory = ValidatedPipelineFactory(config_manager)
                factory.validator = mock_validator.return_value
                factory.orchestrator = mock_orchestrator.return_value
                return factory
    
    def test_create_pipeline_validation_success(self, factory):
        """Test creating pipeline when validation succeeds."""
        # Mock successful validation
        mock_report = Mock()
        mock_report.overall_valid = True
        factory.validator.validate_pipeline_requirements.return_value = mock_report
        
        with patch.object(factory, '_create_pipeline_instance') as mock_create:
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            result = factory.create_pipeline("basic", validate_requirements=True)
            
            assert result == mock_pipeline
            factory.validator.validate_pipeline_requirements.assert_called_once()
            mock_create.assert_called_once()
    
    def test_create_pipeline_validation_failure_no_auto_setup(self, factory):
        """Test creating pipeline when validation fails and auto_setup is False."""
        # Mock failed validation
        mock_report = Mock()
        mock_report.overall_valid = False
        mock_report.summary = "Missing embeddings"
        mock_report.setup_suggestions = ["Generate embeddings"]
        factory.validator.validate_pipeline_requirements.return_value = mock_report
        
        with pytest.raises(PipelineValidationError, match="Missing embeddings"):
            factory.create_pipeline("basic", validate_requirements=True, auto_setup=False)
    
    def test_create_pipeline_validation_failure_with_auto_setup(self, factory):
        """Test creating pipeline when validation fails but auto_setup is True."""
        # Mock failed initial validation, successful after setup
        mock_report_failed = Mock()
        mock_report_failed.overall_valid = False
        mock_report_failed.summary = "Missing embeddings"
        mock_report_failed.setup_suggestions = ["Generate embeddings"]
        
        mock_report_success = Mock()
        mock_report_success.overall_valid = True
        
        factory.validator.validate_pipeline_requirements.side_effect = [mock_report_failed]
        factory.orchestrator.setup_pipeline.return_value = mock_report_success
        
        with patch.object(factory, '_create_pipeline_instance') as mock_create:
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            result = factory.create_pipeline("basic", validate_requirements=True, auto_setup=True)
            
            assert result == mock_pipeline
            factory.orchestrator.setup_pipeline.assert_called_once_with("basic", auto_fix=True)
    
    def test_create_pipeline_no_validation(self, factory):
        """Test creating pipeline without validation."""
        with patch.object(factory, '_create_pipeline_instance') as mock_create:
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            result = factory.create_pipeline("basic", validate_requirements=False)
            
            assert result == mock_pipeline
            factory.validator.validate_pipeline_requirements.assert_not_called()
            mock_create.assert_called_once()
    
    def test_validate_pipeline_type(self, factory):
        """Test validating pipeline type."""
        # Mock validation report
        mock_report = Mock()
        mock_report.overall_valid = True
        mock_report.summary = "All good"
        mock_report.table_validations = {"SourceDocuments": Mock(is_valid=True)}
        mock_report.embedding_validations = {"document_embeddings": Mock(is_valid=True)}
        mock_report.setup_suggestions = []
        
        factory.validator.validate_pipeline_requirements.return_value = mock_report
        
        result = factory.validate_pipeline_type("basic")
        
        assert result["pipeline_type"] == "basic"
        assert result["valid"] is True
        assert result["summary"] == "All good"
        assert len(result["table_issues"]) == 0
        assert len(result["embedding_issues"]) == 0
    
    def test_get_pipeline_status(self, factory):
        """Test getting pipeline status."""
        # Mock validation report
        mock_table_result = Mock()
        mock_table_result.is_valid = True
        mock_table_result.message = "Table OK"
        mock_table_result.details = {"row_count": 100}
        
        mock_embedding_result = Mock()
        mock_embedding_result.is_valid = True
        mock_embedding_result.message = "Embeddings OK"
        mock_embedding_result.details = {"completeness_ratio": 1.0}
        
        mock_report = Mock()
        mock_report.overall_valid = True
        mock_report.summary = "All requirements satisfied"
        mock_report.table_validations = {"SourceDocuments": mock_table_result}
        mock_report.embedding_validations = {"document_embeddings": mock_embedding_result}
        mock_report.setup_suggestions = []
        
        factory.validator.validate_pipeline_requirements.return_value = mock_report
        
        result = factory.get_pipeline_status("basic")
        
        assert result["pipeline_type"] == "basic"
        assert result["overall_valid"] is True
        assert "SourceDocuments" in result["tables"]
        assert "document_embeddings" in result["embeddings"]
        assert "requirements" in result
    
    def test_setup_pipeline_requirements(self, factory):
        """Test setting up pipeline requirements."""
        # Mock successful setup
        mock_report = Mock()
        mock_report.overall_valid = True
        mock_report.summary = "Setup completed"
        mock_report.table_validations = {}
        mock_report.embedding_validations = {}
        
        factory.orchestrator.setup_pipeline.return_value = mock_report
        
        result = factory.setup_pipeline_requirements("basic")
        
        assert result["pipeline_type"] == "basic"
        assert result["success"] is True
        assert result["setup_completed"] is True
        assert len(result["remaining_issues"]) == 0
    
    def test_list_available_pipelines(self, factory):
        """Test listing available pipelines."""
        # Mock validation for each pipeline type
        def mock_validate(pipeline_type):
            return {
                "pipeline_type": pipeline_type,
                "valid": True,
                "summary": f"{pipeline_type} is ready",
                "table_issues": [],
                "embedding_issues": [],
                "suggestions": []
            }
        
        with patch.object(factory, 'validate_pipeline_type', side_effect=mock_validate):
            result = factory.list_available_pipelines()
            
            assert "basic" in result
            assert "colbert" in result
            assert "crag" in result
            
            for pipeline_type, status in result.items():
                assert status["pipeline_type"] == pipeline_type
                assert status["valid"] is True


class TestIntegration:
    """Integration tests for the validation system."""

    @pytest.fixture
    def mock_config_manager(self):
        """Mock ConfigurationManager."""
        config_manager = Mock()
        config_manager.get_database_config.return_value = {}
        config_manager.get.side_effect = lambda key, default=None: {
            "embeddings.backend": "sentence_transformers",
            "embeddings.model": "all-MiniLM-L6-v2",
            "pipelines.colbert.token_embedding_dimension": 768,
            "rag.schemas.SourceDocuments.embedding.dimension": 384,
        }.get(key, default)
        return config_manager

    def test_iris_rag_create_pipeline_with_validation(self, mock_config_manager):
        """Test creating pipeline through iris_rag.create_pipeline with validation."""
        with patch('iris_rag.ConfigurationManager', return_value=mock_config_manager):
            with patch('iris_rag.ConnectionManager'):
                # Don't mock the factory - let it create a real pipeline
                import iris_rag
                result = iris_rag.create_pipeline(
                    "basic",
                    validate_requirements=True,
                    auto_setup=True
                )
    
                # Assert that we get a real BasicRAGPipeline instance
                assert hasattr(result, 'query')  # BasicRAGPipeline should have a query method
                assert result.__class__.__name__ == 'BasicRAGPipeline'

    def test_iris_rag_validate_pipeline(self, mock_config_manager):
        """Test validating pipeline through iris_rag.validate_pipeline."""
        with patch('iris_rag.ConfigurationManager', return_value=mock_config_manager):
            with patch('iris_rag.ConnectionManager'):
                # Don't mock the factory - let it return real validation results
                import iris_rag
                result = iris_rag.validate_pipeline("basic")
    
                # Assert that we get a real validation result with expected structure
                assert isinstance(result, dict)
                assert "pipeline_type" in result
                assert "valid" in result
                assert "summary" in result
                assert result["pipeline_type"] == "basic"

    def test_iris_rag_setup_pipeline(self, mock_config_manager):
        """Test setting up pipeline through iris_rag.setup_pipeline."""
        with patch('iris_rag.ConfigurationManager', return_value=mock_config_manager):
            with patch('iris_rag.ConnectionManager'):
                # Don't mock the factory - let it return real setup results
                import iris_rag
                result = iris_rag.setup_pipeline("basic")
    
                # Assert that we get a real setup result with expected structure
                assert isinstance(result, dict)
                assert "pipeline_type" in result
                assert "success" in result
                assert "setup_completed" in result
                assert result["pipeline_type"] == "basic"