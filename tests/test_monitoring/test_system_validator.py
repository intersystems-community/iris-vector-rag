"""
Tests for the System Validator module.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from iris_rag.monitoring.system_validator import SystemValidator, ValidationResult
from iris_rag.config.manager import ConfigurationManager

class TestSystemValidator:
    """Test cases for SystemValidator."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        return Mock(spec=ConfigurationManager)
    
    @pytest.fixture
    def system_validator(self, mock_config_manager):
        """Create a SystemValidator instance for testing."""
        with patch('iris_rag.monitoring.system_validator.ConnectionManager'):
            with patch('iris_rag.monitoring.system_validator.HealthMonitor'):
                validator = SystemValidator(mock_config_manager)
                return validator
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            test_name='test_validation',
            success=True,
            message='Test passed',
            details={'test_detail': 'value'},
            duration_ms=100.0,
            timestamp=datetime.now()
        )
        
        assert result.test_name == 'test_validation'
        assert result.success is True
        assert result.message == 'Test passed'
        assert result.details['test_detail'] == 'value'
        assert result.duration_ms == 100.0
    
    def test_validate_data_integrity_success(self, system_validator):
        """Test successful data integrity validation."""
        # Mock database connection and queries
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results for successful validation
        mock_cursor.fetchall.side_effect = [
            [],  # No duplicates
            [384],  # Consistent embedding dimensions
        ]
        mock_cursor.fetchone.side_effect = [
            [0],  # No null embeddings
            [0],  # No orphaned chunks
            [0],  # No empty content
        ]
        
        system_validator.connection_manager.get_connection.return_value = mock_connection
        
        result = system_validator.validate_data_integrity()
        
        assert result.test_name == 'data_integrity'
        assert result.success is True
        assert 'passed' in result.message.lower()
        assert 'documents_without_embeddings' in result.details
        assert 'embedding_dimensions' in result.details
    
    def test_validate_data_integrity_with_issues(self, system_validator):
        """Test data integrity validation with issues found."""
        # Mock database connection and queries
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results with issues
        mock_cursor.fetchall.side_effect = [
            [('doc1', 2), ('doc2', 3)],  # Duplicates found
            [384, 512],  # Inconsistent embedding dimensions
        ]
        mock_cursor.fetchone.side_effect = [
            [50],  # 50 documents without embeddings
            [10],  # 10 orphaned chunks
            [5],   # 5 documents with empty content
        ]
        
        system_validator.connection_manager.get_connection.return_value = mock_connection
        
        result = system_validator.validate_data_integrity()
        
        assert result.test_name == 'data_integrity'
        assert result.success is False
        assert 'issues' in result.message.lower()
        assert result.details['documents_without_embeddings'] == 50
        assert result.details['orphaned_chunks'] == 10
        assert result.details['documents_with_empty_content'] == 5
    
    def test_validate_data_integrity_database_error(self, system_validator):
        """Test data integrity validation with database error."""
        # Mock database connection failure
        system_validator.connection_manager.get_connection.side_effect = Exception("Database error")
        
        result = system_validator.validate_data_integrity()
        
        assert result.test_name == 'data_integrity'
        assert result.success is False
        assert 'failed' in result.message.lower()
        assert 'error' in result.details
    
    @patch('iris_rag.monitoring.system_validator.BasicRAGPipeline')
    def test_validate_pipeline_functionality_success(self, mock_pipeline_class, system_validator):
        """Test successful pipeline functionality validation."""
        # Mock pipeline execution
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = {
            'query': 'test query',
            'answer': 'test answer',
            'retrieved_documents': [{'doc_id': 'doc1'}, {'doc_id': 'doc2'}]
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        result = system_validator.validate_pipeline_functionality(['test query'])
        
        assert result.test_name == 'pipeline_functionality'
        assert result.success is True
        assert 'passed' in result.message.lower()
        assert result.details['test_queries_count'] == 1
        assert result.details['successful_queries'] == 1
        assert result.details['failed_queries'] == 0
    
    @patch('iris_rag.monitoring.system_validator.BasicRAGPipeline')
    def test_validate_pipeline_functionality_with_failures(self, mock_pipeline_class, system_validator):
        """Test pipeline functionality validation with failures."""
        # Mock pipeline execution with missing keys
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = {
            'query': 'test query',
            # Missing 'answer' and 'retrieved_documents'
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        result = system_validator.validate_pipeline_functionality(['test query'])
        
        assert result.test_name == 'pipeline_functionality'
        assert result.success is False
        assert 'issues' in result.message.lower()
        assert result.details['successful_queries'] == 0
        assert result.details['failed_queries'] == 1
        assert len(result.details['issues']) > 0
    
    @patch('iris_rag.monitoring.system_validator.BasicRAGPipeline')
    def test_validate_pipeline_functionality_exception(self, mock_pipeline_class, system_validator):
        """Test pipeline functionality validation with exception."""
        # Mock pipeline execution exception
        mock_pipeline_class.side_effect = Exception("Pipeline error")
        
        result = system_validator.validate_pipeline_functionality(['test query'])
        
        assert result.test_name == 'pipeline_functionality'
        assert result.success is False
        assert 'failed' in result.message.lower()
        assert 'error' in result.details
    
    def test_validate_vector_operations_success(self, system_validator):
        """Test successful vector operations validation."""
        # Mock database connection and queries
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock successful vector operations
        mock_cursor.fetchone.side_effect = [
            ['test_vector'],  # Vector creation
            [1.0],  # Vector similarity (should be 1.0 for identical vectors)
            [1000],  # Embedded document count
        ]
        mock_cursor.fetchall.side_effect = [
            [('doc1', 0.95), ('doc2', 0.90)],  # Vector query results
            [('embedding_idx', 'HNSW')],  # HNSW indexes
        ]
        
        system_validator.connection_manager.get_connection.return_value = mock_connection
        
        result = system_validator.validate_vector_operations()
        
        assert result.test_name == 'vector_operations'
        assert result.success is True
        assert 'passed' in result.message.lower()
        assert 'basic_vector_operations' in result.details
        assert 'vector_similarity' in result.details
        assert 'embedded_documents' in result.details
        assert 'vector_query_time_ms' in result.details
    
    def test_validate_vector_operations_no_embeddings(self, system_validator):
        """Test vector operations validation with no embedded documents."""
        # Mock database connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock no embedded documents
        mock_cursor.fetchone.side_effect = [
            ['test_vector'],  # Vector creation works
            [1.0],  # Vector similarity works
            [0],  # No embedded documents
        ]
        
        system_validator.connection_manager.get_connection.return_value = mock_connection
        
        result = system_validator.validate_vector_operations()
        
        assert result.test_name == 'vector_operations'
        assert result.success is False
        assert 'no embedded documents' in result.message.lower()
        assert result.details['embedded_documents'] == 0
    
    def test_validate_vector_operations_database_error(self, system_validator):
        """Test vector operations validation with database error."""
        # Mock database connection failure
        system_validator.connection_manager.get_connection.side_effect = Exception("Database error")
        
        result = system_validator.validate_vector_operations()
        
        assert result.test_name == 'vector_operations'
        assert result.success is False
        assert 'failed' in result.message.lower()
        assert 'error' in result.details
    
    @patch('builtins.__import__')
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_validate_system_configuration_success(self, mock_open, mock_exists, mock_import, system_validator):
        """Test successful system configuration validation."""
        # Mock successful package imports
        mock_import.return_value = Mock()
        
        # Mock config files exist and are valid
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = '{"test": "config"}'
        
        # Mock health check
        system_validator.health_monitor.run_comprehensive_health_check.return_value = {}
        system_validator.health_monitor.get_overall_health_status.return_value = 'healthy'
        
        result = system_validator.validate_system_configuration()
        
        assert result.test_name == 'system_configuration'
        assert result.success is True
        assert 'passed' in result.message.lower()
        assert 'overall_health_status' in result.details
    
    @patch('builtins.__import__')
    def test_validate_system_configuration_missing_packages(self, mock_import, system_validator):
        """Test system configuration validation with missing packages."""
        # Mock missing packages
        mock_import.side_effect = [ImportError("No module"), Mock(), Mock(), Mock()]
        
        # Mock health check
        system_validator.health_monitor.run_comprehensive_health_check.return_value = {}
        system_validator.health_monitor.get_overall_health_status.return_value = 'healthy'
        
        result = system_validator.validate_system_configuration()
        
        assert result.test_name == 'system_configuration'
        assert result.success is False
        assert 'issues' in result.message.lower()
    
    def test_run_comprehensive_validation(self, system_validator):
        """Test comprehensive validation execution."""
        # Mock all validation methods
        system_validator.validate_data_integrity = Mock(return_value=ValidationResult(
            'data_integrity', True, 'passed', {}, 10.0, datetime.now()
        ))
        system_validator.validate_pipeline_functionality = Mock(return_value=ValidationResult(
            'pipeline_functionality', True, 'passed', {}, 20.0, datetime.now()
        ))
        system_validator.validate_vector_operations = Mock(return_value=ValidationResult(
            'vector_operations', True, 'passed', {}, 15.0, datetime.now()
        ))
        system_validator.validate_system_configuration = Mock(return_value=ValidationResult(
            'system_configuration', True, 'passed', {}, 25.0, datetime.now()
        ))
        
        results = system_validator.run_comprehensive_validation()
        
        assert len(results) == 4
        assert 'data_integrity' in results
        assert 'pipeline_functionality' in results
        assert 'vector_operations' in results
        assert 'system_configuration' in results
        
        # Verify all validations were called
        system_validator.validate_data_integrity.assert_called_once()
        system_validator.validate_pipeline_functionality.assert_called_once()
        system_validator.validate_vector_operations.assert_called_once()
        system_validator.validate_system_configuration.assert_called_once()
    
    def test_generate_validation_report(self, system_validator):
        """Test validation report generation."""
        # Create mock validation results
        results = {
            'test1': ValidationResult('test1', True, 'passed', {'detail1': 'value1'}, 10.0, datetime.now()),
            'test2': ValidationResult('test2', False, 'failed', {'detail2': 'value2'}, 20.0, datetime.now())
        }
        
        report = system_validator.generate_validation_report(results)
        
        assert 'validation_timestamp' in report
        assert 'summary' in report
        assert 'validation_results' in report
        assert 'recommendations' in report
        
        summary = report['summary']
        assert summary['total_validations'] == 2
        assert summary['successful_validations'] == 1
        assert summary['failed_validations'] == 1
        assert summary['success_rate'] == 50.0
        assert summary['overall_status'] == 'FAIL'
    
    def test_generate_recommendations_all_passed(self, system_validator):
        """Test recommendation generation when all validations pass."""
        results = {
            'test1': ValidationResult('test1', True, 'passed', {}, 10.0, datetime.now()),
            'test2': ValidationResult('test2', True, 'passed', {}, 20.0, datetime.now())
        }
        
        recommendations = system_validator._generate_recommendations(results)
        
        assert len(recommendations) == 1
        assert 'no immediate actions required' in recommendations[0].lower()
    
    def test_generate_recommendations_with_failures(self, system_validator):
        """Test recommendation generation with validation failures."""
        results = {
            'data_integrity': ValidationResult('data_integrity', False, 'failed', {}, 10.0, datetime.now()),
            'pipeline_functionality': ValidationResult('pipeline_functionality', False, 'failed', {}, 20.0, datetime.now())
        }
        
        recommendations = system_validator._generate_recommendations(results)
        
        assert len(recommendations) == 2
        assert any('data cleanup' in rec.lower() for rec in recommendations)
        assert any('pipeline configuration' in rec.lower() for rec in recommendations)