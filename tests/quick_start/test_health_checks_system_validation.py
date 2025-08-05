"""
Comprehensive tests for health checks and system validation in the Quick Start system.

This test suite covers the complete health monitoring and system validation
integration with the Quick Start system, ensuring all components are properly
monitored and validated.

Test Categories:
1. Health Monitor Integration Tests - Test health monitoring with Quick Start
2. System Validator Integration Tests - Test system validation with Quick Start
3. Quick Start Health Checks - Test Quick Start specific health checks
4. Profile-Specific Health Tests - Test health checks for each profile
5. Docker Health Integration Tests - Test Docker container health monitoring
6. End-to-End Health Validation - Test complete health validation workflows

Following TDD principles: Write failing tests first, then implement to pass.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Import the components we'll be testing
try:
    from iris_rag.monitoring.health_monitor import HealthMonitor, HealthCheckResult
    from iris_rag.monitoring.system_validator import SystemValidator, ValidationResult
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.connection import ConnectionManager
except ImportError:
    # These modules exist - we'll test integration with them
    HealthMonitor = None
    SystemValidator = None
    ConfigurationManager = None
    ConnectionManager = None

# Import Quick Start components for integration testing
from quick_start.cli.wizard import QuickStartCLIWizard, CLIWizardResult
from quick_start.setup.pipeline import OneCommandSetupPipeline
from quick_start.data.sample_manager import SampleDataManager
from quick_start.config.template_engine import ConfigurationTemplateEngine
from quick_start.docker.compose_generator import DockerComposeGenerator
from quick_start.docker.service_manager import DockerServiceManager

# Import Quick Start health integration components (to be implemented)
try:
    from quick_start.monitoring.health_integration import QuickStartHealthMonitor
    from quick_start.monitoring.system_validation import QuickStartSystemValidator
    from quick_start.monitoring.profile_health import ProfileHealthChecker
    from quick_start.monitoring.docker_health import DockerHealthMonitor
except ImportError:
    # These modules don't exist yet - we'll implement them to make tests pass
    QuickStartHealthMonitor = None
    QuickStartSystemValidator = None
    ProfileHealthChecker = None
    DockerHealthMonitor = None


class TestQuickStartHealthIntegration:
    """Test health monitoring integration with Quick Start system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp(prefix="health_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager for testing."""
        mock_config = Mock()
        mock_config.get_config.return_value = {
            'profile': 'minimal',
            'database': {
                'host': 'localhost',
                'port': 1972,
                'username': 'demo',
                'password': 'demo',
                'namespace': 'USER'
            },
            'docker': {
                'enabled': True,
                'compose_file': 'docker-compose.quick-start.yml'
            },
            'monitoring': {
                'health_check_interval': 30,
                'alert_thresholds': {
                    'cpu_percent': 80,
                    'memory_percent': 85,
                    'disk_percent': 90
                }
            }
        }
        return mock_config
    
    @pytest.fixture
    def mock_health_monitor(self):
        """Mock health monitor for testing."""
        mock_monitor = Mock()
        mock_monitor.check_system_health.return_value = HealthCheckResult(
            component="system",
            status="healthy",
            message="All systems operational",
            metrics={"cpu_percent": 45.2, "memory_percent": 62.1},
            timestamp=datetime.now(),
            duration_ms=150.5
        )
        return mock_monitor
    
    @pytest.fixture
    def mock_system_validator(self):
        """Mock system validator for testing."""
        mock_validator = Mock()
        mock_validator.validate_system.return_value = ValidationResult(
            test_name="system_validation",
            success=True,
            message="System validation passed",
            details={"tests_passed": 15, "tests_failed": 0},
            duration_ms=2500.0,
            timestamp=datetime.now()
        )
        return mock_validator
    
    def test_quick_start_health_monitor_initialization(self, mock_config_manager):
        """Test QuickStartHealthMonitor initialization."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        # Test initialization with config manager
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        
        assert health_monitor is not None
        assert health_monitor.config_manager == mock_config_manager
        assert hasattr(health_monitor, 'base_health_monitor')
        assert hasattr(health_monitor, 'profile_checker')
        assert hasattr(health_monitor, 'docker_health_monitor')
    
    def test_quick_start_health_monitor_check_quick_start_health(self, mock_config_manager, mock_health_monitor):
        """Test Quick Start specific health checks."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        with patch('quick_start.monitoring.health_integration.HealthMonitor', return_value=mock_health_monitor):
            health_monitor = QuickStartHealthMonitor(mock_config_manager)
            
            # Test Quick Start health check
            result = health_monitor.check_quick_start_health()
            
            assert isinstance(result, dict)
            assert 'overall_status' in result
            assert 'components' in result
            assert 'profile_health' in result['components']
            assert 'setup_pipeline_health' in result['components']
            assert 'configuration_health' in result['components']
            assert 'timestamp' in result
    
    def test_quick_start_health_monitor_check_profile_health(self, mock_config_manager):
        """Test profile-specific health checks."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        
        # Test minimal profile health
        result = health_monitor.check_profile_health('minimal')
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'profile_minimal'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'document_count' in result.metrics
        assert 'resource_usage' in result.metrics
        assert 'expected_services' in result.metrics
    
    def test_quick_start_health_monitor_check_setup_pipeline_health(self, mock_config_manager, temp_dir):
        """Test setup pipeline health checks."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        
        # Test setup pipeline health
        result = health_monitor.check_setup_pipeline_health()
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'setup_pipeline'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'pipeline_status' in result.metrics
        assert 'last_setup_time' in result.metrics
        assert 'configuration_valid' in result.metrics
    
    def test_quick_start_health_monitor_check_configuration_health(self, mock_config_manager):
        """Test configuration health checks."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        
        # Test configuration health
        result = health_monitor.check_configuration_health()
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'configuration'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'template_engine_status' in result.metrics
        assert 'schema_validation_status' in result.metrics
        assert 'environment_variables_status' in result.metrics


class TestQuickStartSystemValidation:
    """Test system validation integration with Quick Start system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp(prefix="validation_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager for testing."""
        mock_config = Mock()
        mock_config.get_config.return_value = {
            'profile': 'standard',
            'database': {
                'host': 'localhost',
                'port': 1972,
                'username': 'demo',
                'password': 'demo',
                'namespace': 'USER'
            },
            'sample_data': {
                'document_count': 500,
                'source': 'pmc_sample'
            },
            'validation': {
                'run_integrity_checks': True,
                'run_performance_tests': True,
                'run_pipeline_tests': True
            }
        }
        return mock_config
    
    def test_quick_start_system_validator_initialization(self, mock_config_manager):
        """Test QuickStartSystemValidator initialization."""
        if QuickStartSystemValidator is None:
            pytest.skip("QuickStartSystemValidator not implemented yet")
        
        # Test initialization with config manager
        validator = QuickStartSystemValidator(mock_config_manager)
        
        assert validator is not None
        assert validator.config_manager == mock_config_manager
        assert hasattr(validator, 'base_validator')
        assert hasattr(validator, 'health_monitor')
        assert hasattr(validator, 'sample_data_manager')
    
    def test_quick_start_system_validator_validate_quick_start_setup(self, mock_config_manager):
        """Test Quick Start setup validation."""
        if QuickStartSystemValidator is None:
            pytest.skip("QuickStartSystemValidator not implemented yet")
        
        validator = QuickStartSystemValidator(mock_config_manager)
        
        # Test Quick Start setup validation
        result = validator.validate_quick_start_setup()
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == 'quick_start_setup'
        assert isinstance(result.success, bool)
        assert 'configuration_valid' in result.details
        assert 'templates_valid' in result.details
        assert 'sample_data_valid' in result.details
        assert 'pipeline_functional' in result.details
    
    def test_quick_start_system_validator_validate_profile_configuration(self, mock_config_manager):
        """Test profile configuration validation."""
        if QuickStartSystemValidator is None:
            pytest.skip("QuickStartSystemValidator not implemented yet")
        
        validator = QuickStartSystemValidator(mock_config_manager)
        
        # Test profile configuration validation
        result = validator.validate_profile_configuration('standard')
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == 'profile_configuration_standard'
        assert isinstance(result.success, bool)
        assert 'profile_exists' in result.details
        assert 'schema_valid' in result.details
        assert 'resource_requirements_met' in result.details
        assert 'dependencies_available' in result.details
    
    def test_quick_start_system_validator_validate_sample_data_integrity(self, mock_config_manager):
        """Test sample data integrity validation."""
        if QuickStartSystemValidator is None:
            pytest.skip("QuickStartSystemValidator not implemented yet")
        
        validator = QuickStartSystemValidator(mock_config_manager)
        
        # Test sample data integrity validation
        result = validator.validate_sample_data_integrity()
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == 'sample_data_integrity'
        assert isinstance(result.success, bool)
        assert 'document_count' in result.details
        assert 'data_quality_score' in result.details
        assert 'missing_documents' in result.details
        assert 'corrupted_documents' in result.details
    
    def test_quick_start_system_validator_validate_pipeline_functionality(self, mock_config_manager):
        """Test pipeline functionality validation."""
        if QuickStartSystemValidator is None:
            pytest.skip("QuickStartSystemValidator not implemented yet")
        
        validator = QuickStartSystemValidator(mock_config_manager)
        
        # Test pipeline functionality validation
        result = validator.validate_pipeline_functionality()
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == 'pipeline_functionality'
        assert isinstance(result.success, bool)
        assert 'embedding_pipeline' in result.details
        assert 'retrieval_pipeline' in result.details
        assert 'generation_pipeline' in result.details
        assert 'end_to_end_test' in result.details


class TestProfileHealthChecker:
    """Test profile-specific health checking."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager for testing."""
        mock_config = Mock()
        mock_config.get_config.return_value = {
            'profile': 'extended',
            'profiles': {
                'minimal': {'document_count': 50, 'memory_limit': '2G'},
                'standard': {'document_count': 500, 'memory_limit': '4G'},
                'extended': {'document_count': 5000, 'memory_limit': '8G'}
            }
        }
        return mock_config
    
    def test_profile_health_checker_initialization(self, mock_config_manager):
        """Test ProfileHealthChecker initialization."""
        if ProfileHealthChecker is None:
            pytest.skip("ProfileHealthChecker not implemented yet")
        
        # Test initialization
        checker = ProfileHealthChecker(mock_config_manager)
        
        assert checker is not None
        assert checker.config_manager == mock_config_manager
        assert hasattr(checker, 'supported_profiles')
        assert 'minimal' in checker.supported_profiles
        assert 'standard' in checker.supported_profiles
        assert 'extended' in checker.supported_profiles
    
    def test_profile_health_checker_check_minimal_profile(self, mock_config_manager):
        """Test minimal profile health check."""
        if ProfileHealthChecker is None:
            pytest.skip("ProfileHealthChecker not implemented yet")
        
        checker = ProfileHealthChecker(mock_config_manager)
        
        # Test minimal profile health check
        result = checker.check_profile_health('minimal')
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'profile_minimal'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'expected_document_count' in result.metrics
        assert result.metrics['expected_document_count'] == 50
        assert 'memory_usage' in result.metrics
        assert 'cpu_usage' in result.metrics
    
    def test_profile_health_checker_check_standard_profile(self, mock_config_manager):
        """Test standard profile health check."""
        if ProfileHealthChecker is None:
            pytest.skip("ProfileHealthChecker not implemented yet")
        
        checker = ProfileHealthChecker(mock_config_manager)
        
        # Test standard profile health check
        result = checker.check_profile_health('standard')
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'profile_standard'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'expected_document_count' in result.metrics
        assert result.metrics['expected_document_count'] == 500
        assert 'mcp_server_status' in result.metrics
        assert 'service_count' in result.metrics
    
    def test_profile_health_checker_check_extended_profile(self, mock_config_manager):
        """Test extended profile health check."""
        if ProfileHealthChecker is None:
            pytest.skip("ProfileHealthChecker not implemented yet")
        
        checker = ProfileHealthChecker(mock_config_manager)
        
        # Test extended profile health check
        result = checker.check_profile_health('extended')
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'profile_extended'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'expected_document_count' in result.metrics
        assert result.metrics['expected_document_count'] == 5000
        assert 'nginx_status' in result.metrics
        assert 'monitoring_services_status' in result.metrics
        assert 'scaling_metrics' in result.metrics
    
    def test_profile_health_checker_validate_profile_requirements(self, mock_config_manager):
        """Test profile requirements validation."""
        if ProfileHealthChecker is None:
            pytest.skip("ProfileHealthChecker not implemented yet")
        
        checker = ProfileHealthChecker(mock_config_manager)
        
        # Test profile requirements validation
        result = checker.validate_profile_requirements('extended')
        
        assert isinstance(result, dict)
        assert 'memory_sufficient' in result
        assert 'cpu_sufficient' in result
        assert 'disk_space_sufficient' in result
        assert 'dependencies_available' in result
        assert 'ports_available' in result


class TestDockerHealthIntegration:
    """Test Docker health monitoring integration."""
    
    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client for testing."""
        mock_client = Mock()
        mock_container = Mock()
        mock_container.status = 'running'
        mock_container.attrs = {
            'State': {'Health': {'Status': 'healthy'}},
            'Config': {'Labels': {'com.docker.compose.service': 'iris'}}
        }
        mock_client.containers.list.return_value = [mock_container]
        return mock_client
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager for testing."""
        mock_config = Mock()
        mock_config.get_config.return_value = {
            'docker': {
                'enabled': True,
                'compose_file': 'docker-compose.quick-start.yml',
                'services': ['iris', 'rag_app', 'mcp_server']
            }
        }
        return mock_config
    
    def test_docker_health_monitor_initialization(self, mock_config_manager):
        """Test DockerHealthMonitor initialization."""
        if DockerHealthMonitor is None:
            pytest.skip("DockerHealthMonitor not implemented yet")
        
        # Test initialization
        monitor = DockerHealthMonitor(mock_config_manager)
        
        assert monitor is not None
        assert monitor.config_manager == mock_config_manager
        assert hasattr(monitor, 'docker_client')
        assert hasattr(monitor, 'service_manager')
    
    @patch('docker.from_env')
    def test_docker_health_monitor_check_container_health(self, mock_docker_from_env, mock_config_manager, mock_docker_client):
        """Test Docker container health checks."""
        if DockerHealthMonitor is None:
            pytest.skip("DockerHealthMonitor not implemented yet")
        
        mock_docker_from_env.return_value = mock_docker_client
        
        monitor = DockerHealthMonitor(mock_config_manager)
        
        # Test container health check
        result = monitor.check_container_health('iris')
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'docker_container_iris'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'container_status' in result.metrics
        assert 'health_status' in result.metrics
        assert 'uptime' in result.metrics
    
    @patch('docker.from_env')
    def test_docker_health_monitor_check_all_services_health(self, mock_docker_from_env, mock_config_manager, mock_docker_client):
        """Test all Docker services health check."""
        if DockerHealthMonitor is None:
            pytest.skip("DockerHealthMonitor not implemented yet")
        
        mock_docker_from_env.return_value = mock_docker_client
        
        monitor = DockerHealthMonitor(mock_config_manager)
        
        # Test all services health check
        result = monitor.check_all_services_health()
        
        assert isinstance(result, dict)
        assert 'overall_status' in result
        assert 'services' in result
        assert 'healthy_count' in result
        assert 'unhealthy_count' in result
        assert 'total_count' in result
    
    @patch('docker.from_env')
    def test_docker_health_monitor_check_compose_file_health(self, mock_docker_from_env, mock_config_manager, mock_docker_client):
        """Test Docker compose file health check."""
        if DockerHealthMonitor is None:
            pytest.skip("DockerHealthMonitor not implemented yet")
        
        mock_docker_from_env.return_value = mock_docker_client
        
        monitor = DockerHealthMonitor(mock_config_manager)
        
        # Test compose file health check
        result = monitor.check_compose_file_health()
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == 'docker_compose_file'
        assert result.status in ['healthy', 'warning', 'critical']
        assert 'file_exists' in result.metrics
        assert 'file_valid' in result.metrics
        assert 'services_defined' in result.metrics


class TestEndToEndHealthValidation:
    """Test end-to-end health validation workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp(prefix="e2e_health_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager for testing."""
        mock_config = Mock()
        mock_config.get_config.return_value = {
            'profile': 'minimal',
            'monitoring': {
                'enabled': True,
                'health_check_interval': 30,
                'validation_on_startup': True
            }
        }
        return mock_config
    
    def test_complete_health_validation_workflow_minimal(self, mock_config_manager, temp_dir):
        """Test complete health validation workflow for minimal profile."""
        if QuickStartHealthMonitor is None or QuickStartSystemValidator is None:
            pytest.skip("Health monitoring components not implemented yet")
        
        # Test complete workflow
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        validator = QuickStartSystemValidator(mock_config_manager)
        
        # Run complete health validation
        health_result = health_monitor.check_quick_start_health()
        validation_result = validator.validate_quick_start_setup()
        
        # Verify results
        assert isinstance(health_result, dict)
        assert isinstance(validation_result, ValidationResult)
        assert 'overall_status' in health_result
        assert validation_result.test_name == 'quick_start_setup'
    
    def test_complete_health_validation_workflow_standard(self, mock_config_manager, temp_dir):
        """Test complete health validation workflow for standard profile."""
        if QuickStartHealthMonitor is None or QuickStartSystemValidator is None:
            pytest.skip("Health monitoring components not implemented yet")
        
        # Update config for standard profile
        mock_config_manager.get_config.return_value['profile'] = 'standard'
        
        # Test complete workflow
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        validator = QuickStartSystemValidator(mock_config_manager)
        
        # Run complete health validation
        health_result = health_monitor.check_quick_start_health()
        validation_result = validator.validate_quick_start_setup()
        
        # Verify results
        assert isinstance(health_result, dict)
        assert isinstance(validation_result, ValidationResult)
        assert 'mcp_server_health' in health_result['components']
    
    def test_complete_health_validation_workflow_extended(self, mock_config_manager, temp_dir):
        """Test complete health validation workflow for extended profile."""
        if QuickStartHealthMonitor is None or QuickStartSystemValidator is None:
            pytest.skip("Health monitoring components not implemented yet")
        
        # Update config for extended profile
        mock_config_manager.get_config.return_value['profile'] = 'extended'
        
        # Test complete workflow
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        validator = QuickStartSystemValidator(mock_config_manager)
        
        # Run complete health validation
        health_result = health_monitor.check_quick_start_health()
        validation_result = validator.validate_quick_start_setup()
        
        # Verify results
        assert isinstance(health_result, dict)
        assert isinstance(validation_result, ValidationResult)
        assert 'monitoring_services_health' in health_result['components']
        assert 'nginx_health' in health_result['components']
    
    def test_health_validation_with_docker_integration(self, mock_config_manager, temp_dir):
        """Test health validation with Docker integration."""
        if QuickStartHealthMonitor is None or DockerHealthMonitor is None:
            pytest.skip("Health monitoring components not implemented yet")
        
        # Enable Docker in config
        mock_config_manager.get_config.return_value['docker'] = {'enabled': True}
        
        # Test health validation with Docker
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        docker_monitor = DockerHealthMonitor(mock_config_manager)
        
        # Run health checks
        quick_start_health = health_monitor.check_quick_start_health()
        docker_health = docker_monitor.check_all_services_health()
        
        # Verify integration
        assert isinstance(quick_start_health, dict)
        assert isinstance(docker_health, dict)
        assert 'docker_health' in quick_start_health['components']
    
    def test_health_validation_error_handling(self, mock_config_manager):
        """Test health validation error handling."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        # Test with invalid configuration
        mock_config_manager.get_config.side_effect = Exception("Configuration error")
        
        # Test error handling
        try:
            health_monitor = QuickStartHealthMonitor(mock_config_manager)
            result = health_monitor.check_quick_start_health()
            
            # Should handle errors gracefully
            assert isinstance(result, dict)
            assert result['overall_status'] == 'critical'
            assert 'error' in result
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"Unhandled exception: {e}")
    
    def test_health_validation_performance_monitoring(self, mock_config_manager):
        """Test health validation performance monitoring."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        health_monitor = QuickStartHealthMonitor(mock_config_manager)
        
        # Test performance monitoring
        start_time = time.time()
        result = health_monitor.check_quick_start_health()
        end_time = time.time()
        
        # Verify performance metrics
        assert isinstance(result, dict)
        assert 'performance_metrics' in result
        assert 'total_duration_ms' in result['performance_metrics']
        assert result['performance_metrics']['total_duration_ms'] > 0
        assert (end_time - start_time) * 1000 >= result['performance_metrics']['total_duration_ms']


class TestHealthCheckIntegrationWithQuickStartComponents:
    """Test health check integration with existing Quick Start components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp(prefix="integration_health_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Sample configuration for testing."""
        return {
            'profile': 'minimal',
            'database': {
                'host': 'localhost',
                'port': 1972,
                'username': 'demo',
                'password': 'demo',
                'namespace': 'USER'
            },
            'sample_data': {
                'document_count': 50,
                'source': 'pmc_sample'
            },
            'output_dir': str(temp_dir)
        }
    
    def test_health_integration_with_cli_wizard(self, sample_config, temp_dir):
        """Test health monitoring integration with CLI wizard."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        # Mock CLI wizard result
        wizard_result = CLIWizardResult(
            success=True,
            profile='minimal',
            config=sample_config,
            files_created=[str(temp_dir / "config.yaml")],
            errors=[],
            warnings=[]
        )
        
        # Test health integration with wizard
        with patch('quick_start.monitoring.health_integration.QuickStartCLIWizard') as mock_wizard:
            mock_wizard.return_value.run_wizard.return_value = wizard_result
            
            # Create health monitor and check wizard integration
            health_monitor = QuickStartHealthMonitor()
            result = health_monitor.check_wizard_integration()
            
            assert isinstance(result, HealthCheckResult)
            assert result.component == 'cli_wizard_integration'
            assert result.status in ['healthy', 'warning', 'critical']
            assert 'wizard_functional' in result.metrics
    
    def test_health_integration_with_setup_pipeline(self, sample_config, temp_dir):
        """Test health monitoring integration with setup pipeline."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        # Test health integration with setup pipeline
        with patch('quick_start.monitoring.health_integration.OneCommandSetupPipeline') as mock_pipeline:
            mock_pipeline.return_value.execute_setup.return_value = {
                'success': True,
                'steps_completed': 5,
                'total_steps': 5,
                'duration_seconds': 45.2
            }
            
            # Create health monitor and check pipeline integration
            health_monitor = QuickStartHealthMonitor()
            result = health_monitor.check_pipeline_integration()
            
            assert isinstance(result, HealthCheckResult)
            assert result.component == 'setup_pipeline_integration'
            assert result.status in ['healthy', 'warning', 'critical']
            assert 'pipeline_functional' in result.metrics
            assert 'last_execution_successful' in result.metrics
    
    def test_health_integration_with_sample_data_manager(self, sample_config, temp_dir):
        """Test health monitoring integration with sample data manager."""
        if QuickStartHealthMonitor is None:
            pytest.skip("QuickStartHealthMonitor not implemented yet")
        
        # Test health integration with sample data manager
        with patch('quick_start.monitoring.health_integration.SampleDataManager') as mock_manager:
            mock_manager.return_value.get_status.return_value = {
                'documents_loaded': 50,
                'data_quality_score': 0.95,
                'last_update': datetime.now().isoformat()
            }
            
            # Create health monitor and check sample data integration
            health_monitor = QuickStartHealthMonitor()
            result = health_monitor.check_sample_data_integration()
            
            assert isinstance(result, HealthCheckResult)
            assert result.component == 'sample_data_integration'
            assert result.status in ['healthy', 'warning', 'critical']
            assert 'data_manager_functional' in result.metrics
            assert 'document_count_valid' in result.metrics
    
    def test_health_integration_with_docker_services(self, sample_config, temp_dir):
        """Test health monitoring integration with Docker services."""
        if QuickStartHealthMonitor is None or DockerHealthMonitor is None:
            pytest.skip("Health monitoring components not implemented yet")
        
        # Test health integration with Docker services
        with patch('quick_start.monitoring.health_integration.DockerServiceManager') as mock_service_manager:
            mock_service_manager.return_value.get_service_status.return_value = {
                'iris': 'running',
                'rag_app': 'running',
                'mcp_server': 'running'
            }
            
            # Create health monitor and check Docker integration
            health_monitor = QuickStartHealthMonitor()
            docker_monitor = DockerHealthMonitor()
            
            result = health_monitor.check_docker_integration()
            
            assert isinstance(result, HealthCheckResult)
            assert result.component == 'docker_integration'
            assert result.status in ['healthy', 'warning', 'critical']
            assert 'docker_services_functional' in result.metrics
            assert 'compose_file_valid' in result.metrics