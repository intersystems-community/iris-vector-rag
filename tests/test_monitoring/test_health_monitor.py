"""
Tests for the Health Monitor module.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from iris_rag.monitoring.health_monitor import HealthMonitor, HealthCheckResult
from iris_rag.config.manager import ConfigurationManager

class TestHealthMonitor:
    """Test cases for HealthMonitor."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        config_manager = Mock(spec=ConfigurationManager)
        return config_manager
    
    @pytest.fixture
    def health_monitor(self, mock_config_manager):
        """Create a HealthMonitor instance for testing."""
        with patch('iris_rag.monitoring.health_monitor.ConnectionManager'):
            with patch('docker.from_env'):
                monitor = HealthMonitor(mock_config_manager)
                return monitor
    
    def test_health_check_result_creation(self):
        """Test HealthCheckResult creation."""
        result = HealthCheckResult(
            component='test_component',
            status='healthy',
            message='Test message',
            metrics={'test_metric': 100},
            timestamp=datetime.now(),
            duration_ms=50.0
        )
        
        assert result.component == 'test_component'
        assert result.status == 'healthy'
        assert result.message == 'Test message'
        assert result.metrics['test_metric'] == 100
        assert result.duration_ms == 50.0
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_check_system_resources_healthy(self, mock_disk, mock_cpu, mock_memory, health_monitor):
        """Test system resources check when healthy."""
        # Mock healthy system resources
        mock_memory.return_value = Mock(percent=50.0, used=4*1024**3, total=8*1024**3)
        mock_cpu.return_value = 30.0
        mock_disk.return_value = Mock(percent=60.0, free=100*1024**3, total=200*1024**3)
        
        result = health_monitor.check_system_resources()
        
        assert result.component == 'system_resources'
        assert result.status == 'healthy'
        assert 'healthy' in result.message.lower()
        assert 'memory_percent' in result.metrics
        assert 'cpu_percent' in result.metrics
        assert 'disk_percent' in result.metrics
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_check_system_resources_warning(self, mock_disk, mock_cpu, mock_memory, health_monitor):
        """Test system resources check with warning levels."""
        # Mock warning-level system resources
        mock_memory.return_value = Mock(percent=85.0, used=6.8*1024**3, total=8*1024**3)
        mock_cpu.return_value = 85.0
        mock_disk.return_value = Mock(percent=88.0, free=24*1024**3, total=200*1024**3)
        
        result = health_monitor.check_system_resources()
        
        assert result.component == 'system_resources'
        assert result.status == 'warning'
        assert 'high' in result.message.lower()
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_check_system_resources_critical(self, mock_disk, mock_cpu, mock_memory, health_monitor):
        """Test system resources check with critical levels."""
        # Mock critical system resources
        mock_memory.return_value = Mock(percent=95.0, used=7.6*1024**3, total=8*1024**3)
        mock_cpu.return_value = 95.0
        mock_disk.return_value = Mock(percent=98.0, free=4*1024**3, total=200*1024**3)
        
        result = health_monitor.check_system_resources()
        
        assert result.component == 'system_resources'
        assert result.status == 'critical'
        assert 'critical' in result.message.lower()
    
    def test_check_database_connectivity_success(self, health_monitor):
        """Test successful database connectivity check."""
        # Mock successful database connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results
        mock_cursor.fetchone.side_effect = [
            [1],  # Basic connectivity test
            [5],  # Table count
            ['test_vector'],  # Vector operations test
            [1000],  # Document count
            [800]   # Embedded document count
        ]
        
        health_monitor.connection_manager.get_connection.return_value = mock_connection
        
        result = health_monitor.check_database_connectivity()
        
        assert result.component == 'database_connectivity'
        assert result.status == 'healthy'
        assert 'healthy' in result.message.lower()
        assert 'table_count' in result.metrics
        assert 'document_count' in result.metrics
        assert 'embedded_document_count' in result.metrics
    
    def test_check_database_connectivity_failure(self, health_monitor):
        """Test database connectivity check failure."""
        # Mock database connection failure
        health_monitor.connection_manager.get_connection.side_effect = Exception("Connection failed")
        
        result = health_monitor.check_database_connectivity()
        
        assert result.component == 'database_connectivity'
        assert result.status == 'critical'
        assert 'failed' in result.message.lower()
        assert result.metrics == {}
    
    def test_check_docker_containers_healthy(self, health_monitor):
        """Test Docker containers check when healthy."""
        # Mock healthy IRIS container
        mock_container = Mock()
        mock_container.name = 'iris_db'
        mock_container.status = 'running'
        mock_container.stats.return_value = {
            'memory_stats': {
                'usage': 2 * 1024**3,  # 2GB
                'limit': 4 * 1024**3   # 4GB
            }
        }
        
        health_monitor.docker_client.containers.list.return_value = [mock_container]
        
        result = health_monitor.check_docker_containers()
        
        assert result.component == 'docker_containers'
        assert result.status == 'healthy'
        assert 'healthy' in result.message.lower()
        assert 'container_status' in result.metrics
        assert result.metrics['container_status'] == 'running'
    
    def test_check_docker_containers_not_found(self, health_monitor):
        """Test Docker containers check when IRIS container not found."""
        # Mock no IRIS container
        health_monitor.docker_client.containers.list.return_value = []
        
        result = health_monitor.check_docker_containers()
        
        assert result.component == 'docker_containers'
        assert result.status == 'critical'
        assert 'not found' in result.message.lower()
    
    def test_check_docker_containers_not_running(self, health_monitor):
        """Test Docker containers check when IRIS container not running."""
        # Mock stopped IRIS container
        mock_container = Mock()
        mock_container.name = 'iris_db'
        mock_container.status = 'stopped'
        
        health_monitor.docker_client.containers.list.return_value = [mock_container]
        
        result = health_monitor.check_docker_containers()
        
        assert result.component == 'docker_containers'
        assert result.status == 'critical'
        assert 'not running' in result.message.lower()
        assert result.metrics['container_status'] == 'stopped'
    
    def test_check_vector_performance_success(self, health_monitor):
        """Test successful vector performance check."""
        # Mock database connection and queries
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results
        mock_cursor.fetchone.return_value = [1000]  # Embedded document count
        mock_cursor.fetchall.return_value = [
            ['doc1', 0.95],
            ['doc2', 0.90],
            ['doc3', 0.85]
        ]
        
        health_monitor.connection_manager.get_connection.return_value = mock_connection
        
        result = health_monitor.check_vector_performance()
        
        assert result.component == 'vector_performance'
        assert result.status == 'healthy'
        assert 'embedded_document_count' in result.metrics
        assert 'query_time_ms' in result.metrics
        assert 'results_returned' in result.metrics
    
    def test_check_vector_performance_insufficient_data(self, health_monitor):
        """Test vector performance check with insufficient data."""
        # Mock database connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock insufficient embedded documents
        mock_cursor.fetchone.return_value = [5]  # Only 5 embedded documents
        
        health_monitor.connection_manager.get_connection.return_value = mock_connection
        
        result = health_monitor.check_vector_performance()
        
        assert result.component == 'vector_performance'
        assert result.status == 'warning'
        assert 'insufficient' in result.message.lower()
        assert result.metrics['embedded_document_count'] == 5
    
    def test_run_comprehensive_health_check(self, health_monitor):
        """Test comprehensive health check execution."""
        # Mock all individual checks to return healthy results
        health_monitor.check_system_resources = Mock(return_value=HealthCheckResult(
            component='system_resources',
            status='healthy',
            message='System resources healthy',
            metrics={},
            timestamp=datetime.now(),
            duration_ms=10.0
        ))
        
        health_monitor.check_database_connectivity = Mock(return_value=HealthCheckResult(
            component='database_connectivity',
            status='healthy',
            message='Database connectivity healthy',
            metrics={},
            timestamp=datetime.now(),
            duration_ms=20.0
        ))
        
        health_monitor.check_docker_containers = Mock(return_value=HealthCheckResult(
            component='docker_containers',
            status='healthy',
            message='Docker containers healthy',
            metrics={},
            timestamp=datetime.now(),
            duration_ms=15.0
        ))
        
        health_monitor.check_vector_performance = Mock(return_value=HealthCheckResult(
            component='vector_performance',
            status='healthy',
            message='Vector performance healthy',
            metrics={},
            timestamp=datetime.now(),
            duration_ms=30.0
        ))
        
        results = health_monitor.run_comprehensive_health_check()
        
        assert len(results) == 4
        assert 'system_resources' in results
        assert 'database_connectivity' in results
        assert 'docker_containers' in results
        assert 'vector_performance' in results
        
        # Verify all checks were called
        health_monitor.check_system_resources.assert_called_once()
        health_monitor.check_database_connectivity.assert_called_once()
        health_monitor.check_docker_containers.assert_called_once()
        health_monitor.check_vector_performance.assert_called_once()
    
    def test_get_overall_health_status_healthy(self, health_monitor):
        """Test overall health status when all components are healthy."""
        results = {
            'comp1': HealthCheckResult('comp1', 'healthy', 'msg', {}, datetime.now(), 10.0),
            'comp2': HealthCheckResult('comp2', 'healthy', 'msg', {}, datetime.now(), 10.0)
        }
        
        status = health_monitor.get_overall_health_status(results)
        assert status == 'healthy'
    
    def test_get_overall_health_status_warning(self, health_monitor):
        """Test overall health status when some components have warnings."""
        results = {
            'comp1': HealthCheckResult('comp1', 'healthy', 'msg', {}, datetime.now(), 10.0),
            'comp2': HealthCheckResult('comp2', 'warning', 'msg', {}, datetime.now(), 10.0)
        }
        
        status = health_monitor.get_overall_health_status(results)
        assert status == 'warning'
    
    def test_get_overall_health_status_critical(self, health_monitor):
        """Test overall health status when some components are critical."""
        results = {
            'comp1': HealthCheckResult('comp1', 'healthy', 'msg', {}, datetime.now(), 10.0),
            'comp2': HealthCheckResult('comp2', 'warning', 'msg', {}, datetime.now(), 10.0),
            'comp3': HealthCheckResult('comp3', 'critical', 'msg', {}, datetime.now(), 10.0)
        }
        
        status = health_monitor.get_overall_health_status(results)
        assert status == 'critical'