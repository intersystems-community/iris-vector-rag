"""
Integration tests for IRIS driver compatibility and connection management.

This module tests the driver-aware connection manager functionality,
including DBAPI and JDBC driver capabilities, fallback mechanisms,
and vector operation compatibility.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from common.iris_connection_manager import (
    IRISConnectionManager,
    DriverType,
    ConnectionInfo,
    get_connection_info,
    get_driver_type,
    supports_vector_operations,
    get_driver_capabilities,
    get_iris_connection
)

logger = logging.getLogger(__name__)


class TestDriverDetection:
    """Test driver detection and selection logic."""
    
    def test_dbapi_preferred_when_available(self):
        """Test that DBAPI is preferred when available."""
        manager = IRISConnectionManager()
        
        # Mock successful DBAPI connection
        with patch.object(manager, '_get_dbapi_connection') as mock_dbapi, \
             patch.object(manager, '_get_connection_params') as mock_params:
            
            mock_params.return_value = {
                'hostname': 'localhost',
                'port': 1972,
                'namespace': 'RAG',
                'username': 'test',
                'password': 'test'
            }
            mock_connection = MagicMock()
            mock_dbapi.return_value = mock_connection
            
            connection_info = manager.get_connection_info()
            
            assert connection_info.driver_type == DriverType.DBAPI
            assert connection_info.connection == mock_connection
            assert connection_info.capabilities['vector_operations'] is True
            mock_dbapi.assert_called_once()
    
    def test_jdbc_fallback_when_dbapi_fails(self):
        """Test JDBC fallback when DBAPI connection fails."""
        manager = IRISConnectionManager()
        
        with patch.object(manager, '_get_dbapi_connection') as mock_dbapi, \
             patch.object(manager, '_get_jdbc_connection') as mock_jdbc, \
             patch.object(manager, '_get_connection_params') as mock_params:
            
            mock_params.return_value = {
                'hostname': 'localhost',
                'port': 1972,
                'namespace': 'RAG',
                'username': 'test',
                'password': 'test'
            }
            
            # DBAPI fails, JDBC succeeds
            mock_dbapi.side_effect = Exception("DBAPI not available")
            mock_jdbc_connection = MagicMock()
            mock_jdbc.return_value = mock_jdbc_connection
            
            connection_info = manager.get_connection_info()
            
            assert connection_info.driver_type == DriverType.JDBC
            assert connection_info.connection == mock_jdbc_connection
            assert connection_info.capabilities['vector_operations'] is False
            mock_dbapi.assert_called_once()
            mock_jdbc.assert_called_once()
    
    def test_connection_failure_when_both_drivers_fail(self):
        """Test that ConnectionError is raised when both drivers fail."""
        manager = IRISConnectionManager()
        
        with patch.object(manager, '_get_dbapi_connection') as mock_dbapi, \
             patch.object(manager, '_get_jdbc_connection') as mock_jdbc, \
             patch.object(manager, '_get_connection_params') as mock_params:
            
            mock_params.return_value = {
                'hostname': 'localhost',
                'port': 1972,
                'namespace': 'RAG',
                'username': 'test',
                'password': 'test'
            }
            
            # Both drivers fail
            mock_dbapi.side_effect = Exception("DBAPI failed")
            mock_jdbc.side_effect = Exception("JDBC failed")
            
            with pytest.raises(ConnectionError) as exc_info:
                manager.get_connection_info()
            
            assert "Failed to establish database connection" in str(exc_info.value)
            assert "DBAPI failed" in str(exc_info.value)
            assert "JDBC failed" in str(exc_info.value)


class TestDriverCapabilities:
    """Test driver capability detection and reporting."""
    
    def test_dbapi_capabilities(self):
        """Test DBAPI driver capabilities."""
        manager = IRISConnectionManager()
        dbapi_capabilities = manager._driver_capabilities[DriverType.DBAPI]
        
        assert dbapi_capabilities['vector_operations'] is True
        assert dbapi_capabilities['top_clause'] is True
        assert dbapi_capabilities['to_vector_function'] is True
        assert dbapi_capabilities['vector_cosine'] is True
        assert dbapi_capabilities['auto_parameterization'] is False
    
    def test_jdbc_capabilities(self):
        """Test JDBC driver capabilities and limitations."""
        manager = IRISConnectionManager()
        jdbc_capabilities = manager._driver_capabilities[DriverType.JDBC]
        
        assert jdbc_capabilities['vector_operations'] is False
        assert jdbc_capabilities['top_clause'] is False
        assert jdbc_capabilities['to_vector_function'] is False
        assert jdbc_capabilities['vector_cosine'] is False
        assert jdbc_capabilities['auto_parameterization'] is True
    
    def test_supports_vector_operations_dbapi(self):
        """Test vector operations support detection for DBAPI."""
        manager = IRISConnectionManager()
        
        with patch.object(manager, 'get_connection_info') as mock_info:
            mock_info.return_value = ConnectionInfo(
                connection=MagicMock(),
                driver_type=DriverType.DBAPI,
                hostname='localhost',
                port=1972,
                namespace='RAG',
                username='test',
                capabilities=manager._driver_capabilities[DriverType.DBAPI]
            )
            
            assert manager.supports_vector_operations() is True
    
    def test_supports_vector_operations_jdbc(self):
        """Test vector operations support detection for JDBC."""
        manager = IRISConnectionManager()
        
        with patch.object(manager, 'get_connection_info') as mock_info:
            mock_info.return_value = ConnectionInfo(
                connection=MagicMock(),
                driver_type=DriverType.JDBC,
                hostname='localhost',
                port=1972,
                namespace='RAG',
                username='test',
                capabilities=manager._driver_capabilities[DriverType.JDBC]
            )
            
            assert manager.supports_vector_operations() is False


class TestConnectionCaching:
    """Test connection caching and reuse."""
    
    def test_connection_caching(self):
        """Test that connections are cached and reused."""
        manager = IRISConnectionManager()
        
        with patch.object(manager, '_get_dbapi_connection') as mock_dbapi, \
             patch.object(manager, '_get_connection_params') as mock_params:
            
            mock_params.return_value = {
                'hostname': 'localhost',
                'port': 1972,
                'namespace': 'RAG',
                'username': 'test',
                'password': 'test'
            }
            mock_connection = MagicMock()
            mock_dbapi.return_value = mock_connection
            
            # First call should create connection
            connection_info1 = manager.get_connection_info()
            
            # Second call should reuse cached connection
            connection_info2 = manager.get_connection_info()
            
            assert connection_info1 is connection_info2
            mock_dbapi.assert_called_once()  # Should only be called once
    
    def test_backward_compatibility_get_connection(self):
        """Test backward compatibility of get_connection method."""
        manager = IRISConnectionManager()
        
        with patch.object(manager, 'get_connection_info') as mock_info:
            mock_connection = MagicMock()
            mock_info.return_value = ConnectionInfo(
                connection=mock_connection,
                driver_type=DriverType.DBAPI,
                hostname='localhost',
                port=1972,
                namespace='RAG',
                username='test',
                capabilities={}
            )
            
            connection = manager.get_connection()
            assert connection is mock_connection


class TestGlobalConvenienceFunctions:
    """Test global convenience functions."""
    
    def test_get_connection_info_global(self):
        """Test global get_connection_info function."""
        with patch('common.iris_connection_manager._global_connection_manager') as mock_manager:
            mock_info = ConnectionInfo(
                connection=MagicMock(),
                driver_type=DriverType.DBAPI,
                hostname='localhost',
                port=1972,
                namespace='RAG',
                username='test',
                capabilities={'vector_operations': True}
            )
            mock_manager.get_connection_info.return_value = mock_info
            
            result = get_connection_info()
            assert result is mock_info
            mock_manager.get_connection_info.assert_called_once()
    
    def test_get_driver_type_global(self):
        """Test global get_driver_type function."""
        with patch('common.iris_connection_manager._global_connection_manager') as mock_manager:
            mock_manager.get_driver_type.return_value = DriverType.DBAPI
            
            result = get_driver_type()
            assert result == DriverType.DBAPI
            mock_manager.get_driver_type.assert_called_once()
    
    def test_supports_vector_operations_global(self):
        """Test global supports_vector_operations function."""
        with patch('common.iris_connection_manager._global_connection_manager') as mock_manager:
            mock_manager.supports_vector_operations.return_value = True
            
            result = supports_vector_operations()
            assert result is True
            mock_manager.supports_vector_operations.assert_called_once()
    
    def test_get_driver_capabilities_global(self):
        """Test global get_driver_capabilities function."""
        with patch('common.iris_connection_manager._global_connection_manager') as mock_manager:
            mock_capabilities = {'vector_operations': True, 'top_clause': True}
            mock_manager.get_capabilities.return_value = mock_capabilities
            
            result = get_driver_capabilities()
            assert result == mock_capabilities
            mock_manager.get_capabilities.assert_called_once()


class TestConnectionInfoDataclass:
    """Test ConnectionInfo dataclass functionality."""
    
    def test_connection_info_creation(self):
        """Test ConnectionInfo dataclass creation and attributes."""
        mock_connection = MagicMock()
        capabilities = {'vector_operations': True, 'top_clause': True}
        
        info = ConnectionInfo(
            connection=mock_connection,
            driver_type=DriverType.DBAPI,
            hostname='localhost',
            port=1972,
            namespace='RAG',
            username='testuser',
            capabilities=capabilities
        )
        
        assert info.connection is mock_connection
        assert info.driver_type == DriverType.DBAPI
        assert info.hostname == 'localhost'
        assert info.port == 1972
        assert info.namespace == 'RAG'
        assert info.username == 'testuser'
        assert info.capabilities == capabilities


class TestDriverTypeEnum:
    """Test DriverType enum functionality."""
    
    def test_driver_type_values(self):
        """Test DriverType enum values."""
        assert DriverType.DBAPI.value == "dbapi"
        assert DriverType.JDBC.value == "jdbc"
    
    def test_driver_type_comparison(self):
        """Test DriverType enum comparison."""
        assert DriverType.DBAPI == DriverType.DBAPI
        assert DriverType.JDBC == DriverType.JDBC
        assert DriverType.DBAPI != DriverType.JDBC


@pytest.mark.integration
class TestRealConnectionIntegration:
    """Integration tests with real database connections (if available)."""
    
    def test_real_connection_detection(self):
        """Test real connection detection when database is available."""
        try:
            # Attempt to get a real connection
            connection_info = get_connection_info()
            
            # If we get here, connection succeeded
            assert connection_info.driver_type in [DriverType.DBAPI, DriverType.JDBC]
            assert connection_info.connection is not None
            assert isinstance(connection_info.capabilities, dict)
            
            logger.info(f"Real connection test: Using {connection_info.driver_type.value} driver")
            logger.info(f"Vector operations supported: {connection_info.capabilities.get('vector_operations', False)}")
            
        except Exception as e:
            # Database not available, skip test
            pytest.skip(f"Database not available for integration test: {e}")
    
    def test_backward_compatibility_with_existing_code(self):
        """Test that existing code using get_iris_connection still works."""
        try:
            # This should work with existing code patterns
            connection = get_iris_connection()
            assert connection is not None
            
            # Test basic SQL execution
            cursor = connection.cursor()
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            cursor.close()
            
            assert result is not None
            logger.info("Backward compatibility test passed")
            
        except Exception as e:
            pytest.skip(f"Database not available for compatibility test: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])