"""
Integration tests for driver-aware vector SQL utilities.

This module tests the vector SQL utilities with both DBAPI and JDBC drivers
to ensure proper driver-specific SQL generation and execution.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

from common.vector_sql_utils import (
    get_driver_aware_vector_search_sql,
    execute_driver_aware_vector_search,
    format_vector_search_sql,
    execute_vector_search,
    validate_vector_string,
    validate_top_k
)
from common.db_vector_utils import insert_vector
from common.iris_connection_manager import DriverType, get_driver_type, get_driver_capabilities

logger = logging.getLogger(__name__)


class TestDriverAwareVectorSQL:
    """Test driver-aware vector SQL generation and execution."""
    
    def test_get_driver_aware_vector_search_sql_dbapi(self):
        """Test SQL generation for DBAPI driver."""
        with patch('common.vector_sql_utils.get_driver_type', return_value=DriverType.DBAPI), \
             patch('common.vector_sql_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': True,
                'supports_parameterized_queries': True,
                'has_auto_parameterization_bug': False
            }
            
            sql, uses_parameters = get_driver_aware_vector_search_sql(
                table_name="SourceDocuments",
                vector_column="embedding",
                embedding_dim=384,
                top_k=5,
                id_column="doc_id",
                content_column="text_content"
            )
            
            assert uses_parameters is True
            assert "?" in sql  # DBAPI uses ? parameter markers
            assert "TO_VECTOR(?, FLOAT, 384)" in sql
            assert "SELECT TOP 5" in sql
            
    def test_get_driver_aware_vector_search_sql_jdbc(self):
        """Test SQL generation for JDBC driver."""
        with patch('common.vector_sql_utils.get_driver_type', return_value=DriverType.JDBC), \
             patch('common.vector_sql_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': False,
                'supports_parameterized_queries': False,
                'has_auto_parameterization_bug': True
            }
            
            sql, uses_parameters = get_driver_aware_vector_search_sql(
                table_name="SourceDocuments",
                vector_column="embedding",
                embedding_dim=384,
                top_k=5,
                id_column="doc_id",
                content_column="text_content"
            )
            
            assert uses_parameters is False
            assert "?" not in sql  # JDBC fallback uses literal values
            assert "{vector_string}" in sql  # Template placeholder
            assert "SELECT TOP 5" in sql
            
    def test_execute_driver_aware_vector_search_dbapi(self):
        """Test vector search execution with DBAPI driver."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("doc1", "content1", 0.95)]
        
        with patch('common.vector_sql_utils.get_driver_type', return_value=DriverType.DBAPI), \
             patch('common.vector_sql_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': True,
                'supports_parameterized_queries': True,
                'has_auto_parameterization_bug': False
            }
            
            results = execute_driver_aware_vector_search(
                cursor=mock_cursor,
                table_name="SourceDocuments",
                vector_column="embedding",
                vector_string="[0.1,0.2,0.3]",
                embedding_dim=384,
                top_k=5
            )
            
            assert len(results) == 1
            assert results[0][0] == "doc1"
            mock_cursor.execute.assert_called_once()
            
    def test_execute_driver_aware_vector_search_jdbc(self):
        """Test vector search execution with JDBC driver."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("doc1", "content1", 0.95)]
        
        with patch('common.vector_sql_utils.get_driver_type', return_value=DriverType.JDBC), \
             patch('common.vector_sql_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': False,
                'supports_parameterized_queries': False,
                'has_auto_parameterization_bug': True
            }
            
            results = execute_driver_aware_vector_search(
                cursor=mock_cursor,
                table_name="SourceDocuments",
                vector_column="embedding",
                vector_string="[0.1,0.2,0.3]",
                embedding_dim=384,
                top_k=5
            )
            
            assert len(results) == 1
            assert results[0][0] == "doc1"
            mock_cursor.execute.assert_called_once()
            
    def test_format_vector_search_sql_backward_compatibility(self):
        """Test that format_vector_search_sql maintains backward compatibility."""
        with patch('common.vector_sql_utils.get_driver_type', return_value=DriverType.DBAPI), \
             patch('common.vector_sql_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': True,
                'supports_parameterized_queries': True,
                'has_auto_parameterization_bug': False
            }
            
            sql = format_vector_search_sql(
                table_name="SourceDocuments",
                vector_column="embedding",
                vector_string="[0.1,0.2,0.3]",
                embedding_dim=384,
                top_k=5
            )
            
            # Should work without breaking existing code
            assert "SourceDocuments" in sql
            assert "embedding" in sql
            assert "VECTOR_COSINE" in sql
            
    def test_execute_vector_search_jdbc_fallback(self):
        """Test execute_vector_search JDBC fallback for auto-parameterization bug."""
        mock_cursor = Mock()

        # First call raises auto-parameterization error
        mock_cursor.execute.side_effect = [
            Exception(":%qpar(1) not found"),
            None  # Second call succeeds
        ]
        mock_cursor.fetchall.return_value = [("doc1", "content1", 0.95)]

        with patch('common.vector_sql_utils.get_driver_type', return_value=DriverType.JDBC), \
             patch('common.vector_sql_utils.get_driver_capabilities') as mock_caps:

            mock_caps.return_value = {
                'supports_vector_operations': False,
                'supports_parameterized_queries': False,
                'has_auto_parameterization_bug': True
            }

            # Test the backward compatibility function with correct signature
            results = execute_vector_search(
                cursor=mock_cursor,
                sql="SELECT TOP 5 doc_id FROM SourceDocuments"
            )

            assert len(results) == 1
            assert results[0][0] == "doc1"
            # Should have been called twice due to retry logic for JDBC auto-parameterization bug
            assert mock_cursor.execute.call_count == 2
            
    def test_validation_functions(self):
        """Test vector validation utility functions."""
        # Test valid vector string
        assert validate_vector_string("[0.1,0.2,0.3]") is True
        
        # Test invalid vector string
        assert validate_vector_string("invalid") is False
        
        # Test valid top_k
        assert validate_top_k(5) is True
        
        # Test invalid top_k
        assert validate_top_k(0) is False
        assert validate_top_k(-1) is False


class TestDriverAwareVectorInsertion:
    """Test driver-aware vector insertion utilities."""
    
    def test_insert_vector_dbapi(self):
        """Test vector insertion with DBAPI driver."""
        mock_cursor = Mock()
        
        with patch('common.db_vector_utils.get_driver_type', return_value=DriverType.DBAPI), \
             patch('common.db_vector_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': True,
                'supports_parameterized_queries': True,
                'has_auto_parameterization_bug': False
            }
            
            result = insert_vector(
                cursor=mock_cursor,
                table_name="SourceDocuments",
                vector_column_name="embedding",
                vector_data=[0.1, 0.2, 0.3],
                target_dimension=384,
                key_columns={"doc_id": "test_doc"},
                additional_data={"text_content": "test content"}
            )
            
            assert result is True
            mock_cursor.execute.assert_called_once()
            
            # Check that DBAPI-style SQL was used
            call_args = mock_cursor.execute.call_args
            sql = call_args[0][0]
            assert "?" in sql  # DBAPI uses ? parameter markers
            
    def test_insert_vector_jdbc(self):
        """Test vector insertion with JDBC driver."""
        mock_cursor = Mock()
        
        with patch('common.db_vector_utils.get_driver_type', return_value=DriverType.JDBC), \
             patch('common.db_vector_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': False,
                'supports_parameterized_queries': False,
                'has_auto_parameterization_bug': True
            }
            
            vector_data = [0.1, 0.2, 0.3]
            
            result = insert_vector(
                cursor=mock_cursor,
                table_name="SourceDocuments",
                vector_column_name="embedding",
                vector_data=vector_data,
                target_dimension=384,
                key_columns={"doc_id": "test_doc"},
                additional_data={"text_content": "test content"}
            )
            
            assert result is True
            mock_cursor.execute.assert_called_once()
            
            # Check that JDBC-style SQL was used for vector part
            call_args = mock_cursor.execute.call_args
            sql = call_args[0][0]
            # JDBC still uses ? for regular parameters, but vector is interpolated
            assert "TO_VECTOR('[0.1,0.2,0.3," in sql  # Vector is interpolated
            assert "?" in sql  # But other parameters still use ?
            
    def test_insert_vector_update_fallback_dbapi(self):
        """Test vector insertion UPDATE fallback with DBAPI driver."""
        mock_cursor = Mock()
        
        # First call (INSERT) fails with unique constraint
        # Second call (UPDATE) succeeds
        mock_cursor.execute.side_effect = [
            Exception("UNIQUE constraint failed"),
            None
        ]
        
        with patch('common.db_vector_utils.get_driver_type', return_value=DriverType.DBAPI), \
             patch('common.db_vector_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': True,
                'supports_parameterized_queries': True,
                'has_auto_parameterization_bug': False
            }
            
            result = insert_vector(
                cursor=mock_cursor,
                table_name="SourceDocuments",
                vector_column_name="embedding",
                vector_data=[0.1, 0.2, 0.3],
                target_dimension=384,
                key_columns={"doc_id": "test_doc"},
                additional_data={"text_content": "test content"}
            )
            
            assert result is True
            # Should have been called twice (INSERT failed, UPDATE succeeded)
            assert mock_cursor.execute.call_count == 2
            
    def test_insert_vector_update_fallback_jdbc(self):
        """Test vector insertion UPDATE fallback with JDBC driver."""
        mock_cursor = Mock()
        
        # First call (INSERT) fails with unique constraint
        # Second call (UPDATE) succeeds
        mock_cursor.execute.side_effect = [
            Exception("UNIQUE constraint failed"),
            None
        ]
        
        with patch('common.db_vector_utils.get_driver_type', return_value=DriverType.JDBC), \
             patch('common.db_vector_utils.get_driver_capabilities') as mock_caps:
            
            mock_caps.return_value = {
                'supports_vector_operations': False,
                'supports_parameterized_queries': False,
                'has_auto_parameterization_bug': True
            }
            
            result = insert_vector(
                cursor=mock_cursor,
                table_name="SourceDocuments",
                vector_column_name="embedding",
                vector_data=[0.1, 0.2, 0.3],
                target_dimension=384,
                key_columns={"doc_id": "test_doc"},
                additional_data={"text_content": "test content"}
            )
            
            assert result is True
            # Should have been called twice (INSERT failed, UPDATE succeeded)
            assert mock_cursor.execute.call_count == 2


class TestDriverCompatibilityIntegration:
    """Test integration between driver detection and vector utilities."""
    
    def test_driver_detection_consistency(self):
        """Test that driver detection is consistent across modules."""
        with patch('common.iris_connection_manager.get_driver_type', return_value=DriverType.DBAPI) as mock_driver:
            
            # Test that both modules use the same driver detection
            from common.vector_sql_utils import get_driver_type as vector_get_driver_type
            from common.db_vector_utils import get_driver_type as db_get_driver_type
            
            vector_driver = vector_get_driver_type()
            db_driver = db_get_driver_type()
            
            assert vector_driver == db_driver == DriverType.DBAPI
            
    def test_capability_consistency(self):
        """Test that capability reporting is consistent across modules."""
        # Use the actual capability structure from the connection manager
        with patch('common.iris_connection_manager.get_driver_capabilities') as mock_caps:
            
            # Mock the actual capability structure returned by the connection manager
            mock_caps.return_value = {
                'vector_operations': True,
                'top_clause': True,
                'to_vector_function': True,
                'vector_cosine': True,
                'auto_parameterization': False
            }
            
            # Test that both modules use the same capability detection
            from common.vector_sql_utils import get_driver_capabilities as vector_get_caps
            from common.db_vector_utils import get_driver_capabilities as db_get_caps
            
            vector_caps = vector_get_caps()
            db_caps = db_get_caps()
            
            assert vector_caps == db_caps
            assert vector_caps.get('vector_operations') is True
            assert db_caps.get('auto_parameterization') is False