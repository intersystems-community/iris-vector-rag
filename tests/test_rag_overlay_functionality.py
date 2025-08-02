#!/usr/bin/env python3
"""
TDD Tests for RAG Overlay Functionality

Tests the RAG overlay system that allows integrating existing database tables
with RAG capabilities without modifying original data.
"""

import pytest
import os
import sys
import tempfile
import yaml
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.rag_overlay_installer import RAGOverlayInstaller


class TestRAGOverlayInstaller:
    """Test the RAG overlay installation system."""

    @pytest.fixture
    def overlay_config(self) -> Dict[str, Any]:
        """Create test overlay configuration."""
        return {
            "source_tables": [
                {
                    "name": "CustomerDocs.Documents",
                    "id_field": "document_id",
                    "title_field": "title", 
                    "content_field": "content",
                    "metadata_fields": ["author", "created_date", "category"],
                    "enabled": True
                },
                {
                    "name": "KnowledgeBase.Articles",
                    "id_field": "article_id",
                    "title_field": "article_title",
                    "content_field": "full_text",
                    "metadata_fields": ["topic", "last_updated"],
                    "enabled": True
                }
            ],
            "rag_schema": "RAG",
            "view_prefix": "RAG_Overlay_",
            "embedding_table": "RAG.OverlayEmbeddings",
            "ifind_table": "RAG.OverlayIFindIndex"
        }

    @pytest.fixture
    def config_file(self, overlay_config: Dict[str, Any]) -> str:
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(overlay_config, f)
            return f.name

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn, mock_cursor

    @pytest.fixture  
    def installer(self, config_file: str, mock_connection):
        """Create RAGOverlayInstaller with mocked connection."""
        mock_conn, mock_cursor = mock_connection
        
        with patch('scripts.rag_overlay_installer.get_iris_connection', return_value=mock_conn):
            installer = RAGOverlayInstaller(config_file)
            return installer, mock_cursor

    def test_load_overlay_config_from_file(self, config_file: str):
        """Test loading overlay configuration from YAML file."""
        # Arrange & Act
        with patch('scripts.rag_overlay_installer.get_iris_connection'):
            installer = RAGOverlayInstaller(config_file)
        
        # Assert
        assert len(installer.config["source_tables"]) == 2
        assert installer.config["source_tables"][0]["name"] == "CustomerDocs.Documents"
        assert installer.config["source_tables"][1]["name"] == "KnowledgeBase.Articles"
        assert installer.config["rag_schema"] == "RAG"
        assert installer.config["view_prefix"] == "RAG_Overlay_"

    def test_load_default_config_when_file_missing(self):
        """Test that default config is used when file doesn't exist."""
        # Arrange & Act
        with patch('scripts.rag_overlay_installer.get_iris_connection'):
            installer = RAGOverlayInstaller("nonexistent_file.yaml")
        
        # Assert
        assert len(installer.config["source_tables"]) == 1
        assert installer.config["source_tables"][0]["name"] == "CustomerDocs.Documents"
        assert installer.config["rag_schema"] == "RAG"

    def test_discover_existing_tables(self, installer):
        """Test discovery of existing tables with text content."""
        # Arrange
        installer_obj, mock_cursor = installer
        mock_cursor.fetchall.return_value = [
            ("MySchema", "Documents", "content", "longvarchar", 5000),
            ("MySchema", "Documents", "title", "varchar", 255),
            ("AnotherSchema", "Articles", "text", "longvarchar", 8000)
        ]
        
        # Act
        discovered = installer_obj.discover_existing_tables()
        
        # Assert
        assert len(discovered) == 2
        assert discovered[0]["schema"] == "MySchema"
        assert discovered[0]["table"] == "Documents"
        assert len(discovered[0]["columns"]) == 2
        mock_cursor.execute.assert_called_once()

    def test_create_overlay_views(self, installer):
        """Test creation of overlay views from source tables."""
        # Arrange
        installer_obj, mock_cursor = installer
        
        # Act
        result = installer_obj.create_overlay_views()
        
        # Assert
        assert result is True
        # Should create 2 views (one for each source table)
        assert mock_cursor.execute.call_count >= 2
        
        # Check that view SQL contains expected mappings
        calls = mock_cursor.execute.call_args_list
        view_sql_calls = [call for call in calls if 'CREATE VIEW' in str(call)]
        assert len(view_sql_calls) == 2

    def test_create_overlay_embedding_table(self, installer):
        """Test creation of overlay embedding table."""
        # Arrange
        installer_obj, mock_cursor = installer
        
        # Act
        result = installer_obj.create_overlay_embedding_table()
        
        # Assert
        assert result is True
        # Should execute CREATE TABLE and CREATE INDEX
        assert mock_cursor.execute.call_count >= 2
        
        # Check that embedding table SQL is correct
        calls = mock_cursor.execute.call_args_list
        create_calls = [call for call in calls if 'CREATE TABLE' in str(call)]
        assert len(create_calls) >= 1

    def test_create_overlay_ifind_table(self, installer):
        """Test creation of overlay IFind table."""
        # Arrange
        installer_obj, mock_cursor = installer
        
        # Act
        result = installer_obj.create_overlay_ifind_table()
        
        # Assert
        assert result is True
        # Should execute CREATE TABLE and try CREATE FULLTEXT INDEX
        assert mock_cursor.execute.call_count >= 1

    def test_create_unified_rag_view(self, installer):
        """Test creation of unified RAG view combining all overlay sources."""
        # Arrange
        installer_obj, mock_cursor = installer
        
        # Act
        result = installer_obj.create_unified_rag_view()
        
        # Assert
        assert result is True
        mock_cursor.execute.assert_called()

    def test_build_metadata_json_with_fields(self, installer):
        """Test building JSON metadata from specified fields."""
        # Arrange
        installer_obj, _ = installer
        metadata_fields = ["author", "created_date", "category"]
        
        # Act
        result = installer_obj._build_metadata_json(metadata_fields)
        
        # Assert
        assert isinstance(result, str)
        assert "author" in result
        assert "created_date" in result
        assert "category" in result

    def test_build_metadata_json_empty_fields(self, installer):
        """Test building JSON metadata with no fields."""
        # Arrange
        installer_obj, _ = installer
        
        # Act
        result = installer_obj._build_metadata_json([])
        
        # Assert
        assert result == ""

    def test_field_mapping_in_view_creation(self, installer):
        """Test that field mappings are correctly applied in view creation."""
        # Arrange
        installer_obj, mock_cursor = installer
        
        # Act
        installer_obj.create_overlay_views()
        
        # Assert
        calls = mock_cursor.execute.call_args_list
        view_calls = [call for call in calls if 'CREATE VIEW' in str(call)]
        
        # Check first view uses document_id -> doc_id mapping
        first_view_sql = str(view_calls[0])
        assert "document_id as doc_id" in first_view_sql
        assert "title as title" in first_view_sql
        assert "content as text_content" in first_view_sql
        
        # Check second view uses article_id -> doc_id mapping  
        second_view_sql = str(view_calls[1])
        assert "article_id as doc_id" in second_view_sql
        assert "article_title as title" in second_view_sql
        assert "full_text as text_content" in second_view_sql

    def test_overlay_preserves_original_data(self, installer):
        """Test that overlay system doesn't modify original tables."""
        # Arrange
        installer_obj, mock_cursor = installer
        
        # Act
        installer_obj.create_overlay_views()
        installer_obj.create_overlay_embedding_table()
        installer_obj.create_overlay_ifind_table()
        
        # Assert
        calls = mock_cursor.execute.call_args_list
        sql_statements = [str(call) for call in calls]
        
        # Should only create VIEWs and new TABLEs, never ALTER existing tables
        alter_statements = [sql for sql in sql_statements if 'ALTER TABLE' in sql.upper()]
        assert len(alter_statements) == 0
        
        # Should create views, not modify source tables
        view_statements = [sql for sql in sql_statements if 'CREATE VIEW' in sql.upper()]
        assert len(view_statements) > 0

    @pytest.mark.integration
    def test_full_overlay_installation_workflow(self, config_file: str):
        """Integration test: Full overlay installation workflow."""
        # This would be a full integration test that requires actual database
        # For now, we test the workflow with mocks
        
        with patch('scripts.rag_overlay_installer.get_iris_connection') as mock_get_conn:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn
            
            # Arrange
            installer = RAGOverlayInstaller(config_file)
            
            # Act - Full workflow
            installer.create_overlay_views()
            installer.create_overlay_embedding_table()
            installer.create_overlay_ifind_table() 
            installer.create_unified_rag_view()
            
            # Assert - All steps executed
            assert mock_cursor.execute.call_count >= 4
            mock_conn.commit.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])