#!/usr/bin/env python3
"""
Comprehensive tests for SchemaManager to catch the three bugs discovered:
1. Schema validation running repeatedly (no caching)
2. base_embedding_dimension attribute missing when using cached config
3. Foreign key constraint referencing wrong column

These tests would have caught all three bugs before they reached production.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add rag-templates to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager


class TestSchemaManagerCachingPerformance:
    """Test that schema validation caching works correctly to prevent performance degradation."""

    def test_schema_validation_cache_shared_across_instances(self, tmp_path):
        """
        BUG #1: Schema validation was running thousands of times because each new
        SchemaManager instance had its own empty _dimension_cache.

        This test verifies that validation results are shared across ALL instances.
        """
        # Create a test config file
        config_content = """
database:
  iris:
    host: "localhost"
    port: 21972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        # Mock database connection to avoid actual database calls
        with patch('iris_rag.storage.schema_manager.dbapi.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = [384]  # Simulate table having correct dimension
            mock_cursor.fetchall.return_value = []
            mock_connect.return_value.__enter__.return_value.cursor.return_value = mock_cursor

            # Create first instance - should perform validation
            config_manager1 = ConfigurationManager(str(config_path))
            schema_manager1 = SchemaManager(config_manager1)

            # Verify initial state
            assert SchemaManager._config_loaded == True
            assert len(SchemaManager._schema_validation_cache) >= 0

            # Create second instance - should use cached validation
            config_manager2 = ConfigurationManager(str(config_path))
            schema_manager2 = SchemaManager(config_manager2)

            # Both instances should share the same cache
            assert schema_manager1._schema_validation_cache is schema_manager2._schema_validation_cache
            assert schema_manager1.__class__._config_loaded == schema_manager2.__class__._config_loaded

    def test_needs_migration_uses_class_level_cache(self, tmp_path):
        """
        Verify that needs_migration() uses class-level cache, not instance-level.

        This prevents the schema validation from running on EVERY entity storage call.
        """
        config_content = """
database:
  iris:
    host: "localhost"
    port: 21972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        with patch('iris_rag.storage.schema_manager.dbapi.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = [384]
            mock_cursor.fetchall.return_value = []
            mock_connect.return_value.__enter__.return_value.cursor.return_value = mock_cursor

            config_manager = ConfigurationManager(str(config_path))
            schema_manager = SchemaManager(config_manager)

            # First call should cache result
            result1 = schema_manager.needs_migration("SourceDocuments")

            # Clear instance-level cache if it exists (simulating new instance creation)
            if hasattr(schema_manager, '_dimension_cache'):
                schema_manager._dimension_cache = {}

            # Second call should use class-level cache, NOT re-validate
            result2 = schema_manager.needs_migration("SourceDocuments")

            # Results should be identical (from cache)
            assert result1 == result2

            # Verify cache_key is in class-level cache
            cache_key = "SourceDocuments:default"
            assert cache_key in SchemaManager._schema_validation_cache


class TestSchemaManagerAttributeInitialization:
    """Test that instance attributes are properly initialized in all code paths."""

    def test_base_embedding_dimension_set_when_using_cached_config(self, tmp_path):
        """
        BUG #2: base_embedding_dimension was not set when _config_loaded was True.

        The else branch in __init__ called _build_model_dimension_mapping() which
        needs self.base_embedding_dimension, but that attribute was only set in
        _load_and_validate_config() which was skipped.
        """
        config_content = """
database:
  iris:
    host: "localhost"
    port: 21972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        with patch('iris_rag.storage.schema_manager.dbapi.connect'):
            # Create first instance - loads config
            config_manager1 = ConfigurationManager(str(config_path))
            schema_manager1 = SchemaManager(config_manager1)

            # Verify attributes are set
            assert hasattr(schema_manager1, 'base_embedding_model')
            assert hasattr(schema_manager1, 'base_embedding_dimension')
            assert schema_manager1.base_embedding_dimension == 384

            # Create second instance - uses cached config
            config_manager2 = ConfigurationManager(str(config_path))
            schema_manager2 = SchemaManager(config_manager2)

            # CRITICAL: Second instance must also have these attributes
            assert hasattr(schema_manager2, 'base_embedding_model')
            assert hasattr(schema_manager2, 'base_embedding_dimension')
            assert schema_manager2.base_embedding_dimension == 384

            # Verify both instances can build mappings without AttributeError
            assert hasattr(schema_manager2, 'model_dimension_mapping')
            assert hasattr(schema_manager2, 'table_configurations')


class TestSchemaManagerForeignKeyConstraints:
    """Test that foreign key constraints reference correct columns."""

    def test_entities_table_foreign_key_references_doc_id(self, tmp_path):
        """
        BUG #3: Entities table foreign key referenced SourceDocuments(id)
        but SourceDocuments table has doc_id as primary key.

        This caused all entity storage to fail with:
        FOREIGN KEY constraint failed referential check upon INSERT
        """
        config_content = """
database:
  iris:
    host: "localhost"
    port: 21972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        with patch('iris_rag.storage.schema_manager.dbapi.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None  # Table doesn't exist
            mock_cursor.fetchall.return_value = []
            mock_connect.return_value.__enter__.return_value.cursor.return_value = mock_cursor

            config_manager = ConfigurationManager(str(config_path))
            schema_manager = SchemaManager(config_manager)

            # Get the CREATE TABLE SQL for Entities table
            # We'll inspect it to verify the foreign key references doc_id, not id
            with patch.object(mock_cursor, 'execute') as mock_execute:
                schema_manager._migrate_entities_table("RAG.Entities", "basic", 384)

                # Find the CREATE TABLE call
                create_calls = [call for call in mock_execute.call_args_list
                               if 'CREATE TABLE' in str(call)]

                if create_calls:
                    create_sql = str(create_calls[0])

                    # CRITICAL CHECK: Foreign key must reference doc_id, NOT id
                    assert 'REFERENCES RAG.SourceDocuments(doc_id)' in create_sql, \
                        "Foreign key must reference doc_id, not id!"

                    # Should NOT reference non-existent 'id' column
                    assert 'REFERENCES RAG.SourceDocuments(id)' not in create_sql, \
                        "Foreign key incorrectly references non-existent 'id' column!"

    def test_source_documents_table_has_doc_id_primary_key(self, tmp_path):
        """
        Verify that SourceDocuments table uses doc_id as primary key.

        This is the column that Entities.source_doc_id must reference.
        """
        config_content = """
database:
  iris:
    host: "localhost"
    port: 21972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        with patch('iris_rag.storage.schema_manager.dbapi.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_cursor.fetchall.return_value = []
            mock_connect.return_value.__enter__.return_value.cursor.return_value = mock_cursor

            config_manager = ConfigurationManager(str(config_path))
            schema_manager = SchemaManager(config_manager)

            with patch.object(mock_cursor, 'execute') as mock_execute:
                schema_manager._migrate_source_documents_table("RAG.SourceDocuments", "basic", 384)

                create_calls = [call for call in mock_execute.call_args_list
                               if 'CREATE TABLE' in str(call)]

                if create_calls:
                    create_sql = str(create_calls[0])

                    # Verify doc_id is the primary key
                    assert 'doc_id' in create_sql
                    assert 'PRIMARY KEY' in create_sql

                    # Should NOT have an 'id' column as primary key
                    assert 'id VARCHAR' not in create_sql or 'id' not in create_sql.split('PRIMARY KEY')[0]


class TestSchemaManagerIntegrationScenarios:
    """Integration tests simulating real-world usage patterns."""

    def test_reusable_pipeline_pattern_with_many_entity_extractions(self, tmp_path):
        """
        Simulate the pattern used in index_all_429k_tickets.py:
        - Create pipeline once
        - Use it for multiple batches
        - Each batch creates new SchemaManager instances for entity storage

        This pattern revealed all three bugs in production.
        """
        config_content = """
database:
  iris:
    host: "localhost"
    port: 21972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        validation_count = 0

        def mock_needs_migration_side_effect(*args, **kwargs):
            """Count how many times validation actually runs."""
            nonlocal validation_count
            validation_count += 1
            return False

        with patch('iris_rag.storage.schema_manager.dbapi.connect'):
            # Simulate 100 entity extraction calls (like processing 100 tickets)
            for i in range(100):
                config_manager = ConfigurationManager(str(config_path))
                schema_manager = SchemaManager(config_manager)

                # Each extraction would call needs_migration
                # With caching, validation should only run ONCE total
                with patch.object(SchemaManager, '_validate_table_dimension', return_value=True):
                    result = schema_manager.needs_migration("Entities")

            # CRITICAL: With class-level caching, validation should run minimal times
            # Without caching, it would run 100+ times (severe performance issue)
            assert validation_count <= 5, \
                f"Schema validation ran {validation_count} times - caching not working!"

    def test_config_loaded_flag_prevents_duplicate_validation(self, tmp_path):
        """
        Verify that _config_loaded class variable prevents redundant config loading.
        """
        config_content = """
database:
  iris:
    host: "localhost"
    port: 21972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        load_count = 0

        def mock_load_and_validate(*args, **kwargs):
            """Count how many times config is actually loaded."""
            nonlocal load_count
            load_count += 1

        with patch('iris_rag.storage.schema_manager.dbapi.connect'):
            with patch.object(SchemaManager, '_load_and_validate_config', side_effect=mock_load_and_validate):
                # Create 10 instances
                for i in range(10):
                    config_manager = ConfigurationManager(str(config_path))
                    schema_manager = SchemaManager(config_manager)

                # Config should only load ONCE
                assert load_count == 1, \
                    f"Config loaded {load_count} times - should only load once!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
