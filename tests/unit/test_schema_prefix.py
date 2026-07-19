"""
Unit tests for configurable schema prefix in SchemaManager.

Feature: 079-schema-prefix
Tests that SchemaManager accepts, validates, and uses a configurable
schema prefix instead of the hardcoded 'RAG' literal.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call

from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager


def _make_mock_cfg(schema_prefix=None):
    """Build a mock ConfigurationManager suitable for SchemaManager construction."""
    mock_cfg = MagicMock(spec=ConfigurationManager)
    mock_cloud_config = MagicMock()
    mock_cloud_config.vector.vector_dimension = 384
    mock_cfg.get_cloud_config.return_value = mock_cloud_config

    def mock_get(key, default=None):
        if key == "embedding_model.name":
            return "sentence-transformers/all-MiniLM-L6-v2"
        elif key == "storage:iris":
            return {}
        return default

    mock_cfg.get.side_effect = mock_get
    mock_cfg.get_schema_prefix.return_value = schema_prefix or "RAG"
    return mock_cfg


def _make_mock_conn():
    """Build a mock ConnectionManager."""
    return MagicMock(spec=ConnectionManager)


class TestSchemaPrefixValidation:
    """Tests for schema prefix validation logic."""

    def test_valid_prefix_accepted(self):
        """A valid alphanumeric prefix is accepted without error."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg("MYAPP")

        sm = SchemaManager(mock_cm, mock_cfg, schema_prefix="MYAPP")
        assert sm.schema_prefix == "MYAPP"

    def test_valid_prefix_with_underscore_accepted(self):
        """A prefix containing underscores is accepted."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg("TEST_RAG")

        sm = SchemaManager(mock_cm, mock_cfg, schema_prefix="TEST_RAG")
        assert sm.schema_prefix == "TEST_RAG"

    def test_empty_prefix_raises_value_error(self):
        """An empty string prefix raises ValueError at construction time."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()

        with pytest.raises(ValueError, match="empty"):
            SchemaManager(mock_cm, mock_cfg, schema_prefix="")

    def test_sql_injection_prefix_raises_value_error(self):
        """A prefix with SQL injection characters raises ValueError."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()

        with pytest.raises(ValueError):
            SchemaManager(mock_cm, mock_cfg, schema_prefix="RAG'; DROP TABLE")

    def test_prefix_starting_with_digit_raises_value_error(self):
        """A prefix starting with a digit raises ValueError."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()

        with pytest.raises(ValueError):
            SchemaManager(mock_cm, mock_cfg, schema_prefix="123_INVALID")

    def test_prefix_with_dot_raises_value_error(self):
        """A prefix containing a dot raises ValueError."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()

        with pytest.raises(ValueError):
            SchemaManager(mock_cm, mock_cfg, schema_prefix="BAD.PREFIX")

    def test_prefix_with_spaces_raises_value_error(self):
        """A prefix containing spaces raises ValueError."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()

        with pytest.raises(ValueError):
            SchemaManager(mock_cm, mock_cfg, schema_prefix="BAD PREFIX")


class TestSchemaPrefixResolution:
    """Tests for schema prefix precedence resolution."""

    def test_default_prefix_is_rag(self):
        """When no prefix is supplied and no env var is set, default is 'RAG'."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()
        mock_cfg.get_schema_prefix.return_value = "RAG"

        env_without_prefix = {
            k: v for k, v in os.environ.items() if k != "IRIS_SCHEMA_PREFIX"
        }
        with patch.dict(os.environ, env_without_prefix, clear=True):
            sm = SchemaManager(mock_cm, mock_cfg)
        assert sm.schema_prefix == "RAG"

    def test_env_var_sets_prefix(self):
        """IRIS_SCHEMA_PREFIX environment variable is picked up."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()
        mock_cfg.get_schema_prefix.return_value = "ENVAPP"

        with patch.dict(os.environ, {"IRIS_SCHEMA_PREFIX": "ENVAPP"}):
            sm = SchemaManager(mock_cm, mock_cfg)
        assert sm.schema_prefix == "ENVAPP"

    def test_constructor_arg_overrides_env_var(self):
        """Constructor schema_prefix argument wins over IRIS_SCHEMA_PREFIX env var."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()
        mock_cfg.get_schema_prefix.return_value = "ENVAPP"

        with patch.dict(os.environ, {"IRIS_SCHEMA_PREFIX": "ENVAPP"}):
            sm = SchemaManager(mock_cm, mock_cfg, schema_prefix="CTOR_RAG")
        assert sm.schema_prefix == "CTOR_RAG"

    def test_constructor_arg_overrides_config_manager(self):
        """Constructor schema_prefix argument wins over config_manager value."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg()
        mock_cfg.get_schema_prefix.return_value = "CFG_PREFIX"

        env_without_prefix = {
            k: v for k, v in os.environ.items() if k != "IRIS_SCHEMA_PREFIX"
        }
        with patch.dict(os.environ, env_without_prefix, clear=True):
            sm = SchemaManager(mock_cm, mock_cfg, schema_prefix="CTOR_PREFIX")
        assert sm.schema_prefix == "CTOR_PREFIX"


class TestQnHelper:
    """Tests for the _qn() table qualification helper."""

    def test_qn_returns_qualified_name(self):
        """_qn() returns schema_prefix.TableName."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg("MYAPP")

        sm = SchemaManager(mock_cm, mock_cfg, schema_prefix="MYAPP")
        assert sm._qn("SourceDocuments") == "MYAPP.SourceDocuments"

    def test_qn_uses_current_prefix(self):
        """_qn() uses the instance's own schema_prefix."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg("TEST_IVR")

        sm = SchemaManager(mock_cm, mock_cfg, schema_prefix="TEST_IVR")
        assert sm._qn("Entities") == "TEST_IVR.Entities"
        assert sm._qn("SchemaMetadata") == "TEST_IVR.SchemaMetadata"


class TestCacheIsolation:
    """Tests that instance-level caches are isolated between SchemaManager instances."""

    def test_separate_instances_have_separate_validation_caches(self):
        """Two SchemaManager instances must not share _schema_validation_cache."""
        mock_cm = _make_mock_conn()
        mock_cfg_a = _make_mock_cfg("A")
        mock_cfg_b = _make_mock_cfg("B")

        sm1 = SchemaManager(mock_cm, mock_cfg_a, schema_prefix="A")
        sm2 = SchemaManager(mock_cm, mock_cfg_b, schema_prefix="B")

        # Populate sm1's cache
        sm1._schema_validation_cache["A:SourceDocuments:basic"] = True

        # sm2 should NOT see sm1's cache entry
        assert "A:SourceDocuments:basic" not in sm2._schema_validation_cache

    def test_separate_instances_have_separate_tables_validated(self):
        """Two SchemaManager instances must not share _tables_validated."""
        mock_cm = _make_mock_conn()
        mock_cfg_a = _make_mock_cfg("A")
        mock_cfg_b = _make_mock_cfg("B")

        sm1 = SchemaManager(mock_cm, mock_cfg_a, schema_prefix="A")
        sm2 = SchemaManager(mock_cm, mock_cfg_b, schema_prefix="B")

        # Populate sm1's set
        sm1._tables_validated.add("SourceDocuments")

        # sm2 should NOT see sm1's entry
        assert "SourceDocuments" not in sm2._tables_validated

    def test_cache_modification_does_not_affect_class_level(self):
        """Instance cache is not the class-level dict (if class-level still exists)."""
        mock_cm = _make_mock_conn()
        mock_cfg = _make_mock_cfg("TEST")

        sm = SchemaManager(mock_cm, mock_cfg, schema_prefix="TEST")

        # The instance cache should be its own dict, not the class-level one
        assert sm._schema_validation_cache is not SchemaManager.__dict__.get(
            "_schema_validation_cache"
        )


class TestConfigurationManagerSchemaPrefix:
    """Tests for ConfigurationManager.get_schema_prefix()."""

    def test_get_schema_prefix_returns_default(self):
        """Without env var or config, get_schema_prefix() returns 'RAG'."""
        cfg = ConfigurationManager()
        env_without_prefix = {
            k: v for k, v in os.environ.items() if k != "IRIS_SCHEMA_PREFIX"
        }
        with patch.dict(os.environ, env_without_prefix, clear=True):
            result = cfg.get_schema_prefix()
        assert result == "RAG"

    def test_get_schema_prefix_reads_env_var(self):
        """get_schema_prefix() returns IRIS_SCHEMA_PREFIX env var value."""
        cfg = ConfigurationManager()
        with patch.dict(os.environ, {"IRIS_SCHEMA_PREFIX": "TEST_IVR"}):
            result = cfg.get_schema_prefix()
        assert result == "TEST_IVR"
