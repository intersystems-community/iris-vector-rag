"""
Focused Working Unit Tests

Simple unit tests that work with the actual codebase to measure coverage improvement.
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import yaml

from iris_vector_rag.core.models import Document
from iris_vector_rag.config.manager import ConfigurationManager


class TestWorkingUnits(unittest.TestCase):
    """Working unit tests for coverage measurement."""

    def test_document_creation(self):
        """Test Document model creation."""
        doc = Document(page_content="Test content", metadata={"source": "test"})
        self.assertEqual(doc.page_content, "Test content")
        self.assertEqual(doc.metadata["source"], "test")

    def test_config_manager_get_basic(self):
        """Test ConfigurationManager basic get functionality."""
        config_manager = ConfigurationManager.__new__(ConfigurationManager)
        test_config = {
            'database': {'iris': {'host': 'localhost', 'port': 1972}},
            'embedding_model': 'test-model'
        }
        config_manager._config = test_config

        # Test simple key
        self.assertEqual(config_manager.get('embedding_model'), 'test-model')

        # Test nested key with colons
        self.assertEqual(config_manager.get('database:iris:host'), 'localhost')
        self.assertEqual(config_manager.get('database:iris:port'), 1972)

        # Test nonexistent key with default
        self.assertEqual(config_manager.get('nonexistent', 'default'), 'default')

        # Test nonexistent key without default
        self.assertIsNone(config_manager.get('nonexistent'))

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('yaml.safe_load')
    def test_config_manager_initialization(self, mock_yaml_load, mock_exists, mock_file):
        """Test ConfigurationManager initialization."""
        test_config = {'database': {'iris': {'host': 'localhost'}}}
        mock_exists.return_value = True
        mock_yaml_load.return_value = test_config

        config_manager = ConfigurationManager(config_path='/test/config.yaml')

        self.assertEqual(config_manager._config, test_config)
        mock_exists.assert_called_once_with('/test/config.yaml')
        mock_file.assert_called_once_with('/test/config.yaml', 'r')

    def test_config_manager_existing_methods(self):
        """Test ConfigurationManager existing methods."""
        config_manager = ConfigurationManager.__new__(ConfigurationManager)
        config_manager._config = {
            'database': {'iris': {'host': 'localhost'}},
            'embedding_model': 'test-model'
        }

        # Test get_database_config method
        db_config = config_manager.get_database_config()
        self.assertIsInstance(db_config, dict)

        # Test get_embedding_config method
        embedding_config = config_manager.get_embedding_config()
        self.assertIsInstance(embedding_config, (dict, str, type(None)))


if __name__ == '__main__':
    unittest.main()