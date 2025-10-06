"""
Comprehensive tests for IRISStorage enterprise storage implementation.

This test suite covers the IRISStorage class which provides IRIS-specific storage
operations for RAG templates, including document insertion, vector search, and metadata queries.
"""

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.storage.enterprise_storage import (
    IRISStorage,
    _convert_clob_to_string,
    _process_db_row_for_document,
)


class TestCLOBUtilities(unittest.TestCase):
    """Test CLOB conversion utility functions."""

    def test_convert_clob_to_string_with_stream(self):
        """Test converting CLOB/IRISInputStream objects to strings."""
        # Mock stream object
        mock_stream = Mock()
        mock_stream.read.return_value = b"test content"

        result = _convert_clob_to_string(mock_stream)
        self.assertEqual(result, "test content")
        mock_stream.read.assert_called_once()

    def test_convert_clob_to_string_with_string(self):
        """Test converting string values."""
        result = _convert_clob_to_string("test string")
        self.assertEqual(result, "test string")

    def test_convert_clob_to_string_with_none(self):
        """Test converting None values."""
        result = _convert_clob_to_string(None)
        self.assertEqual(result, "")

    def test_convert_clob_to_string_with_error(self):
        """Test error handling in CLOB conversion."""
        mock_stream = Mock()
        mock_stream.read.side_effect = Exception("Read error")

        result = _convert_clob_to_string(mock_stream)
        self.assertEqual(result, "[Error Reading Stream]")

    def test_process_db_row_for_document(self):
        """Test processing database row for document conversion."""
        # Mock CLOB field
        mock_clob = Mock()
        mock_clob.read.return_value = b"clob content"

        row_dict = {
            "text_field": "normal text",
            "clob_field": mock_clob,
            "null_field": None,
            "number_field": 123,
        }

        result = _process_db_row_for_document(row_dict)

        self.assertEqual(result["text_field"], "normal text")
        self.assertEqual(result["clob_field"], "clob content")
        self.assertEqual(result["null_field"], "")
        self.assertEqual(result["number_field"], "123")


class TestIRISStorage(unittest.TestCase):
    """Test the main IRISStorage class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)
        self.mock_config_manager = Mock(spec=ConfigurationManager)

        # Mock configuration
        self.mock_config_manager.get.return_value = {
            "table_name": "RAG.SourceDocuments",
            "vector_dimension": 384,
        }

        # Mock database connection and cursor
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection_manager.get_connection.return_value = self.mock_connection

        self.storage = IRISStorage(
            self.mock_connection_manager, self.mock_config_manager
        )

    def test_initialization(self):
        """Test IRISStorage initialization."""
        self.assertEqual(self.storage.connection_manager, self.mock_connection_manager)
        self.assertEqual(self.storage.config_manager, self.mock_config_manager)
        self.assertEqual(self.storage.table_name, "RAG.SourceDocuments")
        self.assertEqual(self.storage.vector_dimension, 384)

    def test_get_connection(self):
        """Test getting database connection."""
        connection = self.storage._get_connection()
        self.assertEqual(connection, self.mock_connection)
        self.mock_connection_manager.get_connection.assert_called_with("iris")

    @patch("iris_rag.storage.enterprise_storage.logger")
    def test_initialize_schema_success(self, mock_logger):
        """Test successful schema initialization."""
        # Mock successful table creation
        self.mock_cursor.execute.side_effect = [
            None,  # DROP TABLE succeeds
            None,  # CREATE TABLE succeeds
        ]

        with patch("common.iris_sql_utils.IRISSQLUtils") as mock_utils_class:
            mock_utils = Mock()
            mock_utils_class.return_value = mock_utils
            mock_utils.optimize_vector_table.return_value = True

            # Mock vector config
            vector_config = {"M": 16, "efConstruction": 200, "Distance": "Cosine"}
            self.mock_config_manager.get_vector_index_config.return_value = (
                vector_config
            )

            self.storage.initialize_schema()

            # Verify table creation attempts
            self.assertTrue(self.mock_cursor.execute.called)
            self.mock_connection.commit.assert_called_once()
            mock_utils.optimize_vector_table.assert_called_once()

    @patch("iris_rag.storage.enterprise_storage.logger")
    def test_initialize_schema_fallback(self, mock_logger):
        """Test schema initialization with fallback table name."""
        # Mock first table creation failure, second succeeds
        self.mock_cursor.execute.side_effect = [
            Exception("Schema error"),  # First table fails
            None,  # DROP TABLE for fallback
            None,  # CREATE TABLE for fallback succeeds
        ]

        with patch("common.iris_sql_utils.IRISSQLUtils") as mock_utils_class:
            mock_utils = Mock()
            mock_utils_class.return_value = mock_utils
            mock_utils.optimize_vector_table.return_value = True

            vector_config = {"M": 16, "efConstruction": 200, "Distance": "Cosine"}
            self.mock_config_manager.get_vector_index_config.return_value = (
                vector_config
            )

            self.storage.initialize_schema()

            # Should have tried both table names
            self.assertEqual(self.storage.table_name, "SourceDocuments")

    def test_store_document_calls_store_documents(self):
        """Test that store_document delegates to store_documents."""
        document = Document(
            id="test_id", page_content="test content", metadata={"title": "Test"}
        )
        embedding = [0.1, 0.2, 0.3]

        with patch.object(self.storage, "store_documents") as mock_store_documents:
            self.storage.store_document(document, embedding)

            mock_store_documents.assert_called_once_with([document], [embedding])

    def test_store_documents_validation_error(self):
        """Test store_documents with mismatched embeddings."""
        documents = [
            Document(id="1", page_content="content1", metadata={}),
            Document(id="2", page_content="content2", metadata={}),
        ]
        embeddings = [[0.1, 0.2]]  # Only one embedding for two documents

        with self.assertRaises(ValueError) as context:
            self.storage.store_documents(documents, embeddings)

        self.assertIn("Number of embeddings must match", str(context.exception))

    @patch("iris_rag.storage.enterprise_storage.logger")
    def test_store_documents_with_schema_init(self, mock_logger):
        """Test storing documents with automatic schema initialization."""
        documents = [
            Document(
                id="test_id", page_content="test content", metadata={"title": "Test"}
            )
        ]

        # Mock table check failure, then successful operations
        self.mock_cursor.execute.side_effect = [
            Exception("Table not found"),  # Table check fails
            [0],  # EXISTS check returns 0 (document doesn't exist)
            None,  # INSERT succeeds
        ]

        with patch.object(self.storage, "initialize_schema") as mock_init:
            result = self.storage.store_documents(documents)

            mock_init.assert_called_once()
            self.assertIn("documents_stored", result)
            self.assertIn("documents_updated", result)

    def test_store_documents_update_existing(self):
        """Test storing documents that already exist (update path)."""
        documents = [
            Document(
                id="existing_id",
                page_content="updated content",
                metadata={"title": "Updated"},
            )
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        # Mock table check success, document exists
        self.mock_cursor.execute.side_effect = [
            None,  # Table check succeeds
            [1],  # EXISTS check returns 1 (document exists)
            None,  # UPDATE succeeds
        ]
        self.mock_cursor.fetchone.return_value = [1]

        result = self.storage.store_documents(documents, embeddings)

        # Verify UPDATE was called
        update_calls = [
            call
            for call in self.mock_cursor.execute.call_args_list
            if call[0][0].strip().startswith("UPDATE")
        ]
        self.assertTrue(len(update_calls) > 0)

    def test_get_document_count(self):
        """Test getting document count."""
        self.mock_cursor.fetchone.return_value = [42]

        count = self.storage.get_document_count()

        self.assertEqual(count, 42)
        self.mock_cursor.execute.assert_called_with(
            f"SELECT COUNT(*) FROM {self.storage.table_name}"
        )

    def test_clear_documents(self):
        """Test clearing all documents."""
        self.storage.clear_documents()

        self.mock_cursor.execute.assert_called_with(
            f"DELETE FROM {self.storage.table_name}"
        )
        self.mock_connection.commit.assert_called_once()

    def test_retrieve_documents_by_ids(self):
        """Test retrieving documents by IDs."""
        # Mock database response
        mock_rows = [
            {
                "doc_id": "doc1",
                "title": "Title 1",
                "text_content": "Content 1",
                "metadata": '{"key": "value1"}',
                "abstract": "Abstract 1",
                "authors": "Author 1",
                "keywords": "keyword1",
            },
            {
                "doc_id": "doc2",
                "title": "Title 2",
                "text_content": "Content 2",
                "metadata": '{"key": "value2"}',
                "abstract": "Abstract 2",
                "authors": "Author 2",
                "keywords": "keyword2",
            },
        ]
        self.mock_cursor.fetchall.return_value = mock_rows

        doc_ids = ["doc1", "doc2"]
        documents = self.storage.retrieve_documents_by_ids(doc_ids)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].id, "doc1")
        self.assertEqual(documents[0].page_content, "Content 1")
        self.assertEqual(documents[0].metadata["key"], "value1")

        # Verify SQL was executed with correct parameters
        sql_calls = self.mock_cursor.execute.call_args_list
        self.assertTrue(any("SELECT" in str(call) for call in sql_calls))

    def test_retrieve_documents_by_ids_empty_list(self):
        """Test retrieving documents with empty ID list."""
        documents = self.storage.retrieve_documents_by_ids([])

        self.assertEqual(documents, [])
        # Should not execute any SQL
        self.mock_cursor.execute.assert_not_called()

    @patch("iris_rag.storage.enterprise_storage.logger")
    def test_vector_search(self, mock_logger):
        """Test vector search functionality."""
        # Mock database response
        mock_rows = [
            {
                "doc_id": "doc1",
                "title": "Similar Doc",
                "text_content": "Similar content",
                "metadata": '{"similarity": 0.95}',
                "vector_distance": 0.05,
            }
        ]
        self.mock_cursor.fetchall.return_value = mock_rows

        query_vector = [0.1, 0.2, 0.3]
        k = 5

        results = self.storage.vector_search(query_vector, k)

        self.assertEqual(len(results), 1)
        self.assertIn("document", results[0])
        self.assertIn("score", results[0])
        self.assertEqual(results[0]["document"].id, "doc1")

        # Verify vector search SQL was executed
        sql_calls = self.mock_cursor.execute.call_args_list
        self.assertTrue(
            any(
                "VECTOR_DOT_PRODUCT" in str(call) or "TOP" in str(call)
                for call in sql_calls
            )
        )

    def test_vector_search_empty_vector(self):
        """Test vector search with empty vector."""
        with self.assertRaises(ValueError):
            self.storage.vector_search([], 5)

    @patch("iris_rag.storage.enterprise_storage.logger")
    def test_error_handling_in_store_documents(self, mock_logger):
        """Test error handling in store_documents."""
        documents = [Document(id="test", page_content="content", metadata={})]

        # Mock database error
        self.mock_cursor.execute.side_effect = Exception("Database error")

        with self.assertRaises(Exception):
            self.storage.store_documents(documents)

        # Verify rollback was called
        self.mock_connection.rollback.assert_called_once()

    @patch("iris_rag.storage.enterprise_storage.logger")
    def test_error_handling_in_vector_search(self, mock_logger):
        """Test error handling in vector search."""
        query_vector = [0.1, 0.2, 0.3]

        # Mock database error
        self.mock_cursor.execute.side_effect = Exception("Search error")

        results = self.storage.vector_search(query_vector, 5)

        # Should return empty list on error
        self.assertEqual(results, [])
        mock_logger.error.assert_called()


class TestIRISStorageIntegration(unittest.TestCase):
    """Integration tests for realistic IRISStorage usage scenarios."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)
        self.mock_config_manager = Mock(spec=ConfigurationManager)

        # Mock realistic configuration
        self.mock_config_manager.get.return_value = {
            "table_name": "RAG.SourceDocuments",
            "vector_dimension": 384,
        }
        self.mock_config_manager.get_vector_index_config.return_value = {
            "M": 16,
            "efConstruction": 200,
            "Distance": "Cosine",
        }

        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection_manager.get_connection.return_value = self.mock_connection

        self.storage = IRISStorage(
            self.mock_connection_manager, self.mock_config_manager
        )

    def test_typical_document_lifecycle(self):
        """Test a typical document storage and retrieval lifecycle."""
        # Create test documents
        documents = [
            Document(
                id="doc1",
                page_content="This is the first document content",
                metadata={"title": "First Document", "category": "research"},
            ),
            Document(
                id="doc2",
                page_content="This is the second document content",
                metadata={"title": "Second Document", "category": "development"},
            ),
        ]

        embeddings = [
            [0.1, 0.2, 0.3] * 128,  # 384-dimensional vector
            [0.4, 0.5, 0.6] * 128,  # 384-dimensional vector
        ]

        # Mock successful storage operations
        self.mock_cursor.execute.side_effect = [
            None,  # Table check
            [0],  # Doc1 doesn't exist
            None,  # Insert doc1
            [0],  # Doc2 doesn't exist
            None,  # Insert doc2
        ]
        self.mock_cursor.fetchone.side_effect = [[0], [0]]

        # Store documents
        result = self.storage.store_documents(documents, embeddings)

        # Verify storage result
        self.assertIn("documents_stored", result)
        self.assertIn("documents_updated", result)

        # Mock retrieval
        mock_retrieved_rows = [
            {
                "doc_id": "doc1",
                "title": "First Document",
                "text_content": "This is the first document content",
                "metadata": '{"title": "First Document", "category": "research"}',
                "abstract": "",
                "authors": "",
                "keywords": "",
            }
        ]
        self.mock_cursor.fetchall.return_value = mock_retrieved_rows

        # Retrieve documents
        retrieved = self.storage.retrieve_documents_by_ids(["doc1"])

        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].id, "doc1")
        self.assertEqual(retrieved[0].metadata["category"], "research")

    def test_bulk_operations_performance(self):
        """Test bulk operations with larger document sets."""
        # Create 100 test documents
        documents = []
        embeddings = []

        for i in range(100):
            documents.append(
                Document(
                    id=f"bulk_doc_{i}",
                    page_content=f"Bulk document content {i}",
                    metadata={"title": f"Bulk Doc {i}", "index": i},
                )
            )
            embeddings.append([0.1 * i] * 384)

        # Mock successful bulk operations
        execute_responses = [None]  # Table check
        for i in range(100):
            execute_responses.extend([[0], None])  # Each doc: exists check + insert

        self.mock_cursor.execute.side_effect = execute_responses
        self.mock_cursor.fetchone.side_effect = [[0]] * 100

        # Perform bulk storage
        result = self.storage.store_documents(documents, embeddings)

        # Verify bulk operation completed
        self.assertIn("documents_stored", result)

        # Test bulk retrieval
        doc_ids = [f"bulk_doc_{i}" for i in range(10)]  # Retrieve first 10

        mock_bulk_rows = []
        for i in range(10):
            mock_bulk_rows.append(
                {
                    "doc_id": f"bulk_doc_{i}",
                    "title": f"Bulk Doc {i}",
                    "text_content": f"Bulk document content {i}",
                    "metadata": f'{{"title": "Bulk Doc {i}", "index": {i}}}',
                    "abstract": "",
                    "authors": "",
                    "keywords": "",
                }
            )

        self.mock_cursor.fetchall.return_value = mock_bulk_rows

        retrieved = self.storage.retrieve_documents_by_ids(doc_ids)
        self.assertEqual(len(retrieved), 10)


if __name__ == "__main__":
    unittest.main()
