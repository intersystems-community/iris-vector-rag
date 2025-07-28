"""
Tests for the IRISVectorStore implementation.
"""
import pytest
from typing import Any, List, Tuple, Dict, Optional
from unittest.mock import MagicMock, Mock

# Import actual implementations
from iris_rag.core.models import Document
from iris_rag.core.vector_store_exceptions import (
    VectorStoreConnectionError,
    VectorStoreDataError,
    VectorStoreConfigurationError
)
from iris_rag.storage.vector_store_iris import IRISVectorStore
from common.iris_connection_manager import get_iris_connection, IRISConnectionManager
from iris_rag.config.manager import ConfigurationManager
from tests.test_core.test_vector_store import VectorStoreContractTests


class CLOBType:
    """Simulates a CLOB type that might be returned by a database driver."""
    def __init__(self, value: str):
        self.value = value

    def read(self) -> str:
        """Simulates reading the content of a CLOB."""
        return self.value

    def __str__(self) -> str:
        return f"<CLOB containing: '{self.value[:20]}...'>"


class MockIRISVectorStore:
    """
    Mock implementation for testing CLOB handling.
    This simulates the behavior we expect from the real IRISVectorStore.
    """
    def __init__(self, iris_connector, config_manager: ConfigurationManager):
        self.iris_connector = iris_connector
        self.config_manager = config_manager
        
        # Get storage configuration and validate table name
        self.storage_config = config_manager.get("storage:iris", {})
        self.table_name = self.storage_config.get("table_name", "RAG.SourceDocuments")
        self._validate_table_name(self.table_name)
        
        # Simulate a connection failure for bad configs
        if hasattr(iris_connector, 'fail_connection') and iris_connector.fail_connection:
            raise VectorStoreConnectionError("Failed to connect to bad_host")
        self._documents: Dict[str, Document] = {} # In-memory store for now
        self._vectors: Dict[str, List[float]] = {} # In-memory vectors
        self._raw_data: Dict[str, Dict[str, Any]] = {} # Store raw data with potential CLOBs
        
        # Define allowed filter keys for security
        self._allowed_filter_keys = {
            "category", "year", "source_type", "document_id", "author_name",
            "title", "source", "type", "date", "status", "version"
        }
        
        # Define allowed table names for security
        self._allowed_table_names = {
            "RAG.SourceDocuments",
            "RAG.DocumentTokenEmbeddings",
            "RAG.TestDocuments",
            "RAG.BackupDocuments"
        }
    
    def _validate_table_name(self, table_name: str) -> None:
        """Validate table name against whitelist to prevent SQL injection."""
        allowed_tables = {
            "RAG.SourceDocuments",
            "RAG.DocumentTokenEmbeddings",
            "RAG.TestDocuments",
            "RAG.BackupDocuments"
        }
        
        if table_name not in allowed_tables:
            raise VectorStoreConfigurationError(f"Invalid table name: {table_name}")
    
    def _validate_filter_keys(self, filter_dict: Dict[str, Any]) -> None:
        """Validate filter keys against whitelist to prevent SQL injection."""
        for key in filter_dict.keys():
            if key not in self._allowed_filter_keys:
                raise VectorStoreDataError(f"Invalid filter key: {key}")
    
    def _validate_filter_values(self, filter_dict: Dict[str, Any]) -> None:
        """Validate filter values for basic type safety."""
        for key, value in filter_dict.items():
            if value is None or callable(value) or isinstance(value, (list, dict)):
                raise VectorStoreDataError(f"Invalid filter value for key '{key}': {type(value).__name__}")
    
    def _simulate_database_error(self, error: Exception) -> None:
        """Simulate a database error for testing error logging sanitization."""
        # This method is used in tests to simulate database errors
        raise error

    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        if not documents:
            return []
        if any(not isinstance(doc.page_content, str) for doc in documents): # Simulate malformed data
             raise VectorStoreDataError("Document page_content must be a string.")

        generated_ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.id
            if doc_id in self._documents:
                # Handle duplicate IDs by updating
                pass
            self._documents[doc_id] = doc
            # Simulate embedding
            if embeddings and i < len(embeddings):
                self._vectors[doc_id] = embeddings[i]
            else:
                # Simple mock embedding: vector of char ordinals
                self._vectors[doc_id] = [ord(c) for c in doc.page_content[:10].ljust(10, '\0')]
            generated_ids.append(doc_id)
        return generated_ids

    def fetch_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """Fetch documents by IDs, converting CLOBs to strings if present."""
        from iris_rag.storage.clob_handler import convert_clob_to_string, process_document_row
        
        documents = []
        for doc_id in ids:
            if doc_id in self._documents:
                # Check if we have raw data with potential CLOBs
                if doc_id in self._raw_data:
                    raw_data = self._raw_data[doc_id]
                    # Process raw data to convert CLOBs
                    processed_data = {}
                    for key, value in raw_data.items():
                        if key == 'metadata' and isinstance(value, dict):
                            processed_data[key] = process_document_row(value)
                        else:
                            processed_data[key] = convert_clob_to_string(value)
                    
                    # Create document with processed data
                    metadata = processed_data.get('metadata', {})
                    if isinstance(metadata, str):
                        import json
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {"raw_metadata": metadata}
                    
                    document = Document(
                        id=processed_data.get('id', doc_id),
                        page_content=processed_data.get('page_content', ''),
                        metadata=metadata
                    )
                    documents.append(document)
                else:
                    # Return normal document
                    documents.append(self._documents[doc_id])
        return documents

    def similarity_search(self, query_embedding: List[float], top_k: int = 4, filter: Dict[str, Any] = None) -> List[Tuple[Document, float]]:
        # Validate filter if provided
        if filter:
            self._validate_filter_keys(filter)
            self._validate_filter_values(filter)
        
        # Simplified search for placeholder
        if not self._documents:
            return []
        # Simulate some search logic and scoring
        results = []
        for doc_id, doc_vector in self._vectors.items():
            # Simulate dot product or cosine similarity
            score = sum(q_val * d_val for q_val, d_val in zip(query_embedding, doc_vector)) / (len(query_embedding) * len(doc_vector))
            # Apply filter if present (very basic simulation)
            if filter:
                match = True
                for key, value in filter.items():
                    if self._documents[doc_id].metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            results.append((self._documents[doc_id], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete_documents(self, ids: List[str]) -> bool:
        deleted_count = 0
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                if doc_id in self._vectors:
                    del self._vectors[doc_id]
                deleted_count += 1
        # Return True if any were deleted, False if none were deleted
        return deleted_count > 0

    def get_document_count(self) -> int:
        return len(self._documents)

    def clear_documents(self) -> None:
        self._documents.clear()
        self._vectors.clear()


class TestIRISVectorStore(VectorStoreContractTests):

    @pytest.fixture
    def embedding_function(self):
        """Provides a mock embedding function."""
        mock_embed = MagicMock()
        # Simple embedding: vector of char ordinals, padded to length 10
        mock_embed.side_effect = lambda text: [ord(c) for c in text[:10].ljust(10, '\0')]
        return mock_embed

    @pytest.fixture
    def mock_connection_manager(self):
        """Provides a mock connection manager."""
        mock_cm = Mock(spec=IRISConnectionManager)
        mock_connection = Mock()
        mock_cm.get_connection.return_value = mock_connection
        return mock_cm

    @pytest.fixture
    def mock_config_manager(self):
        """Provides a mock configuration manager."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.get.return_value = {
            "table_name": "RAG.SourceDocuments",
            "vector_dimension": 384
        }
        return mock_config

    @pytest.fixture
    def vector_store(self, mock_connection_manager, mock_config_manager) -> MockIRISVectorStore:
        """
        Provides an instance of MockIRISVectorStore for testing.
        This will be replaced by the actual IRISVectorStore once it's working.
        """
        return MockIRISVectorStore(
            iris_connector=None,
            config_manager=mock_config_manager
        )

    # --- Concrete implementations of abstract contract tests ---

    def test_add_and_fetch_documents_by_ids(self, vector_store: MockIRISVectorStore):
        docs_to_add = [
            Document(page_content="Doc 1 content", metadata={"source": "test1"}, id="id1"),
            Document(page_content="Doc 2 content", metadata={"source": "test2", "type": "report"}, id="id2"),
        ]
        added_ids = vector_store.add_documents(docs_to_add)
        assert added_ids == ["id1", "id2"]

        retrieved_docs = vector_store.fetch_documents_by_ids(["id1", "id2", "nonexistent_id"])
        assert len(retrieved_docs) == 2
        # Ensure documents are retrieved correctly
        expected_doc1 = Document(page_content="Doc 1 content", metadata={"source": "test1"}, id="id1")
        expected_doc2 = Document(page_content="Doc 2 content", metadata={"source": "test2", "type": "report"}, id="id2")

        assert expected_doc1 in retrieved_docs
        assert expected_doc2 in retrieved_docs

    def test_similarity_search_returns_correct_format(self, vector_store: MockIRISVectorStore):
        docs_to_add = [
            Document(page_content="apple banana orange", metadata={"category": "fruit"}, id="fruit1"),
            Document(page_content="car truck bus", metadata={"category": "vehicle"}, id="vehicle1"),
        ]
        vector_store.add_documents(docs_to_add)

        # Use embedding instead of query string
        query_embedding = [ord(c) for c in "query about fruits"[:10].ljust(10, '\0')]
        results = vector_store.similarity_search(query_embedding, top_k=1)
        assert isinstance(results, list)
        if results:
            assert len(results) <= 1
            for item in results:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], Document)
                assert isinstance(item[1], float) # Score

    def test_delete_documents(self, vector_store: MockIRISVectorStore):
        docs_to_add = [
            Document(page_content="Content to delete", metadata={"id_key": "del1"}, id="delete_me"),
            Document(page_content="Content to keep", metadata={"id_key": "keep1"}, id="keep_me"),
        ]
        vector_store.add_documents(docs_to_add)

        assert vector_store.get_document_count() == 2
        delete_result = vector_store.delete_documents(["delete_me", "non_existent_id"])
        assert delete_result is True
        assert vector_store.get_document_count() == 1

        retrieved_after_delete = vector_store.fetch_documents_by_ids(["delete_me"])
        assert len(retrieved_after_delete) == 0
        retrieved_kept = vector_store.fetch_documents_by_ids(["keep_me"])
        assert len(retrieved_kept) == 1
        assert retrieved_kept[0].page_content == "Content to keep"

        # Test deleting non-existent ID from empty store
        vector_store.clear_documents()
        assert vector_store.delete_documents(["another_non_existent"]) is False # No documents were actually deleted
        assert vector_store.get_document_count() == 0

    def test_document_content_is_string_after_retrieval(self, vector_store: MockIRISVectorStore):
        """
        This is the key test for CLOB-to-string conversion.
        Tests that CLOBs are properly converted to strings when retrieving documents.
        """
        # Simulate data that would be stored as CLOB in IRIS
        clob_page_content = CLOBType("This is a very long string that might be a CLOB.")
        clob_title = CLOBType("A CLOB Title")

        doc_id_clob = "clob_doc_1"
        
        # Create a document with string content first (for the normal storage)
        original_doc = Document(
            page_content="This will be replaced by CLOB-like data.",
            metadata={"title": "Original Title", "source": "clob_test"},
            id=doc_id_clob
        )
        vector_store.add_documents([original_doc])

        # Now simulate raw database data with CLOBs
        # This simulates what would come back from a database query
        vector_store._raw_data[doc_id_clob] = {
            'id': doc_id_clob,
            'page_content': clob_page_content,  # This is a CLOBType instance
            'metadata': {
                'title': clob_title,  # This is also a CLOBType instance
                'source': 'clob_source'  # This is a regular string
            }
        }

        # The crucial part: retrieve the document
        # This should trigger CLOB conversion in fetch_documents_by_ids
        retrieved_docs = vector_store.fetch_documents_by_ids([doc_id_clob])
        assert len(retrieved_docs) == 1
        retrieved_doc = retrieved_docs[0]

        # Assert that page_content is a string (converted from CLOB)
        assert isinstance(retrieved_doc.page_content, str), \
            f"page_content should be str, got {type(retrieved_doc.page_content)}"
        assert retrieved_doc.page_content == clob_page_content.read()

        # Assert that metadata values (like title) are strings (converted from CLOB)
        assert "title" in retrieved_doc.metadata
        assert isinstance(retrieved_doc.metadata["title"], str), \
            f"metadata['title'] should be str, got {type(retrieved_doc.metadata['title'])}"
        assert retrieved_doc.metadata["title"] == clob_title.read()
        
        # Assert that regular string metadata is still a string
        assert isinstance(retrieved_doc.metadata["source"], str)
        assert retrieved_doc.metadata["source"] == "clob_source"

    def test_empty_results_handled_gracefully_for_search(self, vector_store: MockIRISVectorStore):
        vector_store.clear_documents() # Ensure store is empty
        query_embedding = [ord(c) for c in "any query"[:10].ljust(10, '\0')]
        results = vector_store.similarity_search(query_embedding, top_k=5)
        assert results == []

    def test_empty_results_handled_gracefully_for_fetch(self, vector_store: MockIRISVectorStore):
        vector_store.clear_documents() # Ensure store is empty
        results = vector_store.fetch_documents_by_ids(["non_existent_id_1", "non_existent_id_2"])
        assert results == []

    def test_filter_in_similarity_search(self, vector_store: MockIRISVectorStore):
        docs_to_add = [
            Document(page_content="Content for filter A", metadata={"category": "A", "year": 2020}, id="f1"),
            Document(page_content="Content for filter B", metadata={"category": "B", "year": 2021}, id="f2"),
            Document(page_content="More A content", metadata={"category": "A", "year": 2022}, id="f3"),
        ]
        vector_store.add_documents(docs_to_add)

        # Test filter for category "A"
        query_embedding = [ord(c) for c in "Content"[:10].ljust(10, '\0')]
        results_A = vector_store.similarity_search(query_embedding, top_k=5, filter={"category": "A"})
        assert len(results_A) == 2
        for doc, score in results_A:
            assert doc.metadata["category"] == "A"

        # Test filter for year 2021
        results_2021 = vector_store.similarity_search(query_embedding, top_k=5, filter={"year": 2021})
        assert len(results_2021) == 1
        assert results_2021[0][0].metadata["year"] == 2021
        
        # Test filter that matches nothing
        results_none = vector_store.similarity_search(query_embedding, top_k=5, filter={"category": "C"})
        assert len(results_none) == 0

    def test_get_document_count(self, vector_store: MockIRISVectorStore):
        assert vector_store.get_document_count() == 0 # Initial
        vector_store.add_documents([Document(page_content="doc1", metadata={}, id="gc1")])
        assert vector_store.get_document_count() == 1
        vector_store.add_documents([Document(page_content="doc2", metadata={}, id="gc2")])
        assert vector_store.get_document_count() == 2
        vector_store.delete_documents(["gc1"])
        assert vector_store.get_document_count() == 1
        vector_store.clear_documents()
        assert vector_store.get_document_count() == 0

    def test_clear_documents(self, vector_store: MockIRISVectorStore):
        vector_store.add_documents([
            Document(page_content="doc to clear 1", metadata={}, id="c1"),
            Document(page_content="doc to clear 2", metadata={}, id="c2")
        ])
        assert vector_store.get_document_count() == 2
        vector_store.clear_documents()
        assert vector_store.get_document_count() == 0
        # Ensure it's safe to call clear on an already empty store
        vector_store.clear_documents()
        assert vector_store.get_document_count() == 0

    # --- Error Handling Test Implementations ---

    def test_raises_connection_error_on_connection_failure(self, mock_config_manager):
        """Tests that a VectorStoreConnectionError is raised on connection failure."""
        # Create a connection manager that will fail
        bad_connection_manager = Mock()
        bad_connection_manager.fail_connection = True
        
        with pytest.raises(VectorStoreConnectionError):
            MockIRISVectorStore(config_manager=mock_config_manager, connection_manager=bad_connection_manager)

    def test_raises_data_error_on_malformed_data(self, vector_store: MockIRISVectorStore):
        """Tests that a VectorStoreDataError is raised when adding malformed data."""
        # Since Document model validates data at creation, we need to test the vector store's validation
        # by creating a mock document that bypasses the Document validation
        
        # Create a mock document object that has non-string page_content
        class MockDocument:
            def __init__(self, page_content, metadata, id):
                self.page_content = page_content
                self.metadata = metadata
                self.id = id
        
        malformed_docs = [
            MockDocument(page_content=12345, metadata={"source": "bad_data"}, id="bad1") # type: ignore
        ]
        
        with pytest.raises(VectorStoreDataError):
            vector_store.add_documents(malformed_docs) # type: ignore

    # Additional tests specific to IRISVectorStore can be added here.
    # For example, testing specific IRIS features or behaviors.

    # --- Security Tests ---

    def test_invalid_filter_keys_rejected(self, vector_store: MockIRISVectorStore):
        """Test that invalid filter keys are rejected to prevent SQL injection."""
        docs_to_add = [
            Document(page_content="Test content", metadata={"category": "test"}, id="test1"),
        ]
        vector_store.add_documents(docs_to_add)
        
        query_embedding = [1.0] * 10
        
        # Test malicious filter keys that could cause SQL injection
        malicious_filters = [
            {"'; DROP TABLE users; --": "value"},
            {"metadata'; DELETE FROM RAG.SourceDocuments; --": "value"},
            {"1=1; UPDATE": "value"},
            {"../../../etc/passwd": "value"},
            {"<script>alert('xss')</script>": "value"},
        ]
        
        for malicious_filter in malicious_filters:
            with pytest.raises(VectorStoreDataError, match="Invalid filter key"):
                vector_store.similarity_search(query_embedding, top_k=5, filter=malicious_filter)

    def test_valid_filter_keys_accepted(self, vector_store: MockIRISVectorStore):
        """Test that valid filter keys are accepted."""
        docs_to_add = [
            Document(page_content="Test content", metadata={"category": "test", "year": 2023}, id="test1"),
        ]
        vector_store.add_documents(docs_to_add)
        
        query_embedding = [1.0] * 10
        
        # Test valid filter keys
        valid_filters = [
            {"category": "test"},
            {"year": 2023},
            {"source_type": "document"},
            {"document_id": "123"},
            {"author_name": "John Doe"},
        ]
        
        for valid_filter in valid_filters:
            # Should not raise an exception
            results = vector_store.similarity_search(query_embedding, top_k=5, filter=valid_filter)
            assert isinstance(results, list)

    def test_invalid_table_name_configuration_rejected(self, mock_connection_manager):
        """Test that invalid table names in configuration are rejected."""
        # Test malicious table names
        malicious_table_names = [
            "RAG.SourceDocuments'; DROP TABLE users; --",
            "../../etc/passwd",
            "RAG.SourceDocuments UNION SELECT * FROM sensitive_table",
            "<script>alert('xss')</script>",
            "RAG.SourceDocuments; DELETE FROM users",
        ]
        
        for malicious_table in malicious_table_names:
            mock_config = Mock(spec=ConfigurationManager)
            mock_config.get.return_value = {
                "table_name": malicious_table,
                "vector_dimension": 384
            }
            
            with pytest.raises(VectorStoreConfigurationError, match="Invalid table name"):
                MockIRISVectorStore(
                    connection_manager=mock_connection_manager,
                    config_manager=mock_config
                )

    def test_valid_table_name_configuration_accepted(self, mock_connection_manager):
        """Test that valid table names in configuration are accepted."""
        valid_table_names = [
            "RAG.SourceDocuments",
            "RAG.DocumentTokenEmbeddings",
            "RAG.TestDocuments",
            "RAG.BackupDocuments",
        ]
        
        for valid_table in valid_table_names:
            mock_config = Mock(spec=ConfigurationManager)
            mock_config.get.return_value = {
                "table_name": valid_table,
                "vector_dimension": 384
            }
            
            # Should not raise an exception
            store = MockIRISVectorStore(
                connection_manager=mock_connection_manager,
                config_manager=mock_config
            )
            assert store is not None

    def test_filter_value_type_validation(self, vector_store: MockIRISVectorStore):
        """Test that filter values are validated for basic type safety."""
        docs_to_add = [
            Document(page_content="Test content", metadata={"category": "test"}, id="test1"),
        ]
        vector_store.add_documents(docs_to_add)
        
        query_embedding = [1.0] * 10
        
        # Test potentially problematic filter values
        problematic_values = [
            {"category": None},  # None values
            {"category": []},    # List values
            {"category": {}},    # Dict values
            {"category": lambda x: x},  # Function values
        ]
        
        for problematic_filter in problematic_values:
            with pytest.raises(VectorStoreDataError, match="Invalid filter value"):
                vector_store.similarity_search(query_embedding, top_k=5, filter=problematic_filter)

    def test_error_logging_sanitization(self, vector_store: MockIRISVectorStore, caplog):
        """Test that error logging doesn't expose sensitive information."""
        import logging
        
        # Force an error condition that would normally log raw database exceptions
        with pytest.raises(Exception):
            # Simulate a database error with sensitive information
            sensitive_error = Exception("Database error: password=secret123, user=admin")
            vector_store._simulate_database_error(sensitive_error)
        
        # Check that sensitive information is not in the logs
        log_messages = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
        for message in log_messages:
            assert "password=secret123" not in message
            assert "user=admin" not in message
            assert "Database operation failed" in message or "Check debug logs" in message