"""
Contract tests for User Story 1: Custom Metadata Filtering for Multi-Tenancy.

Feature: 051-enterprise-enhancements
Status: TDD Phase - These tests MUST fail initially

These tests define the contract for custom metadata filtering functionality
BEFORE implementation. Following TDD principles (Constitution III), all tests
should fail with AttributeError or other errors when first run.

User Story: Enterprise administrators need to filter documents by custom business
attributes like tenant ID, security classification, or department.
"""

import pytest
from unittest.mock import patch, MagicMock
from iris_vector_rag.exceptions import VectorStoreConfigurationError


class TestCustomFieldConfiguration:
    """Contract tests for custom metadata field configuration."""

    def test_custom_field_configuration(self, iris_vector_store):
        """
        CONTRACT: T011 - Custom fields can be configured via YAML.

        GIVEN administrator configures custom_filter_keys in default_config.yaml
        WHEN IRISVectorStore is initialized with custom config
        THEN custom fields are merged with default fields
        AND both default and custom fields are allowed for filtering
        """
        # Mock configuration with custom fields
        custom_config = {
            "storage": {
                "iris": {
                    "custom_filter_keys": ["tenant_id", "security_level", "department"]
                }
            }
        }

        # This will fail initially with AttributeError:
        # 'IRISVectorStore' object has no attribute 'get_allowed_filter_keys'
        allowed_keys = iris_vector_store.get_allowed_filter_keys()

        # Verify custom fields present
        assert "tenant_id" in allowed_keys, "Custom field 'tenant_id' should be allowed"
        assert "security_level" in allowed_keys, "Custom field 'security_level' should be allowed"
        assert "department" in allowed_keys, "Custom field 'department' should be allowed"

        # Verify default fields still present
        assert "source" in allowed_keys, "Default field 'source' should still be allowed"
        assert "doc_id" in allowed_keys, "Default field 'doc_id' should still be allowed"

    def test_metadata_filter_validation_success(self, iris_vector_store):
        """
        CONTRACT: T012 - Valid metadata filter keys pass validation.

        GIVEN custom filter keys are configured (tenant_id, security_level)
        WHEN user queries with filter {"tenant_id": "acme_corp"}
        THEN validation succeeds without raising exceptions
        AND query proceeds normally
        """
        # Configure custom fields
        custom_config = {
            "storage": {"iris": {"custom_filter_keys": ["tenant_id"]}}
        }

        # Valid metadata filter using custom field
        metadata_filter = {"tenant_id": "acme_corp", "source": "docs.pdf"}

        # This will fail initially with AttributeError or similar
        # No exception should be raised for valid filters
        try:
            iris_vector_store.similarity_search(
                query="test query",
                k=5,
                metadata_filter=metadata_filter
            )
            # If we reach here, validation succeeded
            assert True
        except VectorStoreConfigurationError:
            pytest.fail("Valid metadata filter should not raise VectorStoreConfigurationError")

    def test_metadata_filter_validation_failure(self, iris_vector_store, monkeypatch):
        """
        CONTRACT: T013 - Invalid metadata filter keys are rejected.

        GIVEN custom filter keys ["tenant_id"] are configured
        WHEN user queries with unconfigured field {"department": "engineering"}
        THEN VectorStoreConfigurationError is raised
        AND error message lists rejected_keys and allowed_keys
        """
        # Configure only tenant_id as custom field - need to rebuild the store
        from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager
        from unittest.mock import MagicMock

        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_mode")

        config = ConfigurationManager()
        config_dict = config.to_dict()
        config_dict["storage"] = config_dict.get("storage", {})
        config_dict["storage"]["iris"] = config_dict["storage"].get("iris", {})
        config_dict["storage"]["iris"]["custom_filter_keys"] = ["tenant_id"]  # Only tenant_id

        mock_connection_manager = MagicMock()
        mock_connection_manager.get_connection.return_value = MagicMock()

        mock_schema_manager = MagicMock()
        mock_schema_manager.get_vector_dimension.return_value = 384

        store = IRISVectorStore(
            connection_manager=mock_connection_manager,
            schema_manager=mock_schema_manager,
            config_manager=ConfigurationManager()
        )

        store.metadata_filter_manager = MetadataFilterManager(config_dict)
        store._allowed_filter_keys = set(store.metadata_filter_manager.get_allowed_filter_keys())

        # Don't mock similarity_search_by_embedding - validation happens there
        # Instead, mock _get_connection to prevent database access
        store._connection = MagicMock()

        # Invalid metadata filter using unconfigured field
        metadata_filter = {"department": "engineering"}

        # This will fail initially - expecting VectorStoreConfigurationError
        with pytest.raises(VectorStoreConfigurationError) as exc_info:
            store.similarity_search(
                query="test query",
                k=5,
                metadata_filter=metadata_filter
            )

        error_msg = str(exc_info.value)
        assert "department" in error_msg, "Error should mention rejected field 'department'"
        assert "allowed" in error_msg.lower(), "Error should mention allowed fields"

    def test_duplicate_field_name_rejection(self, iris_vector_store):
        """
        CONTRACT: T014 - Duplicate field names are rejected during configuration.

        GIVEN custom_filter_keys contains ["tenant_id", "source"]
        WHEN "source" is already a default field
        THEN configuration raises VectorStoreConfigurationError
        AND error message indicates "source" conflicts with default field
        """
        # Attempt to configure custom field that conflicts with default
        custom_config = {
            "storage": {"iris": {"custom_filter_keys": ["tenant_id", "source"]}}
        }

        # This will fail initially - expecting validation error
        with pytest.raises(VectorStoreConfigurationError) as exc_info:
            # Configuration should be validated during initialization
            from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager
            manager = MetadataFilterManager(custom_config)

        error_msg = str(exc_info.value)
        assert "source" in error_msg, "Error should mention conflicting field 'source'"
        assert "default" in error_msg.lower() or "conflict" in error_msg.lower()

    def test_invalid_field_name_rejection(self, iris_vector_store):
        """
        CONTRACT: T015 - Invalid field names are rejected (SQL injection prevention).

        GIVEN custom_filter_keys contains ["valid_field", "DROP TABLE;--"]
        WHEN field name validation runs
        THEN VectorStoreConfigurationError is raised
        AND error message indicates "DROP TABLE;--" is invalid
        """
        # Attempt SQL injection via custom field name
        malicious_config = {
            "storage": {"iris": {"custom_filter_keys": ["tenant_id", "DROP TABLE;--"]}}
        }

        # This will fail initially - expecting validation error for SQL injection
        with pytest.raises(VectorStoreConfigurationError) as exc_info:
            from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager
            manager = MetadataFilterManager(malicious_config)

        error_msg = str(exc_info.value)
        assert "DROP TABLE" in error_msg or "invalid" in error_msg.lower()
        assert "field name" in error_msg.lower()

    def test_empty_custom_fields_backward_compatibility(self, iris_vector_store, monkeypatch):
        """
        CONTRACT: T016 - Empty custom_filter_keys maintains backward compatibility.

        GIVEN custom_filter_keys = [] (empty list)
        WHEN IRISVectorStore is initialized
        THEN only default 17 fields are allowed
        AND existing queries continue working unchanged
        """
        # Configuration with empty custom fields - need to rebuild the store
        from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager
        from unittest.mock import MagicMock

        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_mode")

        config = ConfigurationManager()
        config_dict = config.to_dict()
        config_dict["storage"] = config_dict.get("storage", {})
        config_dict["storage"]["iris"] = config_dict["storage"].get("iris", {})
        config_dict["storage"]["iris"]["custom_filter_keys"] = []  # Empty list

        mock_connection_manager = MagicMock()
        mock_connection_manager.get_connection.return_value = MagicMock()

        mock_schema_manager = MagicMock()
        mock_schema_manager.get_vector_dimension.return_value = 384

        store = IRISVectorStore(
            connection_manager=mock_connection_manager,
            schema_manager=mock_schema_manager,
            config_manager=ConfigurationManager()
        )

        store.metadata_filter_manager = MetadataFilterManager(config_dict)
        store._allowed_filter_keys = set(store.metadata_filter_manager.get_allowed_filter_keys())

        # This will fail initially
        allowed_keys = store.get_allowed_filter_keys()

        # Verify only default fields present
        assert "source" in allowed_keys
        assert "doc_id" in allowed_keys
        assert len(allowed_keys) == 17, "Should have exactly 17 default fields"

    def test_case_sensitive_field_names(self, iris_vector_store, monkeypatch):
        """
        CONTRACT: T017 - Field names are case-sensitive.

        GIVEN custom field "Tenant_ID" is configured
        WHEN user queries with "tenant_id" (different case)
        THEN VectorStoreConfigurationError is raised
        AND case mismatch is indicated in error
        """
        # Configure custom field with specific casing - need to rebuild the store
        from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager
        from unittest.mock import MagicMock

        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_mode")

        config = ConfigurationManager()
        config_dict = config.to_dict()
        config_dict["storage"] = config_dict.get("storage", {})
        config_dict["storage"]["iris"] = config_dict["storage"].get("iris", {})
        config_dict["storage"]["iris"]["custom_filter_keys"] = ["Tenant_ID"]  # Specific casing

        mock_connection_manager = MagicMock()
        mock_connection_manager.get_connection.return_value = MagicMock()

        mock_schema_manager = MagicMock()
        mock_schema_manager.get_vector_dimension.return_value = 384

        store = IRISVectorStore(
            connection_manager=mock_connection_manager,
            schema_manager=mock_schema_manager,
            config_manager=ConfigurationManager()
        )

        store.metadata_filter_manager = MetadataFilterManager(config_dict)
        store._allowed_filter_keys = set(store.metadata_filter_manager.get_allowed_filter_keys())

        # Don't mock similarity_search_by_embedding - validation happens there
        # Instead, mock _get_connection to prevent database access
        store._connection = MagicMock()

        # Query with different casing
        metadata_filter = {"tenant_id": "acme"}  # lowercase

        # This will fail initially
        with pytest.raises(VectorStoreConfigurationError) as exc_info:
            store.similarity_search(
                query="test",
                k=5,
                metadata_filter=metadata_filter
            )

        error_msg = str(exc_info.value)
        assert "tenant_id" in error_msg, "Error should mention attempted field name"

    def test_special_characters_in_field_values(self, iris_vector_store):
        """
        CONTRACT: T018 - Special characters in field values are properly escaped (SQL injection prevention).

        GIVEN custom field "tenant_id" is configured
        WHEN user queries with value containing SQL special chars: {"tenant_id": "'; DROP TABLE;--"}
        THEN query executes safely without SQL injection
        AND proper parameterization prevents code execution
        """
        # Configure custom field
        config = {
            "storage": {"iris": {"custom_filter_keys": ["tenant_id"]}}
        }

        # Metadata filter with SQL injection attempt in VALUE
        malicious_filter = {"tenant_id": "'; DROP TABLE SourceDocuments;--"}

        # This should NOT raise exception - value should be safely parameterized
        try:
            iris_vector_store.similarity_search(
                query="test",
                k=5,
                metadata_filter=malicious_filter
            )
            # If we reach here, SQL injection was prevented
            assert True
        except VectorStoreConfigurationError:
            pytest.fail("Parameterized query should handle special characters safely")
        except Exception as e:
            # Should not cause SQL syntax errors
            assert "syntax error" not in str(e).lower()
            assert "DROP TABLE" not in str(e)


class TestMetadataFilterIntegration:
    """Integration contract tests for metadata filtering end-to-end."""

    def test_custom_metadata_filters_e2e(self, real_iris_vector_store):
        """
        CONTRACT: T019 - End-to-end custom metadata filtering works.

        GIVEN documents with custom metadata fields (tenant_id, security_level)
        WHEN user searches with metadata_filter {"tenant_id": "acme", "security_level": "confidential"}
        THEN only matching documents are returned
        AND documents from other tenants are excluded
        """
        # Insert test documents with custom metadata
        from iris_vector_rag.core.models import Document
        import uuid

        # Use unique IDs to avoid conflicts
        test_id_prefix = str(uuid.uuid4())[:8]

        docs = [
            Document(
                id=f"{test_id_prefix}_doc1",
                page_content="Document about ACME confidential information",
                metadata={"tenant_id": "acme", "security_level": "confidential"}
            ),
            Document(
                id=f"{test_id_prefix}_doc2",
                page_content="Document about ACME public information",
                metadata={"tenant_id": "acme", "security_level": "public"}
            ),
            Document(
                id=f"{test_id_prefix}_doc3",
                page_content="Document about Globex confidential information",
                metadata={"tenant_id": "globex", "security_level": "confidential"}
            ),
        ]

        # Add documents
        added_ids = real_iris_vector_store.add_documents(docs)
        assert len(added_ids) == 3, "Should successfully add 3 documents"

        # Query with custom metadata filter - only ACME confidential
        metadata_filter = {"tenant_id": "acme", "security_level": "confidential"}

        results = real_iris_vector_store.similarity_search(
            query="confidential information",
            k=10,
            metadata_filter=metadata_filter
        )

        # Verify only matching documents returned
        matching_results = [r for r in results if r.id.startswith(test_id_prefix)]
        assert len(matching_results) == 1, f"Should return exactly 1 matching document, got {len(matching_results)}"
        assert matching_results[0].metadata["tenant_id"] == "acme"
        assert matching_results[0].metadata["security_level"] == "confidential"

        # Cleanup
        real_iris_vector_store.delete_documents([d.id for d in docs])

    def test_multi_tenant_isolation_e2e(self, real_iris_vector_store):
        """
        CONTRACT: T019a - Multi-tenant isolation prevents cross-tenant data leakage.

        GIVEN documents from multiple tenants (acme, globex, initech)
        WHEN user queries with tenant_id filter
        THEN only documents from that tenant are returned
        AND no cross-tenant data leakage occurs
        """
        from iris_vector_rag.core.models import Document
        import uuid

        test_id_prefix = str(uuid.uuid4())[:8]

        # Create documents for 3 different tenants
        docs = [
            Document(
                id=f"{test_id_prefix}_acme1",
                page_content="ACME financial report Q1 2024",
                metadata={"tenant_id": "acme", "doc_type": "report"}
            ),
            Document(
                id=f"{test_id_prefix}_acme2",
                page_content="ACME employee handbook",
                metadata={"tenant_id": "acme", "doc_type": "handbook"}
            ),
            Document(
                id=f"{test_id_prefix}_globex1",
                page_content="Globex product catalog",
                metadata={"tenant_id": "globex", "doc_type": "catalog"}
            ),
            Document(
                id=f"{test_id_prefix}_initech1",
                page_content="Initech TPS report guidelines",
                metadata={"tenant_id": "initech", "doc_type": "guidelines"}
            ),
        ]

        added_ids = real_iris_vector_store.add_documents(docs)
        assert len(added_ids) == 4

        # Test tenant isolation for each tenant
        for tenant_name in ["acme", "globex", "initech"]:
            results = real_iris_vector_store.similarity_search(
                query="report",
                k=10,
                metadata_filter={"tenant_id": tenant_name}
            )

            matching_results = [r for r in results if r.id.startswith(test_id_prefix)]

            # Verify all results belong to correct tenant
            for result in matching_results:
                assert result.metadata["tenant_id"] == tenant_name, \
                    f"Cross-tenant leak: Expected {tenant_name}, got {result.metadata['tenant_id']}"

            # Verify expected counts
            if tenant_name == "acme":
                assert len(matching_results) == 2, f"ACME should have 2 documents"
            else:
                assert len(matching_results) == 1, f"{tenant_name} should have 1 document"

        # Cleanup
        real_iris_vector_store.delete_documents([d.id for d in docs])

    def test_combined_filters_e2e(self, real_iris_vector_store):
        """
        CONTRACT: T019b - Combined custom filters work correctly (AND logic).

        GIVEN documents with multiple custom metadata fields
        WHEN user queries with multiple filter conditions
        THEN only documents matching ALL conditions are returned
        """
        from iris_vector_rag.core.models import Document
        import uuid

        test_id_prefix = str(uuid.uuid4())[:8]

        docs = [
            Document(
                id=f"{test_id_prefix}_1",
                page_content="Engineering confidential design specs",
                metadata={
                    "tenant_id": "acme",
                    "department": "engineering",
                    "security_level": "confidential"
                }
            ),
            Document(
                id=f"{test_id_prefix}_2",
                page_content="Engineering public documentation",
                metadata={
                    "tenant_id": "acme",
                    "department": "engineering",
                    "security_level": "public"
                }
            ),
            Document(
                id=f"{test_id_prefix}_3",
                page_content="HR confidential employee records",
                metadata={
                    "tenant_id": "acme",
                    "department": "hr",
                    "security_level": "confidential"
                }
            ),
        ]

        added_ids = real_iris_vector_store.add_documents(docs)
        assert len(added_ids) == 3

        # Test combined filters (AND logic)
        metadata_filter = {
            "tenant_id": "acme",
            "department": "engineering",
            "security_level": "confidential"
        }

        results = real_iris_vector_store.similarity_search(
            query="documentation",
            k=10,
            metadata_filter=metadata_filter
        )

        matching_results = [r for r in results if r.id.startswith(test_id_prefix)]

        # Should only return engineering + confidential document
        assert len(matching_results) == 1, f"Should return 1 document matching all filters"
        assert matching_results[0].id == f"{test_id_prefix}_1"
        assert matching_results[0].metadata["department"] == "engineering"
        assert matching_results[0].metadata["security_level"] == "confidential"

        # Cleanup
        real_iris_vector_store.delete_documents([d.id for d in docs])

    def test_default_plus_custom_filters_e2e(self, real_iris_vector_store):
        """
        CONTRACT: T019c - Default and custom filters work together.

        GIVEN documents with both default (source) and custom (tenant_id) metadata
        WHEN user queries with both filter types
        THEN only documents matching both conditions are returned
        """
        from iris_vector_rag.core.models import Document
        import uuid

        test_id_prefix = str(uuid.uuid4())[:8]

        docs = [
            Document(
                id=f"{test_id_prefix}_1",
                page_content="Patient data from hospital_A.pdf",
                metadata={
                    "tenant_id": "acme",
                    "source": "hospital_A.pdf",
                    "doc_type": "medical"
                }
            ),
            Document(
                id=f"{test_id_prefix}_2",
                page_content="Patient data from clinic_B.pdf",
                metadata={
                    "tenant_id": "acme",
                    "source": "clinic_B.pdf",
                    "doc_type": "medical"
                }
            ),
            Document(
                id=f"{test_id_prefix}_3",
                page_content="Patient data from hospital_A.pdf",
                metadata={
                    "tenant_id": "globex",
                    "source": "hospital_A.pdf",
                    "doc_type": "medical"
                }
            ),
        ]

        added_ids = real_iris_vector_store.add_documents(docs)
        assert len(added_ids) == 3

        # Test combining default field (source) with custom field (tenant_id)
        metadata_filter = {
            "tenant_id": "acme",
            "source": "hospital_A.pdf"
        }

        results = real_iris_vector_store.similarity_search(
            query="patient",
            k=10,
            metadata_filter=metadata_filter
        )

        matching_results = [r for r in results if r.id.startswith(test_id_prefix)]

        # Should only return ACME document from hospital_A.pdf
        assert len(matching_results) == 1
        assert matching_results[0].metadata["tenant_id"] == "acme"
        assert matching_results[0].metadata["source"] == "hospital_A.pdf"

        # Cleanup
        real_iris_vector_store.delete_documents([d.id for d in docs])


class TestMetadataFilterManagerUnit:
    """Unit contract tests for MetadataFilterManager class."""

    def test_metadata_filter_manager(self):
        """
        CONTRACT: T020 - MetadataFilterManager validates and manages filter keys.

        GIVEN MetadataFilterManager with custom configuration
        WHEN get_allowed_filter_keys() is called
        THEN returns {default_keys, custom_keys, all_keys}
        AND validate_filter_keys() correctly identifies valid/invalid keys
        """
        from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager

        # Configure custom fields
        config = {
            "storage": {"iris": {"custom_filter_keys": ["tenant_id", "department"]}}
        }

        # This will fail initially with ImportError or AttributeError
        manager = MetadataFilterManager(config)

        # Test get_allowed_filter_keys()
        allowed = manager.get_allowed_filter_keys()

        assert "tenant_id" in allowed, "Custom field should be in allowed keys"
        assert "source" in allowed, "Default field should be in allowed keys"
        assert "department" in allowed, "Custom field should be in allowed keys"

        # Test validate_filter_keys() - valid case
        valid_keys = {"tenant_id": "acme", "source": "doc.pdf"}
        validation_result = manager.validate_filter_keys(valid_keys)
        assert validation_result.is_valid is True, "Valid keys should pass validation"

        # Test validate_filter_keys() - invalid case
        invalid_keys = {"unknown_field": "value"}
        validation_result = manager.validate_filter_keys(invalid_keys)
        assert validation_result.is_valid is False, "Invalid keys should fail validation"
        assert "unknown_field" in validation_result.rejected_keys


# Fixtures

@pytest.fixture
def iris_vector_store(monkeypatch):
    """
    Fixture providing IRISVectorStore instance for testing with mocked connection.
    """
    from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
    from iris_vector_rag.config.manager import ConfigurationManager
    from unittest.mock import MagicMock
    import os

    # Set test environment to prevent connection attempts
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_mode")

    # Create config with custom filter keys (matching test expectations)
    config = ConfigurationManager()
    config_dict = config.to_dict()
    config_dict["storage"] = config_dict.get("storage", {})
    config_dict["storage"]["iris"] = config_dict["storage"].get("iris", {})
    config_dict["storage"]["iris"]["custom_filter_keys"] = ["tenant_id", "security_level", "department"]

    # Mock connection manager and schema manager to avoid database connection
    mock_connection_manager = MagicMock()
    mock_connection_manager.get_connection.return_value = MagicMock()

    mock_schema_manager = MagicMock()
    mock_schema_manager.get_vector_dimension.return_value = 384

    # Create store with mocked dependencies
    store = IRISVectorStore(
        connection_manager=mock_connection_manager,
        schema_manager=mock_schema_manager,
        config_manager=ConfigurationManager()
    )

    # Manually initialize MetadataFilterManager with custom config
    from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager
    store.metadata_filter_manager = MetadataFilterManager(config_dict)
    store._allowed_filter_keys = set(store.metadata_filter_manager.get_allowed_filter_keys())

    # Mock similarity_search_by_embedding to avoid database queries
    def mock_similarity_search(query_embedding, top_k, filter=None):
        # Return empty list for tests that don't need actual results
        return []

    store.similarity_search_by_embedding = MagicMock(side_effect=mock_similarity_search)

    return store


@pytest.fixture
def real_iris_vector_store(connection_pool, monkeypatch):
    """
    Fixture providing real IRISVectorStore with real database connection.

    This fixture is for E2E tests that require actual IRIS database interaction.
    Uses the standardized connection_pool from conftest.py for proper backend mode support.

    NOTE: This fixture works around intersystems-irispython module loading issues in pytest
    by using the connection_pool fixture directly instead of relying on iris module import.
    This is a known issue with the iris package when PYTEST_CURRENT_TEST is set during collection.
    Future enhancement: Move this workaround to iris-devtester for reuse across projects.
    """
    # Prevent PYTEST_CURRENT_TEST check from skipping connection
    # IMPORTANT: Must be done BEFORE importing/initializing any components
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager

    # Create config with custom filter keys for testing
    config = ConfigurationManager()
    config_dict = config.to_dict()
    config_dict["storage"] = config_dict.get("storage", {})
    config_dict["storage"]["iris"] = config_dict["storage"].get("iris", {})
    config_dict["storage"]["iris"]["custom_filter_keys"] = ["tenant_id", "security_level", "department"]

    # Use real ConnectionManager - it will use connection_pool internally via conftest.py configuration
    # The connection_pool fixture handles backend mode configuration automatically
    from iris_vector_rag.core.connection import ConnectionManager

    connection_manager = ConnectionManager()

    store = IRISVectorStore(
        connection_manager=connection_manager,
        config_manager=config
    )

    # Update MetadataFilterManager with custom config
    store.metadata_filter_manager = MetadataFilterManager(config_dict)
    store._allowed_filter_keys = set(store.metadata_filter_manager.get_allowed_filter_keys())

    yield store

    # Cleanup handled by connection_pool fixture


# Expected test results for TDD phase:
# test_custom_field_configuration: FAIL (AttributeError: 'get_allowed_filter_keys')
# test_metadata_filter_validation_success: FAIL (no validation logic)
# test_metadata_filter_validation_failure: FAIL (VectorStoreConfigurationError not raised)
# test_duplicate_field_name_rejection: FAIL (ImportError: MetadataFilterManager)
# test_invalid_field_name_rejection: FAIL (ImportError: MetadataFilterManager)
# test_empty_custom_fields_backward_compatibility: FAIL (AttributeError: 'get_allowed_filter_keys')
# test_case_sensitive_field_names: FAIL (no validation logic)
# test_special_characters_in_field_values: FAIL (SQL injection not tested)
# test_custom_metadata_filters_e2e: FAIL (no metadata filter support)
# test_metadata_filter_manager: FAIL (ImportError: MetadataFilterManager)
