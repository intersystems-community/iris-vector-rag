"""
Tests for EntityExtractionService to ensure it follows project patterns.

Simple tests that validate the service works and integrates with existing components.
"""

from unittest.mock import Mock, patch

import pytest

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import (
    Document,
    Entity,
    EntityTypes,
    Relationship,
    RelationshipTypes,
)
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.services.entity_extraction import EntityExtractionService


class TestEntityExtractionService:
    """Test suite for EntityExtractionService."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = {
            "entity_extraction": {
                "enabled": True,
                "method": "llm_basic",
                "confidence_threshold": 0.7,
                "entity_types": ["PERSON", "DISEASE", "DRUG"],
                "storage": {
                    "entities_table": "RAG.Entities",
                    "relationships_table": "RAG.EntityRelationships",
                },
            }
        }

        # Create mock managers
        self.config_manager = Mock(spec=ConfigurationManager)
        self.config_manager.get.return_value = self.mock_config["entity_extraction"]
        self.config_manager._config = self.mock_config

        self.connection_manager = Mock(spec=ConnectionManager)
        self.embedding_manager = Mock(spec=EmbeddingManager)

        # Create test document
        self.test_document = Document(
            page_content="Patient John Doe was prescribed aspirin for hypertension. The drug helps prevent heart disease.",
            metadata={"source": "medical_record"},
            id="test-doc-123",
        )

    def test_service_initialization(self):
        """Test that the service initializes correctly with proper configuration."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
            embedding_manager=self.embedding_manager,
        )

        assert service.config_manager == self.config_manager
        assert service.connection_manager == self.connection_manager
        assert service.embedding_manager == self.embedding_manager
        assert service.method == "llm_basic"
        assert service.confidence_threshold == 0.7
        assert "PERSON" in service.enabled_types
        assert "DISEASE" in service.enabled_types
        assert "DRUG" in service.enabled_types
        assert service.storage_adapter is not None

    def test_service_initialization_without_managers(self):
        """Test service initialization with minimal configuration."""
        service = EntityExtractionService(config_manager=self.config_manager)

        assert service.config_manager == self.config_manager
        assert service.connection_manager is None
        assert service.embedding_manager is None
        assert service.storage_adapter is None

    def test_pattern_extraction(self):
        """Test pattern-based entity extraction."""
        # Configure for pattern-only extraction
        self.config_manager.get.return_value = {
            **self.mock_config["entity_extraction"],
            "method": "pattern_only",
        }

        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )

        entities = service.extract_entities(self.test_document)

        # Should find at least one entity (aspirin as DRUG)
        assert len(entities) > 0

        # Check that entities have proper structure
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.source_document_id == self.test_document.id
            assert hasattr(entity, "text")
            assert hasattr(entity, "entity_type")
            assert hasattr(entity, "confidence")
            assert 0 <= entity.confidence <= 1

    @patch("iris_rag.services.entity_extraction.EntityExtractionService._call_llm")
    def test_llm_extraction(self, mock_llm_call):
        """Test LLM-based entity extraction."""
        # Mock LLM response
        mock_llm_call.return_value = '{"entities": [{"text": "John Doe", "type": "PERSON", "confidence": 0.95, "start": 8, "end": 16}]}'

        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )

        entities = service.extract_entities(self.test_document)

        assert len(entities) == 1
        entity = entities[0]
        assert entity.text == "John Doe"
        assert entity.entity_type == "PERSON"
        assert entity.confidence == 0.95
        assert entity.source_document_id == self.test_document.id
        mock_llm_call.assert_called_once()

    def test_relationship_extraction(self):
        """Test relationship extraction between entities."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )

        # Create test entities
        entities = [
            Entity(
                text="aspirin",
                entity_type=EntityTypes.DRUG,
                confidence=0.9,
                start_offset=30,
                end_offset=37,
                source_document_id=self.test_document.id,
            ),
            Entity(
                text="hypertension",
                entity_type=EntityTypes.DISEASE,
                confidence=0.85,
                start_offset=42,
                end_offset=54,
                source_document_id=self.test_document.id,
            ),
        ]

        relationships = service.extract_relationships(entities, self.test_document)

        # Should find relationships between drug and disease
        assert isinstance(relationships, list)
        for rel in relationships:
            assert isinstance(rel, Relationship)
            assert rel.source_document_id == self.test_document.id
            assert hasattr(rel, "relationship_type")
            assert hasattr(rel, "confidence")

    def test_document_processing_workflow(self):
        """Test complete document processing workflow."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
            embedding_manager=self.embedding_manager,
        )

        # Mock storage adapter methods
        service.storage_adapter.store_entities_batch = Mock(return_value=2)
        service.storage_adapter.store_relationships_batch = Mock(return_value=1)

        # Mock embedding generation
        self.embedding_manager.generate_embedding.return_value = [0.1, 0.2, 0.3]

        result = service.process_document(self.test_document)

        assert "document_id" in result
        assert "entities_count" in result
        assert "relationships_count" in result
        assert "stored" in result
        assert result["document_id"] == self.test_document.id
        assert isinstance(result["entities_count"], int)
        assert isinstance(result["relationships_count"], int)

    def test_configuration_integration(self):
        """Test that service properly integrates with ConfigurationManager."""
        # Test with different configuration
        different_config = {
            "entity_extraction": {
                "method": "hybrid",
                "confidence_threshold": 0.8,
                "entity_types": ["PERSON", "ORGANIZATION"],
            }
        }

        self.config_manager.get.return_value = different_config["entity_extraction"]

        service = EntityExtractionService(config_manager=self.config_manager)

        assert service.method == "hybrid"
        assert service.confidence_threshold == 0.8
        assert "PERSON" in service.enabled_types
        assert "ORGANIZATION" in service.enabled_types
        assert "DRUG" not in service.enabled_types

    def test_error_handling(self):
        """Test that service handles errors gracefully."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )

        # Test with invalid document
        invalid_doc = Document(
            page_content="", metadata={}, id="invalid-doc"  # Empty content
        )

        # Should not raise exception, should return empty list
        entities = service.extract_entities(invalid_doc)
        assert isinstance(entities, list)

        # Test relationship extraction with empty entity list
        relationships = service.extract_relationships([], self.test_document)
        assert isinstance(relationships, list)
        assert len(relationships) == 0

    def test_entity_confidence_filtering(self):
        """Test that entities are filtered by confidence threshold."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )

        # Mock low-confidence entities
        with patch.object(service, "_extract_llm") as mock_extract:
            low_confidence_entities = [
                Entity(
                    text="test",
                    entity_type="PERSON",
                    confidence=0.5,  # Below threshold
                    start_offset=0,
                    end_offset=4,
                    source_document_id=self.test_document.id,
                )
            ]
            mock_extract.return_value = low_confidence_entities

            entities = service.extract_entities(self.test_document)

            # Should filter out low confidence entities
            assert len(entities) == 0

    def test_storage_adapter_integration(self):
        """Test integration with storage adapter."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )

        # Mock storage methods
        service.storage_adapter.store_entities_batch = Mock(return_value=5)
        service.storage_adapter.store_relationships_batch = Mock(return_value=3)

        test_entities = [Mock(spec=Entity) for _ in range(5)]
        test_relationships = [Mock(spec=Relationship) for _ in range(3)]

        result = service.store_entities_and_relationships(
            test_entities, test_relationships
        )

        assert result is True
        service.storage_adapter.store_entities_batch.assert_called_once_with(
            test_entities
        )
        service.storage_adapter.store_relationships_batch.assert_called_once_with(
            test_relationships
        )


if __name__ == "__main__":
    pytest.main([__file__])
