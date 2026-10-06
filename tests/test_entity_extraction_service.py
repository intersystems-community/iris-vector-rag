"""
Tests for EntityExtractionService to ensure it follows project patterns.

Simple tests that validate the service works and integrates with existing components.
"""

from unittest.mock import Mock, patch

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document, Entity, EntityTypes, Relationship
from iris_vector_rag.embeddings.manager import EmbeddingManager
from iris_vector_rag.services.entity_extraction import (
    EntityExtractionError,
    EntityExtractionService,
)


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
        """Extracted entities carry the document id, type and a valid confidence.

        The service has no non-LLM ("pattern_only") method any more, so the
        extraction is driven by a fake ``llm_func`` at the LLM boundary.
        """
        llm = Mock(
            return_value=(
                '[{"text": "aspirin", "type": "DRUG", "confidence": 0.9},'
                ' {"text": "hypertension", "type": "DISEASE", "confidence": 0.85}]'
            )
        )
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
            llm_func=llm,
        )

        entities = service.extract_entities(self.test_document)

        assert {e.text for e in entities} == {"aspirin", "hypertension"}
        llm.assert_called_once()
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.source_document_id == self.test_document.id
            assert entity.entity_type in {"DRUG", "DISEASE"}
            assert 0 <= entity.confidence <= 1

    def test_no_llm_raises(self):
        """Without an llm_func extraction fails loudly instead of returning nothing."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )

        with pytest.raises(EntityExtractionError, match="llm_func"):
            service.extract_entities(self.test_document)

    @patch(
        "iris_vector_rag.services.entity_extraction.EntityExtractionService._call_llm"
    )
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
            llm_func=lambda prompt: (
                '[{"text": "aspirin", "type": "DRUG", "confidence": 0.9},'
                ' {"text": "hypertension", "type": "DISEASE", "confidence": 0.85}]'
            ),
        )

        # Mock the storage adapter write (one call per entity)
        service.storage_adapter.store_entity = Mock(return_value=True)

        result = service.process_document(self.test_document)

        assert result["document_id"] == self.test_document.id
        assert result["entities_extracted"] == 2
        assert result["stored"] is True
        assert result["stored_entities"] == 2
        assert len(result["entities"]) == 2
        assert service.storage_adapter.store_entity.call_count == 2

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
            llm_func=lambda prompt: "[]",
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
        """Entities below the confidence threshold are filtered out."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
            llm_func=lambda prompt: (
                '[{"text": "test", "type": "PERSON", "confidence": 0.5},'
                ' {"text": "John Doe", "type": "PERSON", "confidence": 0.95}]'
            ),
        )

        entities = service.extract_entities(self.test_document)

        # 0.5 is below the 0.7 threshold: only the confident entity remains
        assert [e.text for e in entities] == ["John Doe"]

    def test_entity_confidence_filtering_all_below_threshold(self):
        """When every entity is below the threshold, none are returned."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
            llm_func=lambda prompt: (
                '[{"text": "test", "type": "PERSON", "confidence": 0.5}]'
            ),
        )

        assert service.extract_entities(self.test_document) == []

    def _entities_and_relationships(self):
        entities = [
            Entity(
                text=f"entity{i}",
                entity_type="PERSON",
                confidence=0.9,
                start_offset=0,
                end_offset=5,
                source_document_id=self.test_document.id,
            )
            for i in range(5)
        ]
        relationships = [
            Relationship(
                source_entity_id=entities[i].id,
                target_entity_id=entities[i + 1].id,
                relationship_type="RELATED_TO",
                confidence=0.8,
                source_document_id=self.test_document.id,
            )
            for i in range(3)
        ]
        return entities, relationships

    def test_storage_adapter_integration(self):
        """Test integration with storage adapter."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )
        entities, relationships = self._entities_and_relationships()

        # Mock storage writes (the service writes item by item)
        service.storage_adapter.store_entity = Mock(return_value=True)
        service.storage_adapter.store_relationship = Mock(return_value=True)

        result = service.store_entities_and_relationships(entities, relationships)

        assert result == {"stored_entities": 5, "stored_relationships": 3}
        assert [
            c.args[0] for c in service.storage_adapter.store_entity.call_args_list
        ] == entities
        assert [
            c.args[0] for c in service.storage_adapter.store_relationship.call_args_list
        ] == relationships

    def test_storage_refusal_raises(self):
        """A write the adapter refuses is reported, not swallowed."""
        service = EntityExtractionService(
            config_manager=self.config_manager,
            connection_manager=self.connection_manager,
        )
        entities, relationships = self._entities_and_relationships()
        service.storage_adapter.store_entity = Mock(
            side_effect=[True, False, True, True, True]
        )
        service.storage_adapter.store_relationship = Mock(return_value=True)

        with pytest.raises(EntityExtractionError, match="Could not store 1"):
            service.store_entities_and_relationships(entities, relationships)


if __name__ == "__main__":
    pytest.main([__file__])
