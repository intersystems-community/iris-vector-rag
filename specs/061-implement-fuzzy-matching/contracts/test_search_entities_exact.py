"""
Contract tests for exact entity matching.

Tests acceptance scenarios:
- Scenario 4: Exact matching (fuzzy=False) should not match descriptors
"""

import pytest


class TestSearchEntitiesExact:
    """Contract tests for exact entity name matching."""

    def test_exact_match_does_not_find_entity_with_descriptor(
        self, entity_storage_adapter
    ):
        """
        Scenario 4: Given entity "Scott Derrickson director",
        When searching "Scott Derrickson" with fuzzy=False,
        Then returns no results (exact match fails).
        """
        # Arrange: Entity with descriptor
        entity_storage_adapter.store_entity(
            entity_id="e1",
            entity_name="Scott Derrickson director",
            entity_type="PERSON",
            confidence=0.95,
        )

        # Act: Exact match search
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson", fuzzy=False
        )

        # Assert: No results (exact match fails)
        assert len(results) == 0, "Exact match should not find entity with descriptor"

    def test_exact_match_finds_identical_entity_name(self, entity_storage_adapter):
        """
        Given entity "Scott Derrickson",
        When searching "Scott Derrickson" with fuzzy=False,
        Then returns the entity with exact match.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e2",
            entity_name="Scott Derrickson",
            entity_type="PERSON",
            confidence=0.98,
        )

        # Act
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson", fuzzy=False
        )

        # Assert
        assert len(results) == 1
        assert results[0]["entity_name"] == "Scott Derrickson"
        assert results[0]["entity_id"] == "e2"

    def test_exact_match_is_case_insensitive(self, entity_storage_adapter):
        """
        FR-010: Exact match should be case-insensitive.
        Given entity "Scott Derrickson",
        When searching "scott derrickson" with fuzzy=False,
        Then returns the entity.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e3",
            entity_name="Scott Derrickson",
            entity_type="PERSON",
            confidence=0.95,
        )

        # Act
        results = entity_storage_adapter.search_entities(
            query="scott derrickson",  # Lowercase query
            fuzzy=False,
        )

        # Assert
        assert len(results) == 1
        assert results[0]["entity_name"] == "Scott Derrickson"

    def test_exact_match_returns_all_required_fields(self, entity_storage_adapter):
        """
        FR-004: System must return entity_id, entity_name, entity_type, confidence.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e4",
            entity_name="Test Entity",
            entity_type="ORGANIZATION",
            confidence=0.92,
        )

        # Act
        results = entity_storage_adapter.search_entities(
            query="Test Entity", fuzzy=False
        )

        # Assert
        assert len(results) == 1
        result = results[0]
        assert "entity_id" in result
        assert "entity_name" in result
        assert "entity_type" in result
        assert "confidence" in result
        assert result["entity_id"] == "e4"
        assert result["entity_type"] == "ORGANIZATION"
        assert result["confidence"] == 0.92
