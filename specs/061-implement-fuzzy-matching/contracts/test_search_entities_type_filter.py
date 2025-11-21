"""
Contract tests for entity type filtering in search results.

Tests acceptance scenario:
- Scenario 3: Entity type filtering
"""

import pytest


class TestSearchEntitiesTypeFilter:
    """Contract tests for entity type filtering."""

    def test_type_filter_returns_only_specified_types(self, entity_storage_adapter):
        """
        Scenario 3: Given entities of different types,
        When searching with type filter,
        Then returns only entities of specified types.
        """
        # Arrange: Entities of different types with same name pattern
        entity_storage_adapter.store_entity(
            entity_id="e1",
            entity_name="Scott Derrickson",
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e2",
            entity_name="Scott Derrickson Productions",
            entity_type="ORGANIZATION",
            confidence=0.92,
        )
        entity_storage_adapter.store_entity(
            entity_id="e3",
            entity_name="Scott Derrickson Studio",
            entity_type="LOCATION",
            confidence=0.88,
        )

        # Act: Search with PERSON filter only
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson",
            fuzzy=True,
            entity_types=["PERSON"],
        )

        # Assert: Only PERSON entities returned
        assert len(results) >= 1
        assert all(r["entity_type"] == "PERSON" for r in results)
        # Verify ORGANIZATION and LOCATION were filtered out
        entity_names = [r["entity_name"] for r in results]
        assert "Scott Derrickson" in entity_names
        assert "Scott Derrickson Productions" not in entity_names
        assert "Scott Derrickson Studio" not in entity_names

    def test_type_filter_accepts_multiple_types(self, entity_storage_adapter):
        """
        FR-007: User can filter by multiple entity types.
        Given entities of PERSON, ORGANIZATION, LOCATION types,
        When searching with filter ["PERSON", "ORGANIZATION"],
        Then returns only PERSON and ORGANIZATION entities.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e4",
            entity_name="John Smith",
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e5",
            entity_name="John Smith Corp",
            entity_type="ORGANIZATION",
            confidence=0.92,
        )
        entity_storage_adapter.store_entity(
            entity_id="e6",
            entity_name="John Smith Building",
            entity_type="LOCATION",
            confidence=0.88,
        )

        # Act: Filter by PERSON and ORGANIZATION
        results = entity_storage_adapter.search_entities(
            query="John Smith",
            fuzzy=True,
            entity_types=["PERSON", "ORGANIZATION"],
        )

        # Assert: PERSON and ORGANIZATION returned, LOCATION excluded
        assert len(results) >= 2
        returned_types = {r["entity_type"] for r in results}
        assert "PERSON" in returned_types
        assert "ORGANIZATION" in returned_types
        assert "LOCATION" not in returned_types

    def test_no_type_filter_returns_all_types(self, entity_storage_adapter):
        """
        When entity_types=None (default),
        Then returns entities of all types.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e7",
            entity_name="Universal Entity",
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e8",
            entity_name="Universal Entity Org",
            entity_type="ORGANIZATION",
            confidence=0.92,
        )

        # Act: No type filter
        results = entity_storage_adapter.search_entities(
            query="Universal", fuzzy=True, entity_types=None
        )

        # Assert: All types returned
        assert len(results) >= 2
        returned_types = {r["entity_type"] for r in results}
        assert "PERSON" in returned_types
        assert "ORGANIZATION" in returned_types
