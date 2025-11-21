"""
Contract tests for result ranking in fuzzy entity search.

Tests functional requirement:
- FR-005: Results ranked with exact matches first, then by edit distance
"""

import pytest


class TestSearchEntitiesRanking:
    """Contract tests for search result ranking."""

    def test_exact_matches_appear_first(self, entity_storage_adapter):
        """
        FR-005: Exact matches must appear first in results,
        followed by closest fuzzy matches.
        """
        # Arrange: Mix of exact and fuzzy matches
        entity_storage_adapter.store_entity(
            entity_id="e1",
            entity_name="Scott Derrickson director",  # edit_distance=9
            entity_type="PERSON",
            confidence=0.92,
        )
        entity_storage_adapter.store_entity(
            entity_id="e2",
            entity_name="Scott Derrickson",  # Exact match (edit_distance=0)
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e3",
            entity_name="Scot Derrickson",  # Typo (edit_distance=1)
            entity_type="PERSON",
            confidence=0.88,
        )

        # Act
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson", fuzzy=True, max_results=10
        )

        # Assert: Ranking order correct
        assert len(results) == 3
        # Rank 1: Exact match
        assert results[0]["entity_name"] == "Scott Derrickson"
        assert results[0]["similarity_score"] == 1.0
        # Rank 2: Lowest edit distance (typo)
        assert results[1]["entity_name"] == "Scot Derrickson"
        assert results[1]["edit_distance"] == 1
        # Rank 3: Higher edit distance (descriptor)
        assert results[2]["entity_name"] == "Scott Derrickson director"
        assert results[2]["edit_distance"] > results[1]["edit_distance"]

    def test_ties_broken_by_name_length(self, entity_storage_adapter):
        """
        When multiple entities have same edit distance,
        shorter names should rank higher.
        """
        # Arrange: Entities with same edit distance but different lengths
        entity_storage_adapter.store_entity(
            entity_id="e4",
            entity_name="test entity",  # Shorter
            entity_type="CONCEPT",
            confidence=0.9,
        )
        entity_storage_adapter.store_entity(
            entity_id="e5",
            entity_name="test entity with long descriptor",  # Longer
            entity_type="CONCEPT",
            confidence=0.9,
        )

        # Act: Search that matches both (same edit distance from query)
        results = entity_storage_adapter.search_entities(
            query="test", fuzzy=True, max_results=10
        )

        # Assert: Shorter name ranks higher
        assert len(results) >= 2
        # Find the two entities in results
        entity4_idx = next(
            i for i, r in enumerate(results) if r["entity_id"] == "e4"
        )
        entity5_idx = next(
            i for i, r in enumerate(results) if r["entity_id"] == "e5"
        )
        assert entity4_idx < entity5_idx, "Shorter name should rank higher"

    def test_max_results_limits_returned_count(self, entity_storage_adapter):
        """
        FR-008: User can set max_results to limit returned entities.
        When more matches exist than max_results,
        Then only top-ranked entities are returned.
        """
        # Arrange: Many matching entities
        for i in range(20):
            entity_storage_adapter.store_entity(
                entity_id=f"e{i}",
                entity_name=f"Test Entity {i}",
                entity_type="CONCEPT",
                confidence=0.9,
            )

        # Act: Limit to 5 results
        results = entity_storage_adapter.search_entities(
            query="Test", fuzzy=True, max_results=5
        )

        # Assert: Only 5 results returned
        assert len(results) == 5
        # All results should have similarity scores
        assert all("similarity_score" in r for r in results)
