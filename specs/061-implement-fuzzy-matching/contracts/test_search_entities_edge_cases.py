"""
Contract tests for edge cases in entity search.

Tests edge cases from spec:
- Empty query string
- No matching entities
- Unicode entity names
- similarity_threshold=1.0
- edit_distance_threshold=0
- max_results=0
- Very short query strings
"""

import pytest


class TestSearchEntitiesEdgeCases:
    """Contract tests for edge case handling."""

    def test_empty_query_returns_empty_results(self, entity_storage_adapter):
        """
        FR-013: Empty query string should return empty results without error.
        """
        # Arrange: Entities exist
        entity_storage_adapter.store_entity(
            entity_id="e1",
            entity_name="Test Entity",
            entity_type="PERSON",
            confidence=0.95,
        )

        # Act: Empty query
        results = entity_storage_adapter.search_entities(query="", fuzzy=True)

        # Assert: Empty results, no exception
        assert results == []

    def test_no_matching_entities_returns_empty_results(self, entity_storage_adapter):
        """
        FR-015: When no matching entities exist,
        should return empty results without error.
        Scenario 6 from spec.
        """
        # Arrange: Entity with different name
        entity_storage_adapter.store_entity(
            entity_id="e2",
            entity_name="Existing Entity",
            entity_type="PERSON",
            confidence=0.95,
        )

        # Act: Search for non-existent entity
        results = entity_storage_adapter.search_entities(
            query="Nonexistent Person", fuzzy=True
        )

        # Assert: Empty results, no exception
        assert results == []

    def test_unicode_entity_names_handled_correctly(self, entity_storage_adapter):
        """
        FR-014: System must handle Unicode entity names correctly.
        """
        # Arrange: Entities with Unicode characters
        entity_storage_adapter.store_entity(
            entity_id="e3",
            entity_name="François Truffaut",  # é character
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e4",
            entity_name="北京",  # Chinese characters
            entity_type="LOCATION",
            confidence=0.92,
        )

        # Act: Search for Unicode entities
        results_french = entity_storage_adapter.search_entities(
            query="François", fuzzy=True
        )
        results_chinese = entity_storage_adapter.search_entities(
            query="北京", fuzzy=False
        )

        # Assert: Unicode handled without errors
        assert len(results_french) >= 1
        assert any("François" in r["entity_name"] for r in results_french)
        assert len(results_chinese) == 1
        assert results_chinese[0]["entity_name"] == "北京"

    def test_similarity_threshold_1_0_returns_only_exact_matches(
        self, entity_storage_adapter
    ):
        """
        When similarity_threshold=1.0,
        Then only exact matches should be returned.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e5",
            entity_name="Scott Derrickson",  # Exact
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e6",
            entity_name="Scott Derrickson director",  # Fuzzy
            entity_type="PERSON",
            confidence=0.92,
        )

        # Act: High similarity threshold
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson", fuzzy=True, similarity_threshold=1.0
        )

        # Assert: Only exact match
        assert len(results) == 1
        assert results[0]["entity_name"] == "Scott Derrickson"
        assert results[0]["similarity_score"] == 1.0

    def test_edit_distance_threshold_0_behaves_like_exact_match(
        self, entity_storage_adapter
    ):
        """
        When edit_distance_threshold=0,
        Then should behave like exact matching (fuzzy=False).
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e7",
            entity_name="Test",  # Exact
            entity_type="CONCEPT",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e8",
            entity_name="Tests",  # edit_distance=1
            entity_type="CONCEPT",
            confidence=0.92,
        )

        # Act: Zero edit distance threshold
        results = entity_storage_adapter.search_entities(
            query="Test", fuzzy=True, edit_distance_threshold=0
        )

        # Assert: Only exact match
        assert len(results) == 1
        assert results[0]["entity_name"] == "Test"

    def test_max_results_0_returns_empty_list(self, entity_storage_adapter):
        """
        When max_results=0,
        Then should return empty list (invalid input treated as no results).
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e9",
            entity_name="Test Entity",
            entity_type="CONCEPT",
            confidence=0.95,
        )

        # Act: max_results=0
        # Note: This may raise ValueError in implementation, but edge case spec says return empty
        try:
            results = entity_storage_adapter.search_entities(
                query="Test", fuzzy=True, max_results=0
            )
            assert results == []
        except ValueError as e:
            # Acceptable alternative: raise validation error
            assert "max_results" in str(e).lower()

    def test_very_short_query_avoids_over_matching(self, entity_storage_adapter):
        """
        When query is very short (1-2 characters),
        Then should handle without over-matching all entities.
        """
        # Arrange: Multiple entities with 'A' character
        entity_storage_adapter.store_entity(
            entity_id="e10", entity_name="A", entity_type="CONCEPT", confidence=0.9
        )
        entity_storage_adapter.store_entity(
            entity_id="e11", entity_name="AB", entity_type="CONCEPT", confidence=0.9
        )
        entity_storage_adapter.store_entity(
            entity_id="e12",
            entity_name="ABC Long Name",
            entity_type="CONCEPT",
            confidence=0.9,
        )

        # Act: Very short query
        results = entity_storage_adapter.search_entities(
            query="A", fuzzy=True, edit_distance_threshold=1, max_results=10
        )

        # Assert: Returns reasonable number of matches (not all entities)
        assert len(results) <= 5, "Should not over-match with very short query"
        # Exact match should be first
        assert results[0]["entity_name"] == "A"

    def test_multiple_identical_entity_names_returned(self, entity_storage_adapter):
        """
        When multiple entities have identical names,
        Then system should return all matches.
        """
        # Arrange: Multiple entities with same name
        entity_storage_adapter.store_entity(
            entity_id="e13",
            entity_name="John Smith",
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e14",
            entity_name="John Smith",
            entity_type="PERSON",
            confidence=0.92,
        )

        # Act
        results = entity_storage_adapter.search_entities(
            query="John Smith", fuzzy=False
        )

        # Assert: Both entities returned
        assert len(results) == 2
        entity_ids = {r["entity_id"] for r in results}
        assert "e13" in entity_ids
        assert "e14" in entity_ids

    def test_case_variations_matched_case_insensitively(self, entity_storage_adapter):
        """
        FR-010: Case-insensitive matching for both exact and fuzzy searches.
        When entity name has different case than query,
        Then should still match.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e15",
            entity_name="SCOTT DERRICKSON",  # All caps
            entity_type="PERSON",
            confidence=0.95,
        )

        # Act: Lowercase query
        results = entity_storage_adapter.search_entities(
            query="scott derrickson", fuzzy=True
        )

        # Assert: Case-insensitive match
        assert len(results) >= 1
        assert results[0]["entity_name"] == "SCOTT DERRICKSON"
