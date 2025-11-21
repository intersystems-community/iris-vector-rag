"""
Contract tests for fuzzy entity matching with Levenshtein distance and substring matching.

Tests acceptance scenarios:
- Scenario 1: Fuzzy matching finds entity with descriptor
- Scenario 2: Multiple entities ranked by similarity
- Scenario 7: Typo handling (missing character)
- Scenario 8: Spelling variations (color vs colour)
"""

import pytest


class TestSearchEntitiesFuzzy:
    """Contract tests for fuzzy entity matching."""

    def test_fuzzy_match_finds_entity_with_descriptor(self, entity_storage_adapter):
        """
        Scenario 1: Given entity "Scott Derrickson director",
        When searching "Scott Derrickson" with fuzzy=True,
        Then returns the entity with similarity score.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e1",
            entity_name="Scott Derrickson director",
            entity_type="PERSON",
            confidence=0.95,
        )

        # Act
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson", fuzzy=True
        )

        # Assert
        assert len(results) >= 1, "Fuzzy match should find entity with descriptor"
        assert results[0]["entity_name"] == "Scott Derrickson director"
        assert "similarity_score" in results[0], "Fuzzy results must have similarity_score"
        assert "edit_distance" in results[0], "Fuzzy results must have edit_distance"
        assert 0.0 <= results[0]["similarity_score"] <= 1.0

    def test_fuzzy_match_ranks_by_similarity(self, entity_storage_adapter):
        """
        Scenario 2: Given multiple entities, when searching with fuzzy=True,
        then returns all entities ranked by similarity (exact match first).
        """
        # Arrange: Three entities with varying similarity
        entity_storage_adapter.store_entity(
            entity_id="e1",
            entity_name="Scott Derrickson",  # Exact match
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e2",
            entity_name="Scott Derrickson director",  # With descriptor
            entity_type="PERSON",
            confidence=0.92,
        )
        entity_storage_adapter.store_entity(
            entity_id="e3",
            entity_name="director Scott Derrickson filmmaker",  # Multiple descriptors
            entity_type="PERSON",
            confidence=0.88,
        )

        # Act
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson", fuzzy=True, max_results=10
        )

        # Assert: All three entities returned, ranked by similarity
        assert len(results) == 3
        # Exact match should be first
        assert results[0]["entity_name"] == "Scott Derrickson"
        assert results[0]["similarity_score"] == 1.0
        # Others ranked by edit distance (lower edit distance = higher rank)
        assert results[1]["similarity_score"] > results[2]["similarity_score"]

    def test_fuzzy_match_handles_typos(self, entity_storage_adapter):
        """
        Scenario 7: Given entity "Scott Derrickson",
        When searching "Scot Derrickson" (missing 't') with fuzzy=True,
        Then returns "Scott Derrickson" as fuzzy match.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e4",
            entity_name="Scott Derrickson",
            entity_type="PERSON",
            confidence=0.95,
        )

        # Act: Search with typo
        results = entity_storage_adapter.search_entities(
            query="Scot Derrickson",  # Missing 't'
            fuzzy=True,
            edit_distance_threshold=2,
        )

        # Assert: Fuzzy match finds correct entity
        assert len(results) >= 1
        assert results[0]["entity_name"] == "Scott Derrickson"
        # Edit distance should be 1 (one character missing)
        assert results[0]["edit_distance"] == 1
        # Similarity score should be high (only 1 character difference)
        assert results[0]["similarity_score"] >= 0.9

    def test_fuzzy_match_handles_spelling_variations(self, entity_storage_adapter):
        """
        Scenario 8: Given entities with spelling variations,
        When searching with fuzzy=True and appropriate edit distance,
        Then returns matches accounting for variations.
        """
        # Arrange: Entities with spelling variations
        entity_storage_adapter.store_entity(
            entity_id="e5",
            entity_name="color",
            entity_type="CONCEPT",
            confidence=0.9,
        )
        entity_storage_adapter.store_entity(
            entity_id="e6",
            entity_name="colour",  # British spelling
            entity_type="CONCEPT",
            confidence=0.9,
        )

        # Act: Search for American spelling
        results = entity_storage_adapter.search_entities(
            query="color", fuzzy=True, edit_distance_threshold=2
        )

        # Assert: Both spellings found
        assert len(results) == 2
        # Exact match first
        assert results[0]["entity_name"] == "color"
        assert results[0]["similarity_score"] == 1.0
        # British spelling second (edit_distance=1)
        assert results[1]["entity_name"] == "colour"
        assert results[1]["edit_distance"] == 1

    def test_fuzzy_match_respects_edit_distance_threshold(self, entity_storage_adapter):
        """
        FR-018: User can specify edit distance threshold to filter matches.
        """
        # Arrange: Entities with varying edit distances
        entity_storage_adapter.store_entity(
            entity_id="e7",
            entity_name="test",  # edit_distance=0 from "test"
            entity_type="CONCEPT",
            confidence=0.9,
        )
        entity_storage_adapter.store_entity(
            entity_id="e8",
            entity_name="tests",  # edit_distance=1 from "test"
            entity_type="CONCEPT",
            confidence=0.9,
        )
        entity_storage_adapter.store_entity(
            entity_id="e9",
            entity_name="testing",  # edit_distance=3 from "test"
            entity_type="CONCEPT",
            confidence=0.9,
        )

        # Act: Search with edit_distance_threshold=1
        results = entity_storage_adapter.search_entities(
            query="test", fuzzy=True, edit_distance_threshold=1
        )

        # Assert: Only entities within threshold
        assert len(results) == 2  # "test" and "tests", not "testing"
        entity_names = [r["entity_name"] for r in results]
        assert "test" in entity_names
        assert "tests" in entity_names
        assert "testing" not in entity_names

    def test_fuzzy_match_respects_similarity_threshold(self, entity_storage_adapter):
        """
        FR-009: User can set similarity threshold to filter low-quality matches.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            entity_id="e10",
            entity_name="Scott Derrickson",  # similarity=1.0
            entity_type="PERSON",
            confidence=0.95,
        )
        entity_storage_adapter.store_entity(
            entity_id="e11",
            entity_name="Scott Derrickson director filmmaker actor",  # Lower similarity
            entity_type="PERSON",
            confidence=0.85,
        )

        # Act: Search with high similarity threshold
        results = entity_storage_adapter.search_entities(
            query="Scott Derrickson", fuzzy=True, similarity_threshold=0.8
        )

        # Assert: Only high-similarity matches returned
        assert all(r["similarity_score"] >= 0.8 for r in results)
        # Low-similarity match should be filtered out
        entity_names = [r["entity_name"] for r in results]
        assert "Scott Derrickson" in entity_names
