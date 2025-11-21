"""
Contract tests for fuzzy entity search in EntityStorageAdapter.

Tests Feature 061: Fuzzy Entity Matching for EntityStorageAdapter
These contract tests define the expected behavior of the search_entities() method
before implementation (TDD approach).

Test Coverage:
- Exact matching (fuzzy=False) - 4 test cases
- Fuzzy matching with Levenshtein distance - 7 test cases
- Entity type filtering - 3 test cases
- Result ranking - 3 test cases
- Edge cases - 9 test cases

Total: 29 test cases across 5 test classes

IMPORTANT: These tests are expected to FAIL until search_entities() is implemented.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Entity, EntityTypes
from iris_vector_rag.services.storage import EntityStorageAdapter


def create_test_entity(entity_id: str, entity_name: str, entity_type: str, confidence: float = 0.95) -> Entity:
    """
    Helper function to create Entity objects for testing.

    Maps test parameters to Entity model fields:
    - entity_name → entity.text (stored as entity_name in database)
    - entity_type → entity.entity_type
    - entity_id → entity.id
    """
    return Entity(
        id=entity_id,
        text=entity_name,  # entity.text is stored as entity_name in DB
        entity_type=entity_type,
        confidence=confidence,
        start_offset=0,
        end_offset=len(entity_name),
        source_document_id="TEST-DOC-CONTRACT",
        metadata={}
    )


@pytest.fixture
def connection_manager():
    """Get ConnectionManager for IRIS database."""
    config_manager = ConfigurationManager()
    return ConnectionManager(config_manager)


@pytest.fixture
def entity_storage_adapter(connection_manager):
    """
    Get EntityStorageAdapter instance for testing.

    This fixture creates a fresh adapter instance for each test with standard
    configuration for entity and relationship storage.
    """
    config = {
        "entity_extraction": {
            "storage": {
                "entities_table": "RAG.Entities",
                "relationships_table": "RAG.EntityRelationships",
                "embeddings_table": "RAG.EntityEmbeddings",
            }
        }
    }
    adapter = EntityStorageAdapter(connection_manager, config)

    # Clean up test data before each test
    # This ensures tests start with a clean state
    conn = connection_manager.get_connection()
    try:
        cursor = conn.cursor()
        try:
            # Delete test entities by multiple criteria:
            # 1. ID patterns: TEST-*, e0-e99
            # 2. Source document: TEST-DOC-CONTRACT
            # 3. Entity names containing "test" (case-insensitive)
            cursor.execute(
                f"DELETE FROM {config['entity_extraction']['storage']['entities_table']} "
                "WHERE entity_id LIKE 'TEST-%' "
                "OR (entity_id LIKE 'e%' AND LENGTH(entity_id) <= 3) "
                "OR source_doc_id = 'TEST-DOC-CONTRACT' "
                "OR LOWER(entity_name) LIKE '%test%'"
            )
            conn.commit()
        finally:
            cursor.close()
    except Exception:
        pass

    yield adapter

    # Cleanup after test
    conn = connection_manager.get_connection()
    try:
        cursor = conn.cursor()
        try:
            # Delete test entities (same criteria as setup)
            cursor.execute(
                f"DELETE FROM {config['entity_extraction']['storage']['entities_table']} "
                "WHERE entity_id LIKE 'TEST-%' "
                "OR (entity_id LIKE 'e%' AND LENGTH(entity_id) <= 3) "
                "OR source_doc_id = 'TEST-DOC-CONTRACT' "
                "OR LOWER(entity_name) LIKE '%test%'"
            )
            conn.commit()
        finally:
            cursor.close()
    except Exception:
        pass


# ============================================================================
# Test Class 1: Exact Entity Matching (4 test cases)
# ============================================================================


class TestSearchEntitiesExact:
    """Contract tests for exact entity name matching (fuzzy=False)."""

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
            create_test_entity(
                entity_id="e1",
                entity_name="Scott Derrickson director",
                entity_type="PERSON",
                confidence=0.95
            )
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
            create_test_entity(
                entity_id="e2",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.98
            )
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
            create_test_entity(
                entity_id="e3",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
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
            create_test_entity(
                entity_id="e4",
                entity_name="Test Entity",
                entity_type="ORGANIZATION",
                confidence=0.92
            )
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
        # Use approximate equality for confidence (database may normalize values)
        assert abs(result["confidence"] - 0.92) < 0.1, f"Expected confidence ≈0.92, got {result['confidence']}"


# ============================================================================
# Test Class 2: Fuzzy Matching with Levenshtein (7 test cases)
# ============================================================================


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
            create_test_entity(
                entity_id="e1",
                entity_name="Scott Derrickson director",
                entity_type="PERSON",
                confidence=0.95
            )
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
            create_test_entity(
                entity_id="e1",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e2",
                entity_name="Scott Derrickson director",
                entity_type="PERSON",
                confidence=0.92
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e3",
                entity_name="director Scott Derrickson filmmaker",
                entity_type="PERSON",
                confidence=0.88
            )
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
            create_test_entity(
                entity_id="e4",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
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
            create_test_entity(
                entity_id="e5",
                entity_name="color",
                entity_type="CONCEPT",
                confidence=0.9
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e6",
                entity_name="colour",
                entity_type="CONCEPT",
                confidence=0.9
            )
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
            create_test_entity(
                entity_id="e7",
                entity_name="test",
                entity_type="CONCEPT",
                confidence=0.9
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e8",
                entity_name="tests",
                entity_type="CONCEPT",
                confidence=0.9
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e9",
                entity_name="testing",
                entity_type="CONCEPT",
                confidence=0.9
            )
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
            create_test_entity(
                entity_id="e10",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e11",
                entity_name="Scott Derrickson director filmmaker actor",
                entity_type="PERSON",
                confidence=0.85
            )
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

    def test_fuzzy_match_is_case_insensitive(self, entity_storage_adapter):
        """
        FR-010: Fuzzy matching should be case-insensitive.
        """
        # Arrange
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e12",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
        )

        # Act: Lowercase query
        results = entity_storage_adapter.search_entities(
            query="scott derrickson", fuzzy=True
        )

        # Assert: Case-insensitive match
        assert len(results) >= 1
        assert results[0]["entity_name"] == "Scott Derrickson"


# ============================================================================
# Test Class 3: Entity Type Filtering (3 test cases)
# ============================================================================


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
            create_test_entity(
                entity_id="e1",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e2",
                entity_name="Scott Derrickson Productions",
                entity_type="ORGANIZATION",
                confidence=0.92
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e3",
                entity_name="Scott Derrickson Studio",
                entity_type="LOCATION",
                confidence=0.88
            )
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
            create_test_entity(
                entity_id="e4",
                entity_name="John Smith",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e5",
                entity_name="John Smith Corp",
                entity_type="ORGANIZATION",
                confidence=0.92
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e6",
                entity_name="John Smith Building",
                entity_type="LOCATION",
                confidence=0.88
            )
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
            create_test_entity(
                entity_id="e7",
                entity_name="Universal Entity",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e8",
                entity_name="Universal Entity Org",
                entity_type="ORGANIZATION",
                confidence=0.92
            )
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


# ============================================================================
# Test Class 4: Result Ranking (3 test cases)
# ============================================================================


class TestSearchEntitiesRanking:
    """Contract tests for search result ranking."""

    def test_exact_matches_appear_first(self, entity_storage_adapter):
        """
        FR-005: Exact matches must appear first in results,
        followed by closest fuzzy matches.
        """
        # Arrange: Mix of exact and fuzzy matches
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e1",
                entity_name="Scott Derrickson director",
                entity_type="PERSON",
                confidence=0.92
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e2",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e3",
                entity_name="Scot Derrickson",
                entity_type="PERSON",
                confidence=0.88
            )
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
            create_test_entity(
                entity_id="e4",
                entity_name="test entity",
                entity_type="CONCEPT",
                confidence=0.9
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e5",
                entity_name="test entity with long descriptor",
                entity_type="CONCEPT",
                confidence=0.9
            )
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
                create_test_entity(
                    entity_id=f"e{i}",
                    entity_name=f"Test Entity {i}",
                    entity_type="CONCEPT",
                    confidence=0.9
                )
            )

        # Act: Limit to 5 results
        results = entity_storage_adapter.search_entities(
            query="Test", fuzzy=True, max_results=5
        )

        # Assert: Only 5 results returned
        assert len(results) == 5
        # All results should have similarity scores
        assert all("similarity_score" in r for r in results)


# ============================================================================
# Test Class 5: Edge Cases (9 test cases)
# ============================================================================


class TestSearchEntitiesEdgeCases:
    """Contract tests for edge case handling."""

    def test_empty_query_returns_empty_results(self, entity_storage_adapter):
        """
        FR-013: Empty query string should return empty results without error.
        """
        # Arrange: Entities exist
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e1",
                entity_name="Test Entity",
                entity_type="PERSON",
                confidence=0.95
            )
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
            create_test_entity(
                entity_id="e2",
                entity_name="Existing Entity",
                entity_type="PERSON",
                confidence=0.95
            )
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
            create_test_entity(
                entity_id="e3",
                entity_name="François Truffaut",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e4",
                entity_name="北京",
                entity_type="LOCATION",
                confidence=0.92
            )
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
            create_test_entity(
                entity_id="e5",
                entity_name="Scott Derrickson",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e6",
                entity_name="Scott Derrickson director",
                entity_type="PERSON",
                confidence=0.92
            )
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
            create_test_entity(
                entity_id="e7",
                entity_name="Test",
                entity_type="CONCEPT",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e8",
                entity_name="Tests",
                entity_type="CONCEPT",
                confidence=0.92
            )
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
            create_test_entity(
                entity_id="e9",
                entity_name="Test Entity",
                entity_type="CONCEPT",
                confidence=0.95
            )
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
            create_test_entity(
                entity_id="e10",
                entity_name="A",
                entity_type="CONCEPT",
                confidence=0.9
        
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e11",
                entity_name="AB",
                entity_type="CONCEPT",
                confidence=0.9
        
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e12",
                entity_name="ABC Long Name",
                entity_type="CONCEPT",
                confidence=0.9
            )
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
            create_test_entity(
                entity_id="e13",
                entity_name="John Smith",
                entity_type="PERSON",
                confidence=0.95
            )
        )
        entity_storage_adapter.store_entity(
            create_test_entity(
                entity_id="e14",
                entity_name="John Smith",
                entity_type="PERSON",
                confidence=0.92
            )
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
            create_test_entity(
                entity_id="e15",
                entity_name="SCOTT DERRICKSON",
                entity_type="PERSON",
                confidence=0.95
            )
        )

        # Act: Lowercase query
        results = entity_storage_adapter.search_entities(
            query="scott derrickson", fuzzy=True
        )

        # Assert: Case-insensitive match
        assert len(results) >= 1
        assert results[0]["entity_name"] == "SCOTT DERRICKSON"
