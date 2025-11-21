# Data Model: Fuzzy Entity Matching

**Feature**: 061-implement-fuzzy-matching
**Date**: 2025-01-15
**Status**: Complete

## Overview

This feature adds fuzzy entity search capability to EntityStorageAdapter without requiring database schema changes. It leverages existing RAG.Entities table structure and adds a new search method with configurable matching parameters.

## Entities

### EntitySearchQuery (Input Parameter Object)

Represents parameters for an entity search request.

**Fields**:
| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| query | str | Yes | N/A | len(query) >= 0 | Entity name to search for (empty string returns empty results) |
| fuzzy | bool | No | False | N/A | Enable fuzzy matching (Levenshtein + substring) |
| edit_distance_threshold | int | No | 2 | 0 <= value <= 3 | Maximum edit distance for fuzzy matches (0=exact only) |
| similarity_threshold | float | No | 0.0 | 0.0 <= value <= 1.0 | Minimum similarity score (0.0=all, 1.0=exact only) |
| entity_types | List[str] | No | None | Valid entity types | Filter by entity type (e.g., ["PERSON", "ORGANIZATION"]) |
| max_results | int | No | 10 | 1 <= value <= 100 | Maximum number of results to return |

**Example**:
```python
# Exact match query
query = {
    "query": "Scott Derrickson",
    "fuzzy": False,
    "max_results": 10
}

# Fuzzy match query with filters
query = {
    "query": "Scott Derrickson",
    "fuzzy": True,
    "edit_distance_threshold": 2,
    "similarity_threshold": 0.5,
    "entity_types": ["PERSON"],
    "max_results": 10
}
```

### EntitySearchResult (Output Data Object)

Represents a single matched entity from search results.

**Fields**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| entity_id | str | Yes | Unique entity identifier (from RAG.Entities.entity_id) |
| entity_name | str | Yes | Entity name (from RAG.Entities.entity_name) |
| entity_type | str | Yes | Entity type classification (from RAG.Entities.entity_type) |
| confidence | float | Yes | Extraction confidence score (from RAG.Entities.confidence) |
| similarity_score | float | Fuzzy only | Normalized similarity score 0.0-1.0 (1.0=perfect match) |
| edit_distance | int | Fuzzy only | Levenshtein distance (number of character edits) |
| source_document | str | Optional | Source document ID (from RAG.Entities.source_document) |
| metadata | dict | Optional | Additional entity metadata |

**Similarity Score Calculation**:
```
similarity_score = 1 - (edit_distance / max(len(query), len(entity_name)))

Examples:
- "Scott" vs "Scott" → edit_distance=0 → similarity=1.0
- "Scot" vs "Scott" → edit_distance=1 → similarity=0.8
- "Scott Derrickson" vs "Scott Derrickson director" → edit_distance=9 → similarity=0.67
```

**Example**:
```python
# Exact match result
{
    "entity_id": "e123",
    "entity_name": "Scott Derrickson",
    "entity_type": "PERSON",
    "confidence": 0.95,
    "source_document": "doc456"
}

# Fuzzy match result (includes similarity_score and edit_distance)
{
    "entity_id": "e124",
    "entity_name": "Scott Derrickson director",
    "entity_type": "PERSON",
    "confidence": 0.92,
    "similarity_score": 0.67,
    "edit_distance": 9,
    "source_document": "doc457"
}
```

## Database Schema

### RAG.Entities Table (Existing - No Changes Required)

The feature uses the existing RAG.Entities table without modifications.

**Columns Used**:
| Column | Type | Index | Description |
|--------|------|-------|-------------|
| entity_id | VARCHAR | Primary Key | Unique entity identifier |
| entity_name | VARCHAR | Indexed | Entity name (may include descriptors) |
| entity_type | VARCHAR | Indexed | Entity type classification |
| confidence | FLOAT | No | Extraction confidence score |
| source_document | VARCHAR | No | Source document reference |
| metadata | VARCHAR (JSON) | No | Additional entity metadata |

**Indexes Leveraged**:
- entity_name: B-tree index for LIKE and equality queries
- entity_type: B-tree index for type filtering

**SQL Functions Used**:
- `$SYSTEM.SQL.Functions.LEVENSHTEIN(str1, str2)`: Calculate edit distance
- `LOWER(str)`: Case-insensitive matching
- `GREATEST(val1, val2)`: Maximum value for similarity calculation
- `LENGTH(str)`: String length for normalization

## API Contract

### search_entities() Method

**Signature**:
```python
def search_entities(
    self,
    query: str,
    fuzzy: bool = False,
    edit_distance_threshold: int = 2,
    similarity_threshold: float = 0.0,
    entity_types: Optional[List[str]] = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for entities by name with exact or fuzzy matching.

    Args:
        query: Entity name to search for
        fuzzy: Enable fuzzy matching (substring + Levenshtein)
        edit_distance_threshold: Maximum edit distance (0-3)
        similarity_threshold: Minimum similarity score (0.0-1.0)
        entity_types: Optional list of entity types to filter by
        max_results: Maximum number of results (1-100)

    Returns:
        List of matching entities with similarity scores (if fuzzy=True)

    Raises:
        ValueError: If parameters are invalid (e.g., threshold out of range)
        DatabaseError: If database query fails
    """
```

**Parameter Validation**:
```python
# query: Any string allowed (empty string returns empty list)
if not isinstance(query, str):
    raise ValueError("query must be a string")

# fuzzy: Boolean only
if not isinstance(fuzzy, bool):
    raise ValueError("fuzzy must be a boolean")

# edit_distance_threshold: 0-3 range
if not 0 <= edit_distance_threshold <= 3:
    raise ValueError("edit_distance_threshold must be between 0 and 3")

# similarity_threshold: 0.0-1.0 range
if not 0.0 <= similarity_threshold <= 1.0:
    raise ValueError("similarity_threshold must be between 0.0 and 1.0")

# entity_types: List of strings or None
if entity_types is not None and not isinstance(entity_types, list):
    raise ValueError("entity_types must be a list of strings or None")

# max_results: 1-100 range
if not 1 <= max_results <= 100:
    raise ValueError("max_results must be between 1 and 100")
```

## Result Ranking Strategy

Results are ranked using a three-tier ordering:

1. **Exact matches first**: `entity_name == query` (case-insensitive)
2. **Then by edit distance**: Lower edit distance = higher rank
3. **Then by name length**: Shorter names ranked higher (breaks ties)

**SQL ORDER BY Clause**:
```sql
ORDER BY
    CASE WHEN LOWER(entity_name) = LOWER(?) THEN 0 ELSE 1 END,  -- Tier 1: Exact matches
    edit_distance ASC,                                            -- Tier 2: Lower edit distance
    LENGTH(entity_name) ASC                                       -- Tier 3: Shorter names
```

**Example Ranking**:
Given query "Scott Derrickson" with fuzzy=True:

| Rank | Entity Name | Edit Distance | Similarity Score | Reasoning |
|------|-------------|---------------|------------------|-----------|
| 1 | Scott Derrickson | 0 | 1.0 | Exact match (tier 1) |
| 2 | Scot Derrickson | 1 | 0.94 | Typo (tier 2, edit_distance=1) |
| 3 | Scott Derrickson director | 9 | 0.67 | Descriptor (tier 2, edit_distance=9) |
| 4 | director Scott Derrickson filmmaker | 18 | 0.51 | Multiple descriptors (tier 2, edit_distance=18) |

## State Transitions

No state transitions - search is a stateless read operation.

## Edge Cases

| Edge Case | Behavior | Rationale |
|-----------|----------|-----------|
| Empty query string | Return empty list | FR-013 requirement |
| No matching entities | Return empty list | FR-015 requirement |
| Unicode entity names | Handle correctly | FR-014 requirement |
| similarity_threshold=1.0 | Only exact matches | Logical behavior |
| edit_distance_threshold=0 | Only exact matches | Same as fuzzy=False |
| max_results=0 | Return empty list | Invalid input, treated as no results |
| Very short query (1-2 chars) | Return limited results | Avoid over-matching |
| Multiple identical entity_name | Return all with same rank | No de-duplication |
| Case variations | Case-insensitive | FR-010 requirement |

## Performance Considerations

**Query Performance**:
- Exact match: <10ms (indexed entity_name lookup)
- Fuzzy match: <50ms for 100K entities (indexed + FETCH FIRST optimization)

**Index Usage**:
- entity_name: Used for LIKE and equality queries
- entity_type: Used for filtering

**Optimization Notes**:
- `FETCH FIRST {max_results} ROWS ONLY` limits result set size
- `LOWER()` for case-insensitive matching (index-compatible)
- Entity type filter applied in WHERE clause (indexed)
- No table scans for typical queries

## Integration Points

**EntityStorageAdapter Integration**:
```python
# Existing methods (no changes)
store_entity(entity: Entity) -> bool
get_entities_by_document(doc_id: str) -> List[Entity]
get_entities_by_type(entity_type: str) -> List[Entity]

# New method (added)
search_entities(...) -> List[Dict[str, Any]]
```

**HippoRAG Pipeline Integration**:
```python
# Example usage in HippoRAG query expansion
query_entities = extract_entities_from_query(user_query)
for entity_name in query_entities:
    # Fuzzy search to match query entities to graph entities
    matches = storage_adapter.search_entities(
        query=entity_name,
        fuzzy=True,
        edit_distance_threshold=2,
        entity_types=["PERSON", "ORGANIZATION"],
        max_results=5
    )
    # Use matches for synonym expansion and PPR
    graph_entity_ids = [m["entity_id"] for m in matches]
```

## Configuration

**Default Configuration** (config/default_config.yaml):
```yaml
entity_extraction:
  storage:
    # Existing config (unchanged)
    entities_table: "RAG.Entities"
    relationships_table: "RAG.EntityRelationships"

    # New fuzzy search defaults (optional)
    fuzzy_search:
      default_edit_distance: 2  # Default edit distance threshold
      default_similarity: 0.0   # Default similarity threshold
      max_results_limit: 100    # Maximum allowed max_results value
```

## Backward Compatibility

**No Breaking Changes**:
- New method added to EntityStorageAdapter (existing methods unchanged)
- Existing methods continue to work identically
- No database schema changes required
- Optional configuration section (uses defaults if missing)

**Zero Regression Guarantee**:
- All existing EntityStorageAdapter tests must pass
- New method does not modify any existing behavior
- Feature can be deployed without affecting existing pipelines
