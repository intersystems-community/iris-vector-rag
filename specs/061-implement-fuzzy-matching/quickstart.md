# Quickstart: Fuzzy Entity Matching

**Feature**: 061-implement-fuzzy-matching
**Date**: 2025-01-15
**Estimated Time**: 10 minutes

## Overview

This guide demonstrates how to use fuzzy entity matching in EntityStorageAdapter to match query entities to knowledge graph entities with descriptors and handle typos.

## Prerequisites

- iris-vector-rag installed with EntityStorageAdapter
- IRIS database running and populated with entities
- Python 3.11+

## Installation

```bash
# Install iris-vector-rag (if not already installed)
pip install iris-vector-rag

# Or install from source
cd iris-vector-rag-private
pip install -e .
```

## Basic Usage

### 1. Initialize EntityStorageAdapter

```python
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.services.storage import EntityStorageAdapter

# Initialize configuration and connection
config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)

# Create storage adapter
storage_adapter = EntityStorageAdapter(
    connection_manager=connection_manager,
    config=config_manager.get_all()
)
```

### 2. Exact Entity Search

```python
# Search for entity with exact name match
results = storage_adapter.search_entities(
    query="Scott Derrickson",
    fuzzy=False,  # Exact matching only
    max_results=10
)

print(f"Found {len(results)} exact matches")
for result in results:
    print(f"  - {result['entity_name']} ({result['entity_type']})")
```

**Expected Output**:
```
Found 1 exact matches
  - Scott Derrickson (PERSON)
```

### 3. Fuzzy Entity Search (Descriptors)

```python
# Search for entity that may have descriptors
results = storage_adapter.search_entities(
    query="Scott Derrickson",
    fuzzy=True,  # Enable fuzzy matching
    max_results=10
)

print(f"Found {len(results)} fuzzy matches")
for result in results:
    similarity = result.get('similarity_score', 1.0)
    print(f"  - {result['entity_name']} (similarity: {similarity:.2f})")
```

**Expected Output**:
```
Found 3 fuzzy matches
  - Scott Derrickson (similarity: 1.00)
  - Scott Derrickson director (similarity: 0.67)
  - director Scott Derrickson filmmaker (similarity: 0.51)
```

### 4. Fuzzy Search with Typo Handling

```python
# Search with typo ("Scot" instead of "Scott")
results = storage_adapter.search_entities(
    query="Scot Derrickson",  # Missing 't'
    fuzzy=True,
    edit_distance_threshold=2,  # Allow up to 2 character edits
    max_results=5
)

print(f"Found {len(results)} matches (with typo correction)")
for result in results:
    edit_dist = result.get('edit_distance', 0)
    similarity = result.get('similarity_score', 1.0)
    print(f"  - {result['entity_name']} (edit_distance: {edit_dist}, similarity: {similarity:.2f})")
```

**Expected Output**:
```
Found 2 matches (with typo correction)
  - Scott Derrickson (edit_distance: 1, similarity: 0.94)
  - Scott Derrickson director (edit_distance: 10, similarity: 0.63)
```

### 5. Entity Type Filtering

```python
# Search only for PERSON entities
results = storage_adapter.search_entities(
    query="Scott Derrickson",
    fuzzy=True,
    entity_types=["PERSON"],  # Filter by type
    max_results=10
)

print(f"Found {len(results)} PERSON entities")
for result in results:
    print(f"  - {result['entity_name']} ({result['entity_type']})")
```

**Expected Output**:
```
Found 2 PERSON entities
  - Scott Derrickson (PERSON)
  - Scott Derrickson director (PERSON)
```

### 6. Similarity Threshold Filtering

```python
# Only return high-quality matches (similarity >= 0.8)
results = storage_adapter.search_entities(
    query="Scott Derrickson",
    fuzzy=True,
    similarity_threshold=0.8,  # Filter low-quality matches
    max_results=10
)

print(f"Found {len(results)} high-quality matches")
for result in results:
    similarity = result['similarity_score']
    print(f"  - {result['entity_name']} (similarity: {similarity:.2f})")
```

**Expected Output**:
```
Found 2 high-quality matches
  - Scott Derrickson (similarity: 1.00)
  - Scot Derrickson (similarity: 0.94)
```

## Advanced Usage

### HippoRAG Pipeline Integration

```python
from iris_vector_rag.services.entity_extraction import EntityExtractionService

# Extract entities from user query
extraction_service = EntityExtractionService(llm_func=your_llm_func)
query_entities = extraction_service.extract_entities("Were Scott Derrickson and Ed Wood of the same nationality?")

# Match each query entity to knowledge graph entities
for entity_name in query_entities:
    # Fuzzy search to find graph entities with descriptors
    matches = storage_adapter.search_entities(
        query=entity_name,
        fuzzy=True,
        edit_distance_threshold=2,
        entity_types=["PERSON", "ORGANIZATION"],
        max_results=5
    )

    print(f"\nQuery entity: {entity_name}")
    print(f"Graph matches ({len(matches)}):")
    for match in matches:
        print(f"  → {match['entity_name']} (similarity: {match['similarity_score']:.2f})")
```

**Expected Output**:
```
Query entity: Scott Derrickson
Graph matches (2):
  → Scott Derrickson (similarity: 1.00)
  → Scott Derrickson director (similarity: 0.67)

Query entity: Ed Wood
Graph matches (2):
  → Ed Wood (similarity: 1.00)
  → Ed Wood filmmaker (similarity: 0.78)
```

### Batch Entity Matching

```python
# Match multiple query entities in batch
query_entities = ["Scott Derrickson", "Ed Wood", "Tim Burton"]

all_matches = {}
for entity_name in query_entities:
    matches = storage_adapter.search_entities(
        query=entity_name,
        fuzzy=True,
        max_results=3
    )
    all_matches[entity_name] = matches

# Process results
for query_entity, matches in all_matches.items():
    print(f"\n{query_entity}:")
    for match in matches:
        print(f"  - {match['entity_name']} (sim: {match['similarity_score']:.2f})")
```

## Configuration

### Default Configuration (config/default_config.yaml)

```yaml
entity_extraction:
  storage:
    entities_table: "RAG.Entities"
    relationships_table: "RAG.EntityRelationships"

    # Optional fuzzy search defaults
    fuzzy_search:
      default_edit_distance: 2      # Default edit distance threshold
      default_similarity: 0.0       # Default similarity threshold
      max_results_limit: 100        # Maximum allowed max_results
```

### Runtime Configuration Override

```python
# Override defaults at runtime
results = storage_adapter.search_entities(
    query="Scott Derrickson",
    fuzzy=True,
    edit_distance_threshold=1,  # Override: stricter threshold
    similarity_threshold=0.9,   # Override: higher quality bar
    max_results=5               # Override: fewer results
)
```

## Performance Considerations

**Exact Match**:
- Expected latency: < 10ms
- Uses indexed entity_name lookup
- No fuzzy matching overhead

**Fuzzy Match**:
- Expected latency: < 50ms for 100K entities
- Uses indexed columns + Levenshtein SQL function
- FETCH FIRST limits result set size

**Optimization Tips**:
1. Use exact matching when possible (fuzzy=False)
2. Limit max_results to reduce processing time
3. Apply entity_type filters early (indexed column)
4. Use appropriate edit_distance_threshold (lower = faster)

## Troubleshooting

### No Results Returned

**Problem**: Fuzzy search returns empty results
```python
results = storage_adapter.search_entities(query="Scott", fuzzy=True)
# results == []
```

**Solution**: Check edit_distance_threshold and similarity_threshold
```python
# Increase thresholds to allow more matches
results = storage_adapter.search_entities(
    query="Scott",
    fuzzy=True,
    edit_distance_threshold=3,    # More lenient
    similarity_threshold=0.0      # Accept all similarities
)
```

### Too Many Low-Quality Matches

**Problem**: Fuzzy search returns too many irrelevant results
```python
results = storage_adapter.search_entities(query="A", fuzzy=True)
# len(results) == 500 (too many)
```

**Solution**: Use similarity_threshold and reduce max_results
```python
results = storage_adapter.search_entities(
    query="A",
    fuzzy=True,
    similarity_threshold=0.8,  # Filter low-quality matches
    max_results=10              # Limit result count
)
```

### Exact Match Not Finding Entity

**Problem**: Exact match fails for entity with descriptor
```python
# Entity in database: "Scott Derrickson director"
results = storage_adapter.search_entities(query="Scott Derrickson", fuzzy=False)
# results == [] (exact match fails)
```

**Solution**: Use fuzzy matching to handle descriptors
```python
results = storage_adapter.search_entities(
    query="Scott Derrickson",
    fuzzy=True  # Enable fuzzy matching for descriptors
)
# results[0]['entity_name'] == "Scott Derrickson director"
```

## Next Steps

1. **Integration Testing**: Run contract tests to validate implementation
   ```bash
   pytest specs/061-implement-fuzzy-matching/contracts/ -v
   ```

2. **Performance Testing**: Validate <50ms requirement with 10K+ entities
   ```bash
   pytest tests/integration/test_fuzzy_entity_search_integration.py -v -k performance
   ```

3. **Production Deployment**: Follow standard deployment procedures
   - Run regression tests to ensure zero impact on existing methods
   - Monitor query latency in production
   - Tune edit_distance_threshold based on actual data

## References

- Feature Specification: [spec.md](./spec.md)
- Data Model: [data-model.md](./data-model.md)
- Contract Tests: [contracts/](./contracts/)
- IRIS iFind Documentation: InterSystems IRIS Full-Text Search Guide
