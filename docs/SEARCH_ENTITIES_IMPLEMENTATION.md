# search_entities Implementation Summary

## Overview

This document summarizes the implementation of the `search_entities` method in `EntityStorageAdapter` to fix hipporag2-pipeline F1=0.000 scoring issue.

## Problem Statement

hipporag2-pipeline was failing with F1=0.000 score due to three critical errors:

1. **Missing `search_entities` method**: `AttributeError: 'EntityStorageAdapter' object has no attribute 'search_entities'`
2. **Foreign key validation failures**: 20 missing entity IDs, 30 orphaned relationships
3. **PPR connection type mismatch**: Fixed upstream in iris-vector-graph 1.1.7

## Solution Implemented

### 1. search_entities Method (iris_vector_rag/services/storage.py:526-631)

**Signature**:
```python
def search_entities(
    self,
    query: str,
    fuzzy: bool = True,
    edit_distance_threshold: int = 2,
    max_results: int = 10,
    entity_types: Optional[List[str]] = None,
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]
```

**Implementation**:
- Uses **SQL LIKE** with `LOWER()` for case-insensitive substring matching
- Returns **List[Dict[str, Any]]** (not Entity objects) for hipporag2 compatibility
- Supports entity type filtering and confidence thresholds
- Orders results by confidence DESC

**SQL Query**:
```sql
SELECT entity_id, entity_name, entity_type, source_doc_id, description, confidence
FROM RAG.Entities
WHERE LOWER(entity_name) LIKE LOWER('%query%')
AND confidence >= min_confidence
ORDER BY confidence DESC
FETCH FIRST max_results ROWS ONLY
```

**Return Format**:
```python
{
    "entity_id": "Scott Derrickson director",
    "entity_name": "Scott Derrickson director",
    "entity_type": "PERSON",
    "source_doc_id": "doc-123",
    "description": "American filmmaker and director",
    "confidence": 0.95,
    "similarity_score": 1.0  # Always 1.0 for LIKE-based matching
}
```

### 2. Enhanced Logging (iris_vector_rag/services/batch_entity_processor.py)

**Lines 112-117**: Log entity IDs being stored
```python
entity_ids_to_store = [str(e.id) for e in entities]
logger.debug(
    f"Preparing to store {len(entities)} entities. "
    f"First 5 IDs: {entity_ids_to_store[:5]}"
)
```

**Lines 319-332**: Enhanced foreign key validation error logging
```python
missing_ids_list = sorted(list(missing_ids))[:10]
logger.error(
    f"Foreign key validation failed: {len(missing_ids)} missing entity IDs, "
    f"{orphaned_count} orphaned relationships"
)
logger.error(
    f"Sample missing entity IDs (first 10): {missing_ids_list}"
)
```

## Capabilities and Limitations

### What Works ✅

1. **Case-insensitive matching**: "scott" matches "Scott" ✅
2. **Substring matching**: "Scott" matches "Scott Derrickson" ✅
3. **Partial name matching**: "Derrickson" matches "Scott Derrickson director" ✅
4. **Entity type filtering**: Can filter by PERSON, ORGANIZATION, etc.
5. **Confidence thresholds**: Min confidence filter supported
6. **Result limiting**: Configurable max_results parameter

### What Doesn't Work ❌

1. **Edit distance tolerance**: "Scot" does NOT match "Scott" ❌
2. **Typo tolerance**: "banona" does NOT match "banana" ❌
3. **Levenshtein distance**: `edit_distance_threshold` parameter is ignored
4. **True fuzzy matching**: `fuzzy` parameter accepted but not implemented

## Why LIKE Instead of True Fuzzy Matching?

### Short Answer

**SQL LIKE requires no additional setup**, whereas true Levenshtein-based fuzzy matching requires **iFind (IRIS Full-Text Search) indexes**, which need:

1. Schema changes (CREATE INDEX ... FOR %iFind.Index.Basic)
2. Table-level iFind configuration
3. SchemaManager updates
4. More complex deployment

### Long Answer

See [IFIND_FUZZY_MATCHING.md](./IFIND_FUZZY_MATCHING.md) for complete details on:

- How to implement true fuzzy matching with iFind
- iFind syntax and configuration
- Edit distance thresholds
- Migration path from LIKE to iFind

## Commits

1. **22e28a93** - "feat: implement search_entities with IRIS %CONTAINS for fuzzy matching"
   - Initial implementation with %CONTAINS (failed - requires iFind)

2. **e43f76a1** - "feat: add enhanced logging for entity storage debugging"
   - Added diagnostic logging to trace missing entities

3. **a016a24f** - "fix: implement search_entities with SQL LIKE for fuzzy matching"
   - Final working implementation with LIKE-based substring matching

## Testing

### Unit Tests

None yet - requires test fixtures with RAG.Entities table populated.

### Integration Tests

**hipporag2-pipeline HotpotQA evaluation**:
```bash
cd /Users/tdyar/ws/hipporag2-pipeline
SKIP_IRIS_CONTAINER=0 python3 examples/hotpotqa_evaluation.py 1
```

**Expected Behavior**:
- ✅ No `AttributeError` for missing `search_entities`
- ✅ Entity search executes without SQL errors
- ✅ Returns matches for query entities like "Scott Derrickson", "Ed Wood"

**Known Limitations**:
- ⚠️ Still see "20 missing entity IDs" (root cause TBD - separate investigation)
- ⚠️ F1 score may still be 0.000 due to missing LLM function

## API Compatibility

### hipporag2 Expected Signature

```python
matches = storage.search_entities(
    query=query_entity,
    fuzzy=True,
    edit_distance_threshold=2,
    max_results=3
)

for match in matches:
    entity_name = match["entity_name"]
    similarity_score = match.get("similarity_score", 1.0)
```

### Our Implementation

**✅ Signature matches**: All parameters accepted
**✅ Return type matches**: Returns List[Dict[str, Any]]
**✅ Dictionary keys match**: Includes `entity_name`, `similarity_score`
**⚠️ Fuzzy parameter ignored**: Accepts but doesn't implement true fuzzy matching

## Future Work

### Short-term (iris-vector-rag 0.5.x)

1. ✅ **Implement search_entities** with LIKE-based matching
2. ✅ **Add enhanced logging** for FK validation failures
3. ✅ **Document iFind fuzzy matching** requirements and migration path
4. ⏭️ **Investigate root cause** of 20 missing entity IDs during batch storage
5. ⏭️ **Add unit tests** for search_entities method

### Long-term (iris-vector-rag 0.6.x+)

1. **Implement iFind fuzzy matching**:
   - Add iFind index creation to SchemaManager
   - Implement `_search_with_ifind()` method
   - Add hybrid fallback (iFind → LIKE)

2. **Add configuration**:
   ```yaml
   entity_extraction:
     storage:
       fuzzy_matching:
         enabled: true
         method: "ifind"  # or "like"
         edit_distance: 2
   ```

3. **Performance optimization**:
   - Benchmark LIKE vs iFind performance
   - Add caching for frequently searched entities
   - Optimize substring matching with database indexes

## Related Issues

- **hipporag2 F1=0.000**: Missing search_entities method (FIXED ✅)
- **Foreign key failures**: 20 missing entity IDs (IN PROGRESS 🔄)
- **PPR connection mismatch**: Fixed in iris-vector-graph 1.1.7 (UPSTREAM ⏳)

## References

- **Implementation**: `iris_vector_rag/services/storage.py:526-631`
- **Enhanced Logging**: `iris_vector_rag/services/batch_entity_processor.py:112-117, 319-332`
- **iFind Documentation**: [docs/IFIND_FUZZY_MATCHING.md](./IFIND_FUZZY_MATCHING.md)
- **JIRA**: [DP-246668 - iFind Levenshtein Distance](https://usjira.iscinternal.com/browse/DP-246668)
- **Confluence**: [iFind Syntax Revision](https://usconfluence.iscinternal.com/pages/viewpage.action?pageId=421659474)
