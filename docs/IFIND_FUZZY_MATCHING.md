# IRIS iFind Fuzzy Matching Research

## Summary

IRIS **does support true Levenshtein-based fuzzy matching** through the **iFind (InterSystems IRIS Full-Text Search)** feature. However, it requires specific setup and cannot be used with simple SQL LIKE queries.

## Key Findings

### 1. iFind Fuzzy Matching Syntax

**Current iFind Syntax (IRIS 2024.x)**:
```sql
-- Traditional syntax (requires index name)
SELECT * FROM RAG.Entities
WHERE %ID %FIND search_index(entity_name_idx, 'scott', 3)

-- 3 = fuzzy search option with default edit distance threshold
```

**Proposed New Syntax (Future IRIS versions)**:
```sql
-- Cleaner syntax (uses field name directly)
SELECT * FROM RAG.Entities
WHERE entity_name MATCHES fuzzy('scott', 2)  -- max edit distance of 2
```

### 2. Requirements for iFind Fuzzy Matching

To use iFind fuzzy matching, you need:

1. **Create iFind Index** on the column:
   ```sql
   CREATE INDEX entity_name_idx ON RAG.Entities (entity_name)
   FOR %iFind.Index.Basic
   ```

2. **Use %FIND or MATCHES operator** (not LIKE):
   ```sql
   -- Current syntax
   WHERE %ID %FIND search_index(entity_name_idx, 'search_term', 3)

   -- Future syntax
   WHERE entity_name MATCHES fuzzy('search_term', 2)
   ```

3. **iFind Index Types**:
   - `%iFind.Index.Minimal` - Basic tokenization, no transformations
   - `%iFind.Index.Basic` - Tokenization + case normalization
   - `%iFind.Index.Semantic` - Adds stemming, language detection
   - `%iFind.Index.Analytic` - Adds semantic attributes (negation, certainty, etc.)

### 3. Search Options

The traditional `search_index()` function uses integer flags:

- `0` - Regular search (exact match)
- `1` - Stemmed search
- `2` - Phrase search
- `3` - **Fuzzy search** (Levenshtein distance-based)
- `4` - Decompound search

### 4. Edit Distance Thresholds

Fuzzy matching uses **Levenshtein edit distance** with configurable thresholds:

- **Default threshold**: 2 (allows 2 character substitutions/insertions/deletions)
- **Range**: 1-3 (higher values = more permissive matching)
- **Example**: `fuzzy('banona', 2)` matches "banana" (1 substitution)

## Current Implementation Status

### What We Have: SQL LIKE-based Substring Matching

**File**: `iris_vector_rag/services/storage.py:526-631`

```python
def search_entities(
    self,
    query: str,
    fuzzy: bool = True,
    edit_distance_threshold: int = 2,
    max_results: int = 10,
    entity_types: Optional[List[str]] = None,
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search for entities by name with substring matching.

    NOTE: Despite the 'fuzzy' parameter name, this uses SQL LIKE
    for case-insensitive substring matching, NOT true Levenshtein
    edit distance matching.
    """
    sql = f"""
        SELECT entity_id, entity_name, entity_type, source_doc_id, description, confidence
        FROM {self.entities_table}
        WHERE LOWER(entity_name) LIKE LOWER(?)
    """
    params = [f"%{query}%"]
    # ...
```

**Capabilities**:
- ✅ Case-insensitive matching
- ✅ Substring matching ("Scott" matches "Scott Derrickson")
- ❌ NO edit distance tolerance ("Scot" does NOT match "Scott")
- ❌ NO typo tolerance ("banona" does NOT match "banana")

### What We Need: iFind Fuzzy Matching

To implement **true fuzzy matching** with Levenshtein edit distance, we need:

1. **Create iFind index** on `RAG.Entities.entity_name`
2. **Update SQL query** to use `%FIND` or `MATCHES` operator
3. **Configure edit distance threshold** (currently ignored parameter)

## Implementation Options

### Option 1: Require iFind Index (Recommended)

**Pros**:
- True Levenshtein-based fuzzy matching
- Handles typos, misspellings, variations
- Optimized performance with index

**Cons**:
- Requires schema change (create iFind index)
- Requires SchemaManager updates
- More complex deployment

**Implementation**:
```python
def search_entities_ifind(self, query: str, edit_distance: int = 2, ...):
    """Search entities using iFind fuzzy matching."""
    sql = f"""
        SELECT entity_id, entity_name, entity_type, source_doc_id, description, confidence
        FROM {self.entities_table}
        WHERE %ID %FIND search_index(entity_name_idx, ?, 3)
    """
    # 3 = fuzzy search option
    # Edit distance configured in index definition
    cursor.execute(sql, [query])
```

### Option 2: Hybrid Approach (Current + Future)

**Pros**:
- Works today without schema changes (LIKE-based)
- Upgrade path to iFind when available
- Graceful degradation

**Cons**:
- Two codepaths to maintain
- Confusing API (claims "fuzzy" but delivers substring)

**Implementation**:
```python
def search_entities(self, query: str, fuzzy: bool = True, ...):
    """
    Search entities with optional fuzzy matching.

    If iFind index exists: Use true Levenshtein fuzzy matching
    If no iFind index: Fall back to SQL LIKE substring matching
    """
    if self._has_ifind_index():
        return self._search_with_ifind(query, edit_distance_threshold)
    else:
        return self._search_with_like(query)
```

### Option 3: Keep LIKE, Document Limitations (Current)

**Pros**:
- No schema changes required
- Simple implementation
- Works today

**Cons**:
- NOT true fuzzy matching
- Misleading API (parameter names claim "fuzzy")
- Limited capability (no typo tolerance)

**Status**: ✅ **Currently Implemented**

## Recommendations

### Short-term (iris-vector-rag 0.5.x)

1. **Update docstring** to be honest about LIKE limitations:
   ```python
   """
   Search for entities by name with case-insensitive substring matching.

   NOTE: This uses SQL LIKE for substring matching, NOT true fuzzy matching
   with edit distance. The 'fuzzy' and 'edit_distance_threshold' parameters
   are accepted for API compatibility with hipporag2 but are not implemented.

   Examples:
       - "Scott" WILL match "Scott Derrickson" ✅
       - "Scot" will NOT match "Scott" ❌ (no typo tolerance)
       - "banona" will NOT match "banana" ❌ (no edit distance)
   """
   ```

2. **Add iFind tracking issue** for future implementation

3. **Keep current LIKE-based implementation** (works for substring matching use case)

### Long-term (iris-vector-rag 0.6.x+)

1. **Add iFind index creation** to SchemaManager:
   ```python
   class SchemaManager:
       def ensure_ifind_index(self, table_name: str, column_name: str):
           """Create iFind index for fuzzy matching support."""
           sql = f"""
               CREATE INDEX IF NOT EXISTS {table_name}_{column_name}_idx
               ON {table_name} ({column_name})
               FOR %iFind.Index.Basic
           """
           cursor.execute(sql)
   ```

2. **Implement hybrid search** (Option 2 above)

3. **Add configuration** to enable/disable iFind:
   ```yaml
   entity_extraction:
     storage:
       fuzzy_matching:
         enabled: true
         method: "ifind"  # or "like" for substring
         edit_distance: 2
   ```

## References

- **Confluence**: [iFind Syntax Revision](https://usconfluence.iscinternal.com/pages/viewpage.action?pageId=421659474)
- **JIRA**: [DP-246668 - iFind Levenshtein Distance](https://usjira.iscinternal.com/browse/DP-246668)
- **IRIS Docs**: [Using iFind](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSQLSRCH)
- **Search Options**: [iFind Search Options](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSQLSRCH_txtsrch_select)

## Related Issues

- **hipporag2-pipeline Issue**: F1=0.000 score due to missing fuzzy matching
- **Missing Entities**: 20 entities not found during retrieval (e.g., "Ed Wood", "Johnny Depp")
- **Foreign Key Failures**: 30 orphaned relationships due to missing entity search

## Next Steps

1. ✅ **Document findings** (this file)
2. ⏭️ **Update search_entities docstring** to clarify LIKE limitations
3. ⏭️ **Create tracking issue** for iFind implementation
4. ⏭️ **Test current LIKE implementation** with hipporag2 (validate it works for substring matching)
5. ⏭️ **Plan iFind migration** for iris-vector-rag 0.6.0
