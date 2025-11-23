# Research: Fuzzy Entity Matching for EntityStorageAdapter

**Feature**: 061-implement-fuzzy-matching
**Date**: 2025-01-15
**Status**: Complete

## Research Questions

### 1. IRIS iFind Full-Text Search Capabilities

**Question**: What are the fuzzy search capabilities of IRIS iFind and how can they be leveraged for entity name matching?

**Findings**:
- IRIS iFind supports four index types: Minimal, Basic, Semantic, and Analytic
- Fuzzy search is supported on Basic, Semantic, and Analytic indexes
- Uses Levenshtein distance algorithm (minimum single-character edits: insertions, deletions, substitutions)
- Default edit distance is 2 characters, configurable (1-3 typical range)
- Can match spelling variations: "color" vs "colour", "analyze" vs "analyse"
- Can match grammatical forms: "color" vs "colors" (singular/plural) with edit_distance=1

**Decision**: Use IRIS iFind Levenshtein distance via SQL functions for fuzzy matching

**Rationale**:
- Native IRIS capability means no external dependencies
- Database-side fuzzy matching reduces data transfer (no need to retrieve all entities to Python)
- Better performance for 100K+ entity knowledge graphs
- Proven technology (InterSystems has extensive iFind documentation)

**Alternatives Considered**:
1. **Python Levenshtein library** (rejected):
   - Requires retrieving all entities from database to Python
   - Higher memory usage and latency
   - Adds external dependency

2. **Soundex phonetic matching** (rejected):
   - Designed for phonetic similarity (e.g., "Smith" vs "Smyth")
   - Not suitable for descriptor matching (e.g., "Scott Derrickson" vs "Scott Derrickson director")
   - Would miss exact substring matches

3. **Vector embeddings similarity** (rejected):
   - Different use case (semantic similarity, not string similarity)
   - Higher computational cost
   - Requires separate embedding model

**Sources**:
- Perplexity search: "InterSystems IRIS iFind full-text search capabilities fuzzy matching"
- InterSystems IRIS iFind documentation (inferred from spec research findings)

### 2. iFind Index Requirements

**Question**: Do we need to create new iFind indexes on RAG.Entities table, or can we use existing indexes?

**Findings**:
- IRIS SQL provides `$SYSTEM.SQL.Functions.LEVENSHTEIN()` built-in function
- Levenshtein distance can be calculated without iFind indexes
- Standard SQL indexes on entity_name and entity_type columns already exist
- Creating iFind indexes marked as "Out of Scope" in feature spec

**Decision**: Use existing RAG.Entities table with standard SQL indexes, no iFind index creation

**Rationale**:
- Meets feature scope requirements (spec explicitly marks index creation as out of scope)
- Levenshtein SQL function works without iFind indexes
- Reduces implementation complexity
- Planning phase will determine if existing GraphRAG indexes can be reused (future optimization)

**Alternatives Considered**:
1. **Create iFind Basic index** (rejected - out of scope):
   - Would enable full-text search capabilities
   - Adds schema migration complexity
   - Not required for Levenshtein distance calculation

2. **Create iFind Semantic index** (rejected - out of scope):
   - Would enable semantic search features
   - Higher overhead than Basic index
   - Overkill for string matching use case

**Implementation Note**: Use `$SYSTEM.SQL.Functions.LEVENSHTEIN(query, entity_name)` in SQL WHERE clause to calculate edit distance.

### 3. Fuzzy Matching Strategy

**Question**: What combination of matching strategies will best handle both descriptors (e.g., "director") and typos (e.g., "Scot" vs "Scott")?

**Findings**:
- Substring matching (LIKE '%query%') catches entities with descriptors
- Levenshtein distance catches typos and spelling variations
- Combining both strategies provides comprehensive matching
- FR-017 explicitly requires "both substring matching (LIKE pattern) and Levenshtein distance-based fuzzy matching as complementary search strategies"

**Decision**: Implement hybrid approach with substring matching + Levenshtein distance

**Rationale**:
- Substring matching handles descriptor case: "Scott Derrickson" matches "Scott Derrickson director"
- Levenshtein handles typo case: "Scot Derrickson" matches "Scott Derrickson"
- Constitutional requirement (FR-017) mandates both strategies
- No performance penalty (both can be evaluated in same SQL query)

**SQL Implementation Strategy**:
```sql
-- Fuzzy matching (both substring + Levenshtein)
SELECT entity_id, entity_name, entity_type, confidence,
       $SYSTEM.SQL.Functions.LEVENSHTEIN(?, LOWER(entity_name)) AS edit_distance,
       1 - ($SYSTEM.SQL.Functions.LEVENSHTEIN(?, LOWER(entity_name)) /
            GREATEST(LENGTH(?), LENGTH(entity_name))) AS similarity_score
FROM RAG.Entities
WHERE (LOWER(entity_name) LIKE ? OR
       $SYSTEM.SQL.Functions.LEVENSHTEIN(?, LOWER(entity_name)) <= ?)
  AND (entity_type IN (?, ?, ...) OR ? IS NULL)
ORDER BY
    CASE WHEN LOWER(entity_name) = ? THEN 0 ELSE 1 END,  -- Exact matches first
    edit_distance ASC,                                     -- Then by edit distance
    LENGTH(entity_name) ASC                                -- Then by name length
FETCH FIRST ? ROWS ONLY
```

**Alternatives Considered**:
1. **Substring matching only** (rejected):
   - Would miss typo corrections
   - Fails FR-003 requirement for typo handling

2. **Levenshtein only** (rejected):
   - Would miss entities with long descriptors
   - Fails FR-002 requirement for substring matching

3. **Separate queries for each strategy** (rejected):
   - Higher latency (two round trips)
   - More complex result merging logic
   - Single query approach is cleaner

### 4. Similarity Scoring Approach

**Question**: How should we calculate a normalized similarity score from Levenshtein edit distance?

**Findings**:
- Raw edit distance is not normalized (range varies by string length)
- FR-006 requires "similarity score for each fuzzy match result"
- Need 0.0-1.0 range where 1.0 = perfect match, 0.0 = no similarity
- Industry standard: Levenshtein similarity = `1 - (edit_distance / max_length)`

**Decision**: Calculate similarity score as `1 - (edit_distance / max(len(query), len(entity_name)))`

**Rationale**:
- Normalized to 0.0-1.0 range (meets FR-006)
- Higher score = better match (intuitive)
- Accounts for string length differences
- Standard formula used by many fuzzy matching libraries

**Formula Breakdown**:
```
edit_distance = number of character changes needed
max_length = max(len(query), len(entity_name))
similarity_score = 1 - (edit_distance / max_length)

Examples:
- "Scott" vs "Scott" → edit_distance=0 → similarity=1.0 (perfect match)
- "Scot" vs "Scott" → edit_distance=1 → similarity=1-(1/5)=0.8
- "Scott Derrickson" vs "Scott Derrickson director"
  → edit_distance=9 → similarity=1-(9/27)=0.67
```

**Alternatives Considered**:
1. **Raw edit distance** (rejected):
   - Not normalized (range 0 to infinity)
   - Harder to set meaningful thresholds
   - Fails FR-006 requirement for similarity score

2. **Jaro-Winkler distance** (rejected):
   - More complex algorithm
   - Not available as IRIS built-in function
   - Would require Python implementation (slower)

3. **Dice coefficient** (rejected):
   - Token-based, not character-based
   - Not suitable for single-word matching
   - Less intuitive for typo detection

**Implementation Note**: Calculate in SQL using `1 - (edit_distance / GREATEST(LENGTH(query), LENGTH(entity_name)))`.

### 5. Performance Optimization Strategy

**Question**: How can we meet the <50ms fuzzy search requirement for 100K entities without adding new indexes?

**Findings**:
- FR-011 requires <50ms for 100K entities
- FR-012 requires <10ms for exact match (indexed lookup)
- Existing indexes on entity_name and entity_type columns
- SQL `FETCH FIRST N ROWS ONLY` clause limits result set size
- Case-insensitive matching via `LOWER()` function

**Decision**: Use indexed entity_name and entity_type columns with FETCH FIRST N ROWS ONLY

**Rationale**:
- Existing indexes enable fast entity_name lookups
- FETCH FIRST limits result set (default max_results=10)
- LOWER() for case-insensitive matching (FR-010 requirement)
- No schema changes required
- Meets performance requirements based on IRIS query optimizer

**Optimization Techniques**:
1. **Use existing indexes**: entity_name and entity_type columns already indexed
2. **Limit result set**: `FETCH FIRST {max_results} ROWS ONLY` (default 10)
3. **Filter early**: Apply entity_type filter in WHERE clause (indexed)
4. **Short-circuit exact matches**: Check `entity_name = query` first in ORDER BY
5. **Case-insensitive via LOWER()**: Standard SQL function with index support

**Query Plan**:
```sql
-- Exact match (indexed lookup, <10ms)
WHERE LOWER(entity_name) = LOWER(?)
INDEX: entity_name

-- Fuzzy match (substring + Levenshtein, <50ms)
WHERE (LOWER(entity_name) LIKE ? OR LEVENSHTEIN(...) <= ?)
  AND entity_type IN (...)
INDEX: entity_name, entity_type
LIMIT: FETCH FIRST 10 ROWS ONLY
```

**Alternatives Considered**:
1. **Query result caching** (rejected - premature optimization):
   - Adds complexity (cache invalidation)
   - Memory overhead for cache storage
   - Queries are already fast enough

2. **Materialized views** (rejected - unnecessary complexity):
   - Requires schema changes
   - Adds maintenance overhead
   - No significant performance gain

3. **Full-table scan with Python filtering** (rejected - too slow):
   - Would miss performance requirements
   - Higher data transfer cost
   - Defeats purpose of database indexes

**Performance Validation**: Integration tests will validate actual performance on 10K entities (acceptance scenario #5).

### 6. BUG-001: GraphRAG Pipeline Time Import

**Question**: Does the GraphRAG pipeline have the missing `time` module import that causes UnboundLocalError?

**Findings**:
- Checked `iris_vector_rag/pipelines/graphrag.py`
- Line 8 contains: `import time`
- No UnboundLocalError possible with proper import
- Bug report describes missing import, but current codebase has it

**Decision**: BUG-001 is ALREADY FIXED in the current codebase

**Rationale**:
- Import statement verified at `iris_vector_rag/pipelines/graphrag.py:8`
- Code inspection shows `import time` at top of file
- No implementation work required for this bug
- Only validation test needed to ensure GraphRAG ingest() works

**Implementation Action**:
- Skip bug fix implementation (already fixed)
- Add integration test: `test_graphrag_pipeline_integration.py`
- Test validates GraphRAG ingest() completes without UnboundLocalError
- Test satisfies acceptance scenario #9 and success metric #8

**Evidence**:
```python
# iris_vector_rag/pipelines/graphrag.py:1-20
"""
GraphRAG Pipeline implementation using knowledge graph traversal.

PRODUCTION-HARDENED VERSION: No fallbacks, fail-hard validation, integrated entity extraction.
"""

import logging
import time  # ✅ PRESENT AT LINE 8
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..config.manager import ConfigurationManager
from ..core.base import RAGPipeline
# ... (rest of imports)
```

## Research Summary

All technical unknowns resolved. Key decisions:

1. **IRIS iFind Levenshtein distance**: Use native SQL functions for fuzzy matching
2. **No new indexes**: Leverage existing entity_name and entity_type indexes
3. **Hybrid strategy**: Combine substring (LIKE) + Levenshtein for comprehensive matching
4. **Normalized similarity**: `1 - (edit_distance / max_length)` for 0.0-1.0 scoring
5. **Performance optimization**: Indexed queries + FETCH FIRST for <50ms requirement
6. **BUG-001 status**: Already fixed in codebase, only needs validation test

All NEEDS CLARIFICATION from Technical Context resolved. Ready for Phase 1 (Design & Contracts).

## Next Steps

Phase 1 will generate:
- data-model.md: EntitySearchQuery and EntitySearchResult structures
- contracts/: 5 contract test files with 15+ test cases
- quickstart.md: Usage guide for search_entities() method
