# GraphRAG Data Ingestion Investigation Report

## Executive Summary

The investigation into GraphRAG data ingestion has revealed that while `make validate-graphrag` passes, there are **significant underlying issues** with the data ingestion process that prevent proper population of nodes, entities, and relationships tables.

## Key Findings

### 1. **Tables Exist But Are Not Being Populated Correctly**

**Current State:**
- ✅ `RAG.DocumentEntities` exists with 144 rows (from previous ingestion attempts)
- ❌ `RAG.EntityRelationships` exists but has **0 rows** (completely empty)
- ❌ `RAG.Entities` exists but has **0 rows** (archived table, unused)
- ❌ `RAG.Relationships` exists but has **0 rows** (archived table, unused)

**Critical Issue:** The relationships table is completely empty, meaning no graph connections are being created.

### 2. **Multiple Critical Bugs in Current Implementation**

#### Bug #1: Vector Embedding Storage Failure
```
[SQLCODE: <-104>:<Field validation failed in INSERT>]
Field 'RAG.DocumentEntities.embedding' (value '367FAF0E49796AEF039CFA40D891C895@$vector') failed validation
```

**Root Cause:** The current implementation tries to store embeddings using `TO_VECTOR(?)` with a string representation, but the database expects a different format.

#### Bug #2: Document Storage SQL Syntax Error
```
[SQLCODE: <-1>:<Invalid SQL statement>]
UPDATE expected, IDENTIFIER (REPLACE) found ^INSERT OR REPLACE
```

**Root Cause:** The storage layer uses `INSERT OR REPLACE` syntax which is not supported by IRIS SQL.

### 3. **Schema Inconsistencies**

The investigation revealed **two different table schemas** being used:

#### Current Implementation Schema (iris_rag/pipelines/graphrag.py):
- `RAG.DocumentEntities` - Links documents to extracted entities
- `RAG.EntityRelationships` - Stores relationships between entities

#### Archived Implementation Schema (archived_pipelines/):
- `RAG.Entities` - Master entity table with embeddings
- `RAG.Relationships` - Relationships between entities with proper foreign keys

### 4. **Comparison with "OLD code that did all that very well"**

The archived implementations (`pipeline_original.py` and `pipeline_jdbc_fixed.py`) show a **much more sophisticated approach**:

#### Archived Implementation Advantages:
1. **Proper Entity Management:**
   - Uses semantic similarity for entity matching
   - Implements proper graph traversal algorithms
   - Has fallback mechanisms for vector search

2. **Better Table Design:**
   - `RAG.Entities` table with proper entity metadata
   - `RAG.Relationships` with foreign key constraints
   - Support for entity types (PERSON, ORG, DISEASE, DRUG, etc.)

3. **Robust Query Processing:**
   - Multi-step entity finding (keyword + semantic)
   - Graph traversal with depth limits
   - Proper JDBC stream handling

#### Current Implementation Issues:
1. **Simplistic Entity Extraction:**
   - Only extracts capitalized words > 3 characters
   - No semantic understanding or entity typing
   - No proper entity deduplication

2. **Broken Storage Layer:**
   - Vector embedding storage fails
   - SQL syntax incompatibilities
   - No error handling for storage failures

3. **No Graph Traversal:**
   - Relationships are created but never used for retrieval
   - No knowledge graph navigation logic
   - Falls back to simple vector search

## Why `make validate-graphrag` Passes Despite These Issues

The validation likely passes because:
1. **Tables exist** (structure validation passes)
2. **Some entities exist** (144 rows in DocumentEntities from previous attempts)
3. **Basic query functionality works** (falls back to vector search when graph fails)
4. **No deep validation** of relationship population or graph traversal

## Recommendations

### Immediate Fixes Needed:

1. **Fix Vector Embedding Storage**
   - Update `_store_entities()` method to use proper IRIS vector format
   - Test with actual vector data

2. **Fix Document Storage SQL**
   - Replace `INSERT OR REPLACE` with IRIS-compatible syntax
   - Use proper UPSERT patterns for IRIS

3. **Fix Relationship Population**
   - Debug why `_store_relationships()` is not being called or failing silently
   - Ensure relationships are actually created during ingestion

### Long-term Improvements:

1. **Consider Reverting to Archived Implementation**
   - The archived `pipeline_jdbc_fixed.py` appears much more robust
   - Has proper entity management and graph traversal
   - Better error handling and fallback mechanisms

2. **Implement Proper Entity Extraction**
   - Use NER models instead of simple capitalization rules
   - Add entity typing and semantic matching
   - Implement entity deduplication

3. **Add Comprehensive Testing**
   - Test actual graph traversal functionality
   - Verify relationship creation and usage
   - Test with real PMC data at scale

## Conclusion

The user's skepticism about GraphRAG data ingestion is **completely justified**. While the validation passes, the current implementation has fundamental flaws that prevent proper graph construction and traversal. The "OLD code" in the archived implementations was indeed much more sophisticated and functional.

**Recommendation:** Either fix the critical bugs in the current implementation or revert to the archived `pipeline_jdbc_fixed.py` implementation, which appears to be a working, well-designed GraphRAG system.