# IRIS Vector Search Bug Summary

## Executive Summary

We have identified a critical bug in InterSystems IRIS SQL parser that prevents the use of vector search functions with VARCHAR columns. The bug causes TO_VECTOR() function calls to fail with a "colon found" error when using quoted 'DOUBLE', even when no colons are present in the data.

## Bug Details

### Primary Issue
- **Error**: `< ) expected, : found ^SELECT id , name , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>`
- **Cause**: IRIS SQL parser incorrectly interprets the string literal `'DOUBLE'` in `TO_VECTOR(embedding, 'DOUBLE', 3)` as containing a parameter marker
- **Impact**: All vector search operations on VARCHAR columns fail when using quoted 'DOUBLE'

### Test Results

1. **TO_VECTOR with quoted 'DOUBLE'** - ❌ FAILS
   ```sql
   SELECT TO_VECTOR(embedding, 'DOUBLE', 3) FROM table
   -- Error: colon found
   ```

2. **TO_VECTOR without quotes** - ✅ WORKS
   ```sql
   SELECT TO_VECTOR(embedding, DOUBLE, 3) FROM table
   -- Success!
   ```

3. **Native VECTOR columns** - ✅ WORKS PERFECTLY
   ```sql
   -- V2 tables with VECTOR(DOUBLE, 384) columns
   SELECT VECTOR_COSINE(document_embedding_vector, TO_VECTOR('...', DOUBLE, 384))
   FROM RAG.SourceDocuments_V2
   -- Success! No issues at all
   ```

## Solutions

### 1. Immediate Workaround (for VARCHAR columns)
Use unquoted DOUBLE in TO_VECTOR:
```sql
-- Instead of: TO_VECTOR(embedding, 'DOUBLE', 384)
-- Use: TO_VECTOR(embedding, DOUBLE, 384)
```

### 2. Recommended Solution (Already Implemented)
Use the V2 tables with native VECTOR columns:
- `RAG.SourceDocuments_V2` - 99,990 documents already migrated
- `RAG.DocumentChunks_V2` - Ready for use
- `RAG.DocumentTokenEmbeddings_V2` - Ready for use

### 3. Current BasicRAG Workaround
BasicRAG successfully works around this bug by:
1. Loading embeddings as VARCHAR strings
2. Parsing embeddings in Python
3. Calculating cosine similarity in application code
4. Avoiding all IRIS vector functions

## Files Created

1. `test_iris_vector_bug_pure_sql.sql` - Pure SQL demonstration
2. `test_iris_vector_bug_dbapi.py` - Python script using intersystems-irispython
3. `basic_rag/pipeline_vector_fix.py` - BasicRAG implementation with unquoted DOUBLE workaround
4. `docs/IRIS_VECTOR_SEARCH_BUG_SUMMARY.md` - This summary

## Recommendation

The V2 tables with native VECTOR columns are the best solution:
- No parser bugs
- Better performance with HNSW indexes
- Direct vector operations without TO_VECTOR()
- Already populated with 99,990 documents

For immediate use with VARCHAR columns, use the unquoted DOUBLE workaround in TO_VECTOR().